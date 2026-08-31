#include "qwen35x/cpu/executor.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace qwen35x::cpu {

namespace {

inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  __asm__ __volatile__("pause" ::: "memory");
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

struct RowRange {
  std::size_t begin = 0;
  std::size_t end = 0;
};

[[nodiscard]] RowRange static_row_range(
  const std::size_t row_count,
  const std::size_t partition_count,
  const std::size_t partition_index) noexcept {
  const std::size_t rows_per_partition = row_count / partition_count;
  const std::size_t remainder = row_count % partition_count;
  const std::size_t extra_before = std::min(partition_index, remainder);
  const std::size_t begin = partition_index * rows_per_partition + extra_before;
  return RowRange{
    begin,
    begin + rows_per_partition + (partition_index < remainder ? 1U : 0U),
  };
}

class AtomicFlagGuard final {
public:
  explicit AtomicFlagGuard(std::atomic_flag & flag) noexcept : flag_(flag) {}
  ~AtomicFlagGuard() {
    flag_.clear(std::memory_order_release);
  }

  AtomicFlagGuard(const AtomicFlagGuard &) = delete;
  AtomicFlagGuard & operator=(const AtomicFlagGuard &) = delete;

private:
  std::atomic_flag & flag_;
};

struct Q8MatvecJob {
  const Q8_0Block * matrix = nullptr;
  const Q8_0Block * vector = nullptr;
  float * output = nullptr;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
};

struct Q8MatmulJob {
  const Q8_0Block * matrix = nullptr;
  const Q8_0Block * vectors = nullptr;
  float * output = nullptr;
  std::size_t total_row_count = 0;
  std::size_t vector_count = 0;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
  const float * vector_scales = nullptr;
  const float * matrix_scales = nullptr;
};

struct Q4MatvecJob {
  const Q4_0Block * matrix = nullptr;
  const Q8_0Block * vector = nullptr;
  float * output = nullptr;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
};

struct Q4MatmulJob {
  const Q4_0Block * matrix = nullptr;
  const Q8_0Block * vectors = nullptr;
  float * output = nullptr;
  std::size_t total_row_count = 0;
  std::size_t vector_count = 0;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
  const float * vector_scales = nullptr;
  const float * matrix_scales = nullptr;
};

void run_q8_matvec_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q8MatvecJob *>(opaque_context);
  const Q8_0Block * row_matrix = job.matrix;
  if (job.blocks_per_row != 0) {
    row_matrix += row_begin * job.blocks_per_row;
  }
  qwen35x::cpu::q8_0_matvec(
    row_matrix,
    job.vector,
    job.output + row_begin,
    row_end - row_begin,
    job.blocks_per_row,
    job.backend);
}

void run_q8_matmul_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q8MatmulJob *>(opaque_context);
  const std::size_t local_rows = row_end - row_begin;
  qwen35x::cpu::q8_0_matmul(
    job.matrix + row_begin * job.blocks_per_row,
    job.vectors,
    job.output + row_begin,
    local_rows,
    job.vector_count,
    job.blocks_per_row,
    job.total_row_count,
    job.backend,
    job.vector_scales,
    job.matrix_scales != nullptr
      ? job.matrix_scales + row_begin * job.blocks_per_row
      : nullptr);
}

void run_q4_matvec_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q4MatvecJob *>(opaque_context);
  const Q4_0Block * row_matrix = job.matrix;
  if (job.blocks_per_row != 0) {
    row_matrix += row_begin * job.blocks_per_row;
  }
  qwen35x::cpu::q4_0_matvec_q8_0(
    row_matrix, job.vector, job.output + row_begin,
    row_end - row_begin, job.blocks_per_row, job.backend);
}

void run_q4_matmul_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q4MatmulJob *>(opaque_context);
  qwen35x::cpu::q4_0_matmul_q8_0(
    job.matrix + row_begin * job.blocks_per_row,
    job.vectors,
    job.output + row_begin,
    row_end - row_begin,
    job.vector_count,
    job.blocks_per_row,
    job.total_row_count,
    job.backend,
    job.vector_scales,
    job.matrix_scales != nullptr
      ? job.matrix_scales + row_begin * job.blocks_per_row
      : nullptr);
}

} // namespace

class CpuExecutor::Impl final {
public:
  Impl(
    const std::size_t thread_count,
    const std::size_t min_parallel_rows,
    const std::size_t spin_count) noexcept
      : thread_count_(thread_count), min_parallel_rows_(min_parallel_rows), spin_count_(spin_count) {}

  ~Impl() {
    shutdown();
  }

  Impl(const Impl &) = delete;
  Impl & operator=(const Impl &) = delete;

  [[nodiscard]] bool start(std::error_code & error_code) noexcept {
    error_code.clear();
    try {
      workers_.reserve(thread_count_ - 1);
      for (std::size_t worker_index = 1; worker_index < thread_count_; ++worker_index) {
        workers_.emplace_back([this, worker_index] { worker_loop(worker_index); });
      }
      return true;
    } catch (const std::system_error & error) {
      error_code = error.code();
    } catch (const std::bad_alloc &) {
      error_code = std::make_error_code(std::errc::not_enough_memory);
    } catch (...) {
      error_code = std::make_error_code(std::errc::resource_unavailable_try_again);
    }
    shutdown();
    return false;
  }

  void shutdown() noexcept {
    {
      const std::lock_guard<std::mutex> lock(job_mutex_);
      stopping_.store(true, std::memory_order_release);
    }
    job_available_.notify_all();
    for (std::thread & worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers_.clear();
  }

  [[nodiscard]] std::size_t thread_count() const noexcept {
    return thread_count_;
  }

  [[nodiscard]] std::size_t worker_thread_count() const noexcept {
    return thread_count_ - 1;
  }

  [[nodiscard]] std::size_t min_parallel_rows() const noexcept {
    return min_parallel_rows_;
  }

  [[nodiscard]] CpuExecutorStatus parallel_for_rows(
    const std::size_t row_count,
    CpuRowRangeTask task,
    void * context) noexcept {
    if (row_count == 0) {
      return CpuExecutorStatus::ok;
    }
    if (task == nullptr) {
      return CpuExecutorStatus::invalid_argument;
    }
    if (job_in_progress_.test_and_set(std::memory_order_acquire)) {
      return CpuExecutorStatus::busy;
    }
    const AtomicFlagGuard job_guard(job_in_progress_);

    if (thread_count_ == 1 || row_count < min_parallel_rows_) {
      task(context, 0, row_count);
      return CpuExecutorStatus::ok;
    }

    {
      const std::lock_guard<std::mutex> lock(job_mutex_);
      task_ = task;
      task_context_ = context;
      job_row_count_ = row_count;
      completed_workers_.store(0, std::memory_order_relaxed);
      job_generation_.fetch_add(1, std::memory_order_release);
    }
    job_available_.notify_all();

    const RowRange main_range = static_row_range(row_count, thread_count_, 0);
    if (main_range.begin != main_range.end) {
      task(context, main_range.begin, main_range.end);
    }

    bool workers_complete = false;
    for (std::size_t spin = 0; spin < spin_count_; ++spin) {
      if (completed_workers_.load(std::memory_order_acquire) == workers_.size()) {
        workers_complete = true;
        break;
      }
      cpu_relax();
    }
    {
      std::unique_lock<std::mutex> lock(job_mutex_);
      if (!workers_complete) {
        job_complete_.wait(lock, [this] {
          return completed_workers_.load(std::memory_order_acquire) == workers_.size();
        });
      }
      task_ = nullptr;
      task_context_ = nullptr;
      job_row_count_ = 0;
    }
    return CpuExecutorStatus::ok;
  }

private:
  void worker_loop(const std::size_t worker_index) noexcept {
    std::uint64_t observed_generation = 0;
    while (true) {
      bool work_available = false;
      for (std::size_t spin = 0; spin < spin_count_; ++spin) {
        if (stopping_.load(std::memory_order_acquire)) {
          return;
        }
        if (job_generation_.load(std::memory_order_acquire) != observed_generation) {
          work_available = true;
          break;
        }
        cpu_relax();
      }

      CpuRowRangeTask task = nullptr;
      void * context = nullptr;
      std::size_t row_count = 0;
      if (!work_available) {
        std::unique_lock<std::mutex> lock(job_mutex_);
        job_available_.wait(lock, [this, observed_generation] {
          return stopping_.load(std::memory_order_acquire) ||
            job_generation_.load(std::memory_order_acquire) != observed_generation;
        });
        if (stopping_.load(std::memory_order_acquire)) {
          return;
        }
      }
      observed_generation = job_generation_.load(std::memory_order_acquire);
      task = task_;
      context = task_context_;
      row_count = job_row_count_;

      const RowRange range = static_row_range(row_count, thread_count_, worker_index);
      if (range.begin != range.end) {
        task(context, range.begin, range.end);
      }

      const std::size_t completed = completed_workers_.fetch_add(1, std::memory_order_release) + 1;
      if (completed == workers_.size()) {
        job_complete_.notify_one();
      }
    }
  }

  const std::size_t thread_count_;
  const std::size_t min_parallel_rows_;
  const std::size_t spin_count_;
  std::vector<std::thread> workers_;

  std::mutex job_mutex_;
  std::condition_variable job_available_;
  std::condition_variable job_complete_;
  std::atomic<bool> stopping_{false};
  alignas(64) std::atomic<std::uint64_t> job_generation_{0};
  alignas(64) std::atomic<std::size_t> completed_workers_{0};
  CpuRowRangeTask task_ = nullptr;
  void * task_context_ = nullptr;
  std::size_t job_row_count_ = 0;
  std::atomic_flag job_in_progress_ = ATOMIC_FLAG_INIT;
};

const char * cpu_executor_status_name(const CpuExecutorStatus status) noexcept {
  switch (status) {
    case CpuExecutorStatus::ok:
      return "ok";
    case CpuExecutorStatus::invalid_argument:
      return "invalid_argument";
    case CpuExecutorStatus::busy:
      return "busy";
  }
  return "unknown";
}

std::unique_ptr<CpuExecutor> CpuExecutor::create(
  CpuExecutorConfig config,
  std::error_code & error_code) noexcept {
  error_code.clear();
  if (config.thread_count == 0) {
    config.thread_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  }

  try {
    auto impl = std::make_unique<Impl>(config.thread_count, config.min_parallel_rows, config.spin_count);
    if (!impl->start(error_code)) {
      return nullptr;
    }
    return std::unique_ptr<CpuExecutor>(new CpuExecutor(std::move(impl)));
  } catch (const std::bad_alloc &) {
    error_code = std::make_error_code(std::errc::not_enough_memory);
  } catch (...) {
    error_code = std::make_error_code(std::errc::resource_unavailable_try_again);
  }
  return nullptr;
}

CpuExecutor::CpuExecutor(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

CpuExecutor::~CpuExecutor() = default;

std::size_t CpuExecutor::thread_count() const noexcept {
  return impl_->thread_count();
}

std::size_t CpuExecutor::worker_thread_count() const noexcept {
  return impl_->worker_thread_count();
}

std::size_t CpuExecutor::min_parallel_rows() const noexcept {
  return impl_->min_parallel_rows();
}

CpuExecutorStatus CpuExecutor::parallel_for_rows(
  const std::size_t row_count,
  const CpuRowRangeTask task,
  void * context) noexcept {
  return impl_->parallel_for_rows(row_count, task, context);
}

CpuExecutorStatus CpuExecutor::q8_0_matvec(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend) noexcept {
  if (row_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512) ||
      output == nullptr ||
      (blocks_per_row != 0 && (matrix == nullptr || vector == nullptr)) ||
      (blocks_per_row != 0 &&
       row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row)) {
    return CpuExecutorStatus::invalid_argument;
  }

  Q8MatvecJob job{
    matrix,
    vector,
    output,
    blocks_per_row,
    q8_0_resolve_backend(backend),
  };
  return parallel_for_rows(row_count, run_q8_matvec_rows, &job);
}

CpuExecutorStatus CpuExecutor::q8_0_matmul(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  if (row_count == 0 || vector_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512) ||
      matrix == nullptr || vectors == nullptr || output == nullptr ||
      blocks_per_row == 0 ||
      row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / row_count) {
    return CpuExecutorStatus::invalid_argument;
  }
  Q8MatmulJob job{
    matrix,
    vectors,
    output,
    row_count,
    vector_count,
    blocks_per_row,
    q8_0_resolve_backend(backend),
    vector_scales,
    matrix_scales,
  };
  return parallel_for_rows(row_count, run_q8_matmul_rows, &job);
}

CpuExecutorStatus CpuExecutor::q4_0_matvec_q8_0(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend) noexcept {
  if (row_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512) ||
      output == nullptr ||
      (blocks_per_row != 0 && (matrix == nullptr || vector == nullptr)) ||
      (blocks_per_row != 0 &&
       row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row)) {
    return CpuExecutorStatus::invalid_argument;
  }
  Q4MatvecJob job{
    matrix, vector, output, blocks_per_row, q8_0_resolve_backend(backend),
  };
  return parallel_for_rows(row_count, run_q4_matvec_rows, &job);
}

CpuExecutorStatus CpuExecutor::q4_0_matmul_q8_0(
  const Q4_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  if (row_count == 0 || vector_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512) ||
      matrix == nullptr || vectors == nullptr || output == nullptr ||
      blocks_per_row == 0 ||
      row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / row_count) {
    return CpuExecutorStatus::invalid_argument;
  }
  Q4MatmulJob job{
    matrix, vectors, output, row_count, vector_count, blocks_per_row,
    q8_0_resolve_backend(backend), vector_scales, matrix_scales,
  };
  return parallel_for_rows(row_count, run_q4_matmul_rows, &job);
}

} // namespace qwen35x::cpu
