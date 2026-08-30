#include "qwen35x/cpu/executor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

using qwen35x::cpu::CpuExecutor;
using qwen35x::cpu::CpuExecutorStatus;
using qwen35x::cpu::Q8_0Backend;
using qwen35x::cpu::Q8_0Block;
using qwen35x::cpu::q8_0_values_per_block;

bool expect(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "cpu executor test failure: " << message << '\n';
  }
  return condition;
}

struct RangeContext {
  std::array<std::atomic_uint, 32> hits{};
  std::array<std::size_t, 8> begins{};
  std::array<std::size_t, 8> ends{};
  std::array<std::thread::id, 8> thread_ids{};
  std::atomic_size_t call_count = 0;
};

void record_ranges(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & context = *static_cast<RangeContext *>(opaque_context);
  const std::size_t slot = context.call_count.fetch_add(1, std::memory_order_relaxed);
  if (slot < context.begins.size()) {
    context.begins[slot] = row_begin;
    context.ends[slot] = row_end;
    context.thread_ids[slot] = std::this_thread::get_id();
  }
  for (std::size_t row = row_begin; row < row_end; ++row) {
    context.hits[row].fetch_add(1, std::memory_order_relaxed);
  }
}

bool test_static_partitions(CpuExecutor & executor) {
  constexpr std::size_t row_count = 17;
  RangeContext context;
  const CpuExecutorStatus status = executor.parallel_for_rows(row_count, record_ranges, &context);
  bool ok = expect(status == CpuExecutorStatus::ok, "parallel job failed") &&
    expect(context.call_count.load() == 4, "expected one range per participating thread");
  for (std::size_t row = 0; row < row_count; ++row) {
    ok = expect(context.hits[row].load() == 1, "row was not visited exactly once") && ok;
  }

  std::array<std::pair<std::size_t, std::size_t>, 4> ranges{};
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    ranges[index] = {context.begins[index], context.ends[index]};
  }
  std::sort(ranges.begin(), ranges.end());
  constexpr std::array<std::pair<std::size_t, std::size_t>, 4> expected{{
    {0, 5}, {5, 9}, {9, 13}, {13, 17},
  }};
  ok = expect(ranges == expected, "static contiguous partitions differ") && ok;

  std::array<std::thread::id, 4> thread_ids{};
  std::copy_n(context.thread_ids.begin(), thread_ids.size(), thread_ids.begin());
  std::sort(thread_ids.begin(), thread_ids.end());
  const auto unique_end = std::unique(thread_ids.begin(), thread_ids.end());
  return expect(unique_end == thread_ids.end(), "persistent workers did not use distinct threads") && ok;
}

bool test_serial_threshold(CpuExecutor & executor) {
  RangeContext context;
  const std::thread::id caller = std::this_thread::get_id();
  const CpuExecutorStatus status = executor.parallel_for_rows(7, record_ranges, &context);
  return expect(status == CpuExecutorStatus::ok, "serial job failed") &&
    expect(context.call_count.load() == 1, "small job was not kept serial") &&
    expect(context.begins[0] == 0 && context.ends[0] == 7, "serial range mismatch") &&
    expect(context.thread_ids[0] == caller, "serial job did not run on calling thread");
}

struct FillContext {
  int seed = 0;
  std::vector<int> * values = nullptr;
};

void fill_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & context = *static_cast<FillContext *>(opaque_context);
  for (std::size_t row = row_begin; row < row_end; ++row) {
    (*context.values)[row] = context.seed + static_cast<int>(row * 3);
  }
}

bool test_repeated_jobs(CpuExecutor & executor) {
  constexpr std::size_t row_count = 31;
  std::vector<int> values(row_count);
  for (int repetition = 0; repetition < 64; ++repetition) {
    std::fill(values.begin(), values.end(), -1);
    FillContext context{repetition * 101, &values};
    if (!expect(
          executor.parallel_for_rows(row_count, fill_rows, &context) == CpuExecutorStatus::ok,
          "repeated job failed")) {
      return false;
    }
    for (std::size_t row = 0; row < row_count; ++row) {
      if (!expect(values[row] == context.seed + static_cast<int>(row * 3), "reused worker wrote wrong row")) {
        return false;
      }
    }
  }
  return true;
}

struct NestedContext {
  CpuExecutor * executor = nullptr;
  std::atomic<CpuExecutorStatus> nested_status = CpuExecutorStatus::ok;
};

void no_op_rows(void *, std::size_t, std::size_t) noexcept {}

void try_nested_job(
  void * opaque_context,
  const std::size_t row_begin,
  std::size_t) noexcept {
  if (row_begin != 0) {
    return;
  }
  auto & context = *static_cast<NestedContext *>(opaque_context);
  context.nested_status.store(
    context.executor->parallel_for_rows(1, no_op_rows, nullptr),
    std::memory_order_relaxed);
}

bool test_busy_result(CpuExecutor & executor) {
  NestedContext context{&executor};
  const CpuExecutorStatus outer_status = executor.parallel_for_rows(17, try_nested_job, &context);
  return expect(outer_status == CpuExecutorStatus::ok, "outer nested-job test failed") &&
    expect(context.nested_status.load() == CpuExecutorStatus::busy, "nested job did not return busy");
}

struct BlockingContext {
  std::atomic_bool entered = false;
  std::atomic_bool release = false;
};

void block_main_partition(
  void * opaque_context,
  const std::size_t row_begin,
  std::size_t) noexcept {
  if (row_begin != 0) {
    return;
  }
  auto & context = *static_cast<BlockingContext *>(opaque_context);
  context.entered.store(true, std::memory_order_release);
  while (!context.release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

bool test_concurrent_busy_result(CpuExecutor & executor) {
  BlockingContext context;
  std::atomic<CpuExecutorStatus> first_status = CpuExecutorStatus::invalid_argument;
  std::thread first_submitter([&] {
    first_status.store(
      executor.parallel_for_rows(17, block_main_partition, &context),
      std::memory_order_release);
  });
  while (!context.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  const CpuExecutorStatus competing_status = executor.parallel_for_rows(17, no_op_rows, nullptr);
  context.release.store(true, std::memory_order_release);
  first_submitter.join();
  return expect(competing_status == CpuExecutorStatus::busy, "concurrent job did not return busy") &&
    expect(first_status.load(std::memory_order_acquire) == CpuExecutorStatus::ok, "first concurrent job failed");
}

bool near(const float lhs, const float rhs) {
  return std::fabs(lhs - rhs) <= 1.0e-5F * std::max(1.0F, std::max(std::fabs(lhs), std::fabs(rhs)));
}

bool test_q8_matvec(CpuExecutor & executor) {
  constexpr std::size_t rows = 13;
  constexpr std::size_t blocks_per_row = 3;
  constexpr std::size_t row_values = blocks_per_row * q8_0_values_per_block;
  std::vector<float> matrix_values(rows * row_values);
  std::array<float, row_values> vector_values{};
  for (std::size_t index = 0; index < matrix_values.size(); ++index) {
    matrix_values[index] = 2.0F * std::sin(static_cast<float>(index) * 0.071F);
  }
  for (std::size_t index = 0; index < vector_values.size(); ++index) {
    vector_values[index] = 1.5F * std::cos(static_cast<float>(index) * 0.13F);
  }

  std::vector<Q8_0Block> matrix(rows * blocks_per_row);
  std::array<Q8_0Block, blocks_per_row> vector{};
  qwen35x::cpu::q8_0_quantize(
    matrix_values.data(), matrix.data(), matrix.size(), Q8_0Backend::scalar);
  qwen35x::cpu::q8_0_quantize(
    vector_values.data(), vector.data(), vector.size(), Q8_0Backend::scalar);

  std::array<float, rows> expected{};
  std::array<float, rows> actual{};
  qwen35x::cpu::q8_0_matvec(
    matrix.data(), vector.data(), expected.data(), rows, blocks_per_row, Q8_0Backend::scalar);
  for (int repetition = 0; repetition < 16; ++repetition) {
    actual.fill(-123.0F);
    const CpuExecutorStatus status = executor.q8_0_matvec(
      matrix.data(), vector.data(), actual.data(), rows, blocks_per_row, Q8_0Backend::scalar);
    if (!expect(status == CpuExecutorStatus::ok, "parallel Q8 matvec failed")) {
      return false;
    }
    for (std::size_t row = 0; row < rows; ++row) {
      if (!expect(near(actual[row], expected[row]), "parallel Q8 matvec row mismatch")) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

int main() {
  std::error_code error_code;
  auto executor = CpuExecutor::create(
    qwen35x::cpu::CpuExecutorConfig{
      .thread_count = 4,
      .min_parallel_rows = 8,
    },
    error_code);
  if (!expect(executor != nullptr, "could not create executor")) {
    std::cerr << "creation error: " << error_code.message() << '\n';
    return 1;
  }

  bool ok = expect(!error_code, "successful creation retained an error") &&
    expect(executor->thread_count() == 4, "thread count must include main thread") &&
    expect(executor->worker_thread_count() == 3, "worker thread count mismatch") &&
    expect(executor->min_parallel_rows() == 8, "serial threshold mismatch");
  ok = test_static_partitions(*executor) && ok;
  ok = test_serial_threshold(*executor) && ok;
  ok = test_repeated_jobs(*executor) && ok;
  ok = test_busy_result(*executor) && ok;
  ok = test_concurrent_busy_result(*executor) && ok;
  ok = test_q8_matvec(*executor) && ok;
  ok = expect(
    executor->parallel_for_rows(1, nullptr, nullptr) == CpuExecutorStatus::invalid_argument,
    "null task did not return invalid_argument") && ok;

  if (ok) {
    std::cout << "persistent CPU executor tests passed\n";
    return 0;
  }
  return 1;
}
