#pragma once

#include "qwen35x/common/model_profile.h"
#include "qwen35x/common/runtime_target.h"
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/kernels/kernel_registry.h"

#include <ostream>

namespace qwen35x {

struct Bf16TensorBenchOptions {
  std::string model_dir;
  std::string tensor_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
  int warmup_iterations = 25;
  int benchmark_iterations = 200;
};

struct Bf16TensorBenchResult {
  std::string tensor_file;
  std::string tensor_name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  double avg_iteration_ms = 0.0;
  double matvec_per_second = 0.0;
  double gflops = 0.0;
};

bool run_bf16_tensor_benchmark(
  const Bf16TensorBenchOptions & options,
  Bf16TensorBenchResult & result,
  std::string & error_message);

class EngineRuntime {
public:
  bool initialize(const ModelProfile & profile, const RuntimeTarget & target, std::string & error_message);
  void print_dispatch_table(std::ostream & os) const;

private:
  ModelProfile profile_;
  RuntimeTarget target_;
  CompilationPlan plan_;
  KernelRegistry registry_;
};

} // namespace qwen35x
