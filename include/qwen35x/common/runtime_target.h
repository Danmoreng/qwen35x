#pragma once

#include <cstdint>
#include <string>

namespace qwen35x {

struct RuntimeTarget {
  bool cuda_enabled = true;
  int sm_version = 120;
  std::uint64_t vram_mb = 16 * 1024;
  std::string gpu_name = "RTX 5080 Laptop";
};

} // namespace qwen35x

