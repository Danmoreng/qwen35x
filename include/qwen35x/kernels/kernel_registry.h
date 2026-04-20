#pragma once

#include <optional>
#include <string>
#include <unordered_map>

namespace qwen35x {

struct KernelKey {
  std::string op;
  std::string mode;
  std::string dtype;
  std::string layout;
  int sm = 0;

  bool operator==(const KernelKey & other) const {
    return op == other.op && mode == other.mode && dtype == other.dtype && layout == other.layout && sm == other.sm;
  }
};

struct KernelKeyHash {
  std::size_t operator()(const KernelKey & key) const;
};

class KernelRegistry {
public:
  void register_kernel(const KernelKey & key, const std::string & symbol);
  std::optional<std::string> resolve(const KernelKey & key) const;
  const std::unordered_map<KernelKey, std::string, KernelKeyHash> & table() const;

private:
  std::unordered_map<KernelKey, std::string, KernelKeyHash> kernels_;
};

} // namespace qwen35x

