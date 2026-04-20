#include "qwen35x/kernels/kernel_registry.h"

namespace qwen35x {

std::size_t KernelKeyHash::operator()(const KernelKey & key) const {
  const std::hash<std::string> str_hash;
  const std::hash<int> int_hash;

  std::size_t h = str_hash(key.op);
  h ^= str_hash(key.mode) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= str_hash(key.dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= str_hash(key.layout) + 0x9e3779b9 + (h << 6) + (h >> 2);
  h ^= int_hash(key.sm) + 0x9e3779b9 + (h << 6) + (h >> 2);
  return h;
}

void KernelRegistry::register_kernel(const KernelKey & key, const std::string & symbol) {
  kernels_[key] = symbol;
}

std::optional<std::string> KernelRegistry::resolve(const KernelKey & key) const {
  const auto it = kernels_.find(key);
  if (it == kernels_.end()) {
    return std::nullopt;
  }
  return it->second;
}

const std::unordered_map<KernelKey, std::string, KernelKeyHash> & KernelRegistry::table() const {
  return kernels_;
}

} // namespace qwen35x

