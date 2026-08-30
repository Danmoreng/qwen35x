#include "qwen35x/cpu/delta_net.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using qwen35x::cpu::Q8_0Backend;

bool expect(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "delta_net test failure: " << message << '\n';
  }
  return condition;
}

bool near(const float lhs, const float rhs) {
  const float tolerance = 2.0e-4F + 2.0e-4F * std::max(std::fabs(lhs), std::fabs(rhs));
  return std::fabs(lhs - rhs) <= tolerance;
}

std::vector<float> values(const std::size_t count, const float phase) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const float x = static_cast<float>(index) + phase;
    result[index] = 0.2F * std::sin(0.17F * x) + 0.1F * std::cos(0.07F * x);
  }
  return result;
}

bool test_backend(const Q8_0Backend backend) {
  constexpr std::size_t heads = 3;
  constexpr std::size_t key_dim = 19;
  constexpr std::size_t value_dim = 11;
  std::vector<float> expected_state = values(heads * value_dim * key_dim, 0.3F);
  std::vector<float> actual_state = expected_state;
  const std::vector<float> q = values(heads * key_dim, 1.1F);
  const std::vector<float> k = values(heads * key_dim, 2.7F);
  const std::vector<float> v = values(heads * value_dim, 4.2F);
  const std::vector<float> alpha{0.91F, 0.87F, 0.95F};
  const std::vector<float> beta{0.33F, 0.71F, 0.52F};
  std::vector<float> expected_output(heads * value_dim);
  std::vector<float> actual_output(heads * value_dim);

  qwen35x::cpu::gated_delta_net_update_rows(
    expected_state.data(), q.data(), k.data(), v.data(), alpha.data(), beta.data(),
    expected_output.data(), heads, key_dim, value_dim, 0, heads * value_dim,
    Q8_0Backend::scalar);
  // Exercise disjoint partitions and a non-vector-width tail.
  qwen35x::cpu::gated_delta_net_update_rows(
    actual_state.data(), q.data(), k.data(), v.data(), alpha.data(), beta.data(),
    actual_output.data(), heads, key_dim, value_dim, 0, 13, backend);
  qwen35x::cpu::gated_delta_net_update_rows(
    actual_state.data(), q.data(), k.data(), v.data(), alpha.data(), beta.data(),
    actual_output.data(), heads, key_dim, value_dim, 13, heads * value_dim, backend);

  bool ok = true;
  for (std::size_t index = 0; index < expected_state.size(); ++index) {
    ok = expect(near(expected_state[index], actual_state[index]), "state mismatch") && ok;
  }
  for (std::size_t index = 0; index < expected_output.size(); ++index) {
    ok = expect(near(expected_output[index], actual_output[index]), "output mismatch") && ok;
  }
  return ok;
}

} // namespace

int main() {
  bool ok = test_backend(Q8_0Backend::scalar) && test_backend(Q8_0Backend::auto_select);
  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx2)) {
    ok = test_backend(Q8_0Backend::avx2) && ok;
  }
  if (!ok) {
    return 1;
  }
  std::cout << "gated DeltaNet scalar/AVX2 tests passed\n";
  return 0;
}
