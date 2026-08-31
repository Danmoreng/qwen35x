#include "qwen35x/cpu/q4_0.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using qwen35x::cpu::Q4_0Block;
using qwen35x::cpu::Q4_0BlockX8;
using qwen35x::cpu::Q8_0Backend;
using qwen35x::cpu::Q8_0Block;
using qwen35x::cpu::Q8_0BlockX4;

bool expect(const bool condition, const char * message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

bool near(const float lhs, const float rhs) {
  const float tolerance = 2.0e-4F * std::max({1.0F, std::fabs(lhs), std::fabs(rhs)});
  return std::fabs(lhs - rhs) <= tolerance;
}

std::vector<float> make_values(const std::size_t block_count, const float phase) {
  std::vector<float> values(block_count * 32);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = std::sin(static_cast<float>(index) * 0.173F + phase) * 3.5F +
      std::cos(static_cast<float>(index) * 0.071F - phase) * 0.75F;
  }
  return values;
}

std::vector<Q4_0Block> make_q4(const std::size_t block_count) {
  std::vector<Q4_0Block> blocks(block_count);
  for (std::size_t block = 0; block < block_count; ++block) {
    // 1.0 in IEEE binary16. Varying nibble data exercises all signed values.
    blocks[block].d = (block % 3 == 1) ? 0xbc00U : 0x3c00U;
    for (std::size_t index = 0; index < 16; ++index) {
      const std::uint8_t low = static_cast<std::uint8_t>((block * 5 + index * 3) & 0x0fU);
      const std::uint8_t high = static_cast<std::uint8_t>((block * 7 + index * 11 + 1) & 0x0fU);
      blocks[block].qs[index] = static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  return blocks;
}

bool test_packed_backend(const Q8_0Backend backend) {
  constexpr std::size_t blocks_per_row = 5;
  constexpr std::size_t rows = 16;
  constexpr std::size_t vectors = 8;
  constexpr std::size_t output_stride = rows + 5;
  const std::vector<Q4_0Block> weights = make_q4(rows * blocks_per_row);
  const std::vector<float> activation_values = make_values(vectors * blocks_per_row, -0.27F);
  std::vector<Q8_0Block> activations(vectors * blocks_per_row);
  qwen35x::cpu::q8_0_quantize(
    activation_values.data(), activations.data(), activations.size(), Q8_0Backend::scalar);

  std::vector<Q4_0BlockX8> packed_weights((rows / 8) * blocks_per_row);
  std::vector<Q8_0BlockX4> packed_activations((vectors / 4) * blocks_per_row);
  std::vector<Q8_0BlockX4> directly_quantized((vectors / 4) * blocks_per_row);
  qwen35x::cpu::q4_0_pack_rows_8(
    weights.data(), packed_weights.data(), rows, blocks_per_row);
  qwen35x::cpu::q8_0_pack_vectors_4(
    activations.data(), packed_activations.data(), vectors, blocks_per_row);
  qwen35x::cpu::q8_0_quantize_vectors_4(
    activation_values.data(), directly_quantized.data(), vectors,
    blocks_per_row, backend);

  bool ok = true;
  ok = expect(
    packed_weights[0].d[0] == weights[0].d &&
      packed_weights[0].d[1] == weights[blocks_per_row].d,
    "packed scale layout mismatch") && ok;
  ok = expect(
    packed_weights[0].qs[0] == static_cast<std::uint8_t>(weights[0].qs[0] ^ 0x88U) &&
      packed_weights[0].qs[8] ==
        static_cast<std::uint8_t>(weights[blocks_per_row].qs[0] ^ 0x88U),
    "packed quant layout mismatch") && ok;
  for (std::size_t block = 0; block < packed_activations.size(); ++block) {
    for (std::size_t token = 0; token < 4; ++token) {
      ok = expect(
        directly_quantized[block].scales[token] == packed_activations[block].scales[token],
        "direct packed quantize scale mismatch") && ok;
      ok = expect(
        directly_quantized[block].sums[token] == packed_activations[block].sums[token],
        "direct packed quantize sum mismatch") && ok;
    }
    for (std::size_t index = 0; index < 128; ++index) {
      ok = expect(
        directly_quantized[block].qs[index] == packed_activations[block].qs[index],
        "direct packed quantize value mismatch") && ok;
    }
  }

  std::vector<float> reference(vectors * output_stride, -17.0F);
  std::vector<float> actual(vectors * output_stride, -17.0F);
  qwen35x::cpu::q4_0_matmul_q8_0(
    weights.data(), activations.data(), reference.data(), rows, vectors,
    blocks_per_row, output_stride, Q8_0Backend::scalar);
  qwen35x::cpu::q4_0_packed_matmul_q8_0(
    packed_weights.data(), packed_activations.data(), actual.data(), rows,
    vectors, blocks_per_row, output_stride, backend);
  for (std::size_t vector = 0; vector < vectors; ++vector) {
    for (std::size_t row = 0; row < rows; ++row) {
      ok = expect(
        near(reference[vector * output_stride + row], actual[vector * output_stride + row]),
        "packed matmul mismatch") && ok;
    }
    for (std::size_t row = rows; row < output_stride; ++row) {
      ok = expect(actual[vector * output_stride + row] == -17.0F,
                  "packed matmul overwrote output padding") && ok;
    }
  }

  std::vector<float> reference_matvec(rows);
  std::vector<float> packed_matvec(rows);
  std::vector<float> prepared_matvec(rows);
  std::vector<Q8_0BlockX4> prepared_input(blocks_per_row);
  qwen35x::cpu::q4_0_matvec_q8_0(
    weights.data(), activations.data(), reference_matvec.data(), rows,
    blocks_per_row, Q8_0Backend::scalar);
  qwen35x::cpu::q4_0_packed_matvec_q8_0(
    packed_weights.data(), activations.data(), packed_matvec.data(), rows,
    blocks_per_row, backend);
  qwen35x::cpu::q8_0_quantize_vector_1(
    activation_values.data(), prepared_input.data(), blocks_per_row, backend);
  qwen35x::cpu::q4_0_packed_matvec_prepared_q8_0(
    packed_weights.data(), prepared_input.data(), prepared_matvec.data(), rows,
    blocks_per_row, backend);
  for (std::size_t row = 0; row < rows; ++row) {
    ok = expect(near(reference_matvec[row], packed_matvec[row]),
                "packed matvec mismatch") && ok;
    ok = expect(near(reference_matvec[row], prepared_matvec[row]),
                "prepared packed matvec mismatch") && ok;
  }

  constexpr std::size_t gather_row = 9;
  std::vector<float> reference_row(blocks_per_row * 32);
  std::vector<float> packed_row(blocks_per_row * 32);
  qwen35x::cpu::q4_0_dequantize(
    weights.data() + gather_row * blocks_per_row, reference_row.data(),
    blocks_per_row, Q8_0Backend::scalar);
  qwen35x::cpu::q4_0_packed_dequantize_row(
    packed_weights.data(), gather_row, packed_row.data(), blocks_per_row);
  for (std::size_t index = 0; index < reference_row.size(); ++index) {
    ok = expect(reference_row[index] == packed_row[index],
                "packed row gather mismatch") && ok;
  }
  return ok;
}

bool test_backend(const Q8_0Backend backend) {
  constexpr std::size_t blocks_per_row = 5;
  constexpr std::size_t rows = 9;
  constexpr std::size_t vectors = 7;
  constexpr std::size_t output_stride = rows + 3;
  const std::vector<Q4_0Block> weights = make_q4(rows * blocks_per_row);
  const std::vector<float> activation_values = make_values(vectors * blocks_per_row, 0.31F);
  std::vector<Q8_0Block> activations(vectors * blocks_per_row);
  qwen35x::cpu::q8_0_quantize(
    activation_values.data(), activations.data(), activations.size(), Q8_0Backend::scalar);

  bool ok = true;
  std::vector<float> scalar_dequant(weights.size() * 32);
  std::vector<float> test_dequant(weights.size() * 32);
  qwen35x::cpu::q4_0_dequantize(
    weights.data(), scalar_dequant.data(), weights.size(), Q8_0Backend::scalar);
  qwen35x::cpu::q4_0_dequantize(
    weights.data(), test_dequant.data(), weights.size(), backend);
  for (std::size_t index = 0; index < scalar_dequant.size(); ++index) {
    ok = expect(scalar_dequant[index] == test_dequant[index], "dequantize mismatch") && ok;
  }
  ok = expect(scalar_dequant[0] == -8.0F, "low-nibble layout mismatch") && ok;
  ok = expect(scalar_dequant[16] == -7.0F, "high-nibble layout mismatch") && ok;

  const float scalar_dot = qwen35x::cpu::q4_0_dot_q8_0(
    weights.data(), activations.data(), blocks_per_row, Q8_0Backend::scalar);
  const float test_dot = qwen35x::cpu::q4_0_dot_q8_0(
    weights.data(), activations.data(), blocks_per_row, backend);
  ok = expect(near(scalar_dot, test_dot), "dot mismatch") && ok;

  std::vector<float> scalar_matvec(rows);
  std::vector<float> test_matvec(rows);
  qwen35x::cpu::q4_0_matvec_q8_0(
    weights.data(), activations.data(), scalar_matvec.data(), rows,
    blocks_per_row, Q8_0Backend::scalar);
  qwen35x::cpu::q4_0_matvec_q8_0(
    weights.data(), activations.data(), test_matvec.data(), rows,
    blocks_per_row, backend);
  for (std::size_t row = 0; row < rows; ++row) {
    ok = expect(near(scalar_matvec[row], test_matvec[row]), "matvec mismatch") && ok;
  }

  std::vector<float> activation_scales(activations.size());
  std::vector<float> weight_scales(weights.size());
  qwen35x::cpu::q8_0_scales_to_f32(
    activations.data(), activation_scales.data(), activations.size());
  qwen35x::cpu::q4_0_scales_to_f32(weights.data(), weight_scales.data(), weights.size());
  std::vector<float> scalar_matmul(vectors * output_stride, -19.0F);
  std::vector<float> test_matmul(vectors * output_stride, -19.0F);
  qwen35x::cpu::q4_0_matmul_q8_0(
    weights.data(), activations.data(), scalar_matmul.data(), rows, vectors,
    blocks_per_row, output_stride, Q8_0Backend::scalar);
  qwen35x::cpu::q4_0_matmul_q8_0(
    weights.data(), activations.data(), test_matmul.data(), rows, vectors,
    blocks_per_row, output_stride, backend,
    activation_scales.data(), weight_scales.data());
  for (std::size_t vector = 0; vector < vectors; ++vector) {
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t index = vector * output_stride + row;
      ok = expect(near(scalar_matmul[index], test_matmul[index]), "matmul mismatch") && ok;
    }
    for (std::size_t padding = rows; padding < output_stride; ++padding) {
      ok = expect(
        test_matmul[vector * output_stride + padding] == -19.0F,
        "matmul overwrote output padding") && ok;
    }
  }
  return ok;
}

} // namespace

int main() {
  bool ok = test_backend(Q8_0Backend::scalar) &&
    test_backend(Q8_0Backend::auto_select) &&
    test_packed_backend(Q8_0Backend::scalar) &&
    test_packed_backend(Q8_0Backend::auto_select);
  if (qwen35x::cpu::q8_0_backend_available(Q8_0Backend::avx2)) {
    ok = test_backend(Q8_0Backend::avx2) &&
      test_packed_backend(Q8_0Backend::avx2) && ok;
  }
  if (ok) {
    std::cout << "q4_0 tests passed\n";
    return 0;
  }
  return 1;
}
