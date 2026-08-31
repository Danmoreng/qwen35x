#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q4_h128.h"
#include "qwen35x/weights/q4_h128_artifact.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(const bool condition, const char * message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

} // namespace

int main() {
  namespace fs = std::filesystem;
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path path = fs::temp_directory_path() /
    ("qwen35x-q4-h128-artifact-" + std::to_string(nonce) + ".bin");

  qwen35x::Q4H128ArtifactMetadata metadata;
  metadata.num_hidden_layers = 24;
  metadata.hidden_size = 1024;
  metadata.intermediate_size = 3584;
  metadata.vocabulary_size = 248320;
  metadata.sign_seed = qwen35x::cpu::q4_h128_default_sign_seed;

  qwen35x::Q4H128TensorInfo norm;
  norm.name = "norm.weight";
  norm.shape = {128};
  norm.encoding = qwen35x::Q4H128TensorEncoding::f32;

  qwen35x::Q4H128TensorInfo projection;
  projection.name = "projection.weight";
  projection.shape = {8, 128};
  projection.encoding = qwen35x::Q4H128TensorEncoding::q4_h128;
  projection.transform_size = 128;
  projection.scale_group = 32;
  projection.sign_seed = metadata.sign_seed;

  std::vector<float> norm_data(128);
  for (std::size_t index = 0; index < norm_data.size(); ++index) {
    norm_data[index] = static_cast<float>(index) * 0.03125F;
  }
  std::vector<qwen35x::cpu::Q4_0Block> projection_data(
    8 * 128 / qwen35x::cpu::q4_0_values_per_block);
  for (std::size_t block = 0; block < projection_data.size(); ++block) {
    projection_data[block].d = static_cast<std::uint16_t>(0x3400U + block);
    for (std::size_t index = 0; index < 16; ++index) {
      projection_data[block].qs[index] = static_cast<std::uint8_t>(block + 3 * index);
    }
  }

  bool ok = true;
  std::string error;
  qwen35x::Q4H128ArtifactWriter writer;
  ok = expect(writer.open(path.string(), metadata, {norm, projection}, error),
              error.c_str()) && ok;
  ok = expect(writer.write_tensor(
                norm.name, norm_data.data(), norm_data.size() * sizeof(float), error),
              error.c_str()) && ok;
  ok = expect(writer.write_tensor(
                projection.name, projection_data.data(),
                projection_data.size() * sizeof(qwen35x::cpu::Q4_0Block), error),
              error.c_str()) && ok;
  ok = expect(writer.finalize(error), error.c_str()) && ok;

  qwen35x::Q4H128ArtifactReader reader;
  ok = expect(reader.open(path.string(), error), error.c_str()) && ok;
  ok = expect(reader.metadata().hidden_size == metadata.hidden_size,
              "artifact metadata mismatch") && ok;
  const qwen35x::Q4H128TensorInfo * loaded = reader.find_tensor(projection.name);
  ok = expect(loaded != nullptr && loaded->encoding == projection.encoding &&
                loaded->shape == projection.shape && loaded->data_offset % 64 == 0,
              "artifact tensor metadata mismatch") && ok;
  std::vector<std::uint8_t> bytes;
  ok = expect(reader.read_tensor_bytes(projection.name, bytes, error), error.c_str()) && ok;
  ok = expect(bytes.size() == projection_data.size() * sizeof(qwen35x::cpu::Q4_0Block) &&
                std::memcmp(bytes.data(), projection_data.data(), bytes.size()) == 0,
              "artifact tensor payload mismatch") && ok;
  const std::uint64_t projection_offset = loaded == nullptr ? 0 : loaded->data_offset;
  reader.close();

  // A payload mutation must pass structural indexing but fail checksum
  // validation when the tensor is read.
  if (projection_offset != 0) {
    std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekg(static_cast<std::streamoff>(projection_offset + 3), std::ios::beg);
    char value = 0;
    corrupt.read(&value, 1);
    corrupt.clear();
    corrupt.seekp(static_cast<std::streamoff>(projection_offset + 3), std::ios::beg);
    value ^= 0x5a;
    corrupt.write(&value, 1);
  }
  ok = expect(reader.open(path.string(), error), "mutated artifact header was rejected") && ok;
  ok = expect(!reader.read_tensor_bytes(projection.name, bytes, error),
              "mutated payload passed checksum validation") && ok;
  reader.close();
  fs::remove(path);

  if (ok) {
    std::cout << "Q4_H128 artifact tests passed\n";
    return 0;
  }
  return 1;
}
