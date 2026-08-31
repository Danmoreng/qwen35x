#include "qwen35x/common/model_profile.h"
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q4_h128.h"
#include "qwen35x/weights/q4_h128_artifact.h"
#include "qwen35x/weights/safetensors.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using qwen35x::Q4H128TensorEncoding;
using qwen35x::Q4H128TensorInfo;

Q4H128TensorInfo make_tensor(
  std::string name,
  std::vector<std::uint64_t> shape,
  const Q4H128TensorEncoding encoding,
  const std::uint64_t sign_seed) {
  Q4H128TensorInfo tensor;
  tensor.name = std::move(name);
  tensor.shape = std::move(shape);
  tensor.encoding = encoding;
  if (encoding == Q4H128TensorEncoding::q4_h128) {
    tensor.transform_size = qwen35x::cpu::q4_h128_transform_size;
    tensor.scale_group = qwen35x::cpu::q4_0_values_per_block;
    tensor.sign_seed = sign_seed;
  } else if (encoding == Q4H128TensorEncoding::q4_0) {
    tensor.scale_group = qwen35x::cpu::q4_0_values_per_block;
  }
  return tensor;
}

bool append_required_tensors(
  const qwen35x::ModelProfile & profile,
  const std::uint64_t sign_seed,
  std::vector<Q4H128TensorInfo> & tensors,
  std::string & error) {
  const auto & config = profile.text;
  if (config.num_hidden_layers <= 0 || config.hidden_size <= 0 ||
      config.intermediate_size <= 0 || config.vocab_size <= 0 ||
      config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
      config.head_dim <= 0 || config.linear_conv_kernel_dim <= 0 ||
      config.linear_num_key_heads <= 0 || config.linear_num_value_heads <= 0 ||
      config.linear_key_head_dim <= 0 || config.linear_value_head_dim <= 0 ||
      profile.fingerprint.attention_schedule.size() !=
        static_cast<std::size_t>(config.num_hidden_layers)) {
    error = "Model profile lacks the dimensions required by Q4_H128 conversion.";
    return false;
  }

  const std::uint64_t hidden = static_cast<std::uint64_t>(config.hidden_size);
  const std::uint64_t intermediate = static_cast<std::uint64_t>(config.intermediate_size);
  const std::uint64_t vocab = static_cast<std::uint64_t>(config.vocab_size);
  const std::uint64_t head_dim = static_cast<std::uint64_t>(config.head_dim);
  const std::uint64_t full_q_out =
    static_cast<std::uint64_t>(config.num_attention_heads) * head_dim * 2;
  const std::uint64_t full_kv_out =
    static_cast<std::uint64_t>(config.num_key_value_heads) * head_dim;
  const std::uint64_t full_o_in =
    static_cast<std::uint64_t>(config.num_attention_heads) * head_dim;
  const std::uint64_t linear_q_dim =
    static_cast<std::uint64_t>(config.linear_num_key_heads) *
    static_cast<std::uint64_t>(config.linear_key_head_dim);
  const std::uint64_t linear_v_dim =
    static_cast<std::uint64_t>(config.linear_num_value_heads) *
    static_cast<std::uint64_t>(config.linear_value_head_dim);
  const std::uint64_t linear_conv_channels = linear_q_dim * 2 + linear_v_dim;

  tensors.push_back(make_tensor(
    "model.language_model.embed_tokens.weight", {vocab, hidden},
    Q4H128TensorEncoding::q4_0, sign_seed));
  tensors.push_back(make_tensor(
    "model.language_model.norm.weight", {hidden},
    Q4H128TensorEncoding::f32, sign_seed));

  for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string base =
      "model.language_model.layers." + std::to_string(layer) + ".";
    tensors.push_back(make_tensor(
      base + "input_layernorm.weight", {hidden}, Q4H128TensorEncoding::f32, sign_seed));
    tensors.push_back(make_tensor(
      base + "post_attention_layernorm.weight", {hidden},
      Q4H128TensorEncoding::f32, sign_seed));
    tensors.push_back(make_tensor(
      base + "mlp.gate_proj.weight", {intermediate, hidden},
      Q4H128TensorEncoding::q4_h128, sign_seed));
    tensors.push_back(make_tensor(
      base + "mlp.up_proj.weight", {intermediate, hidden},
      Q4H128TensorEncoding::q4_h128, sign_seed));
    tensors.push_back(make_tensor(
      base + "mlp.down_proj.weight", {hidden, intermediate},
      Q4H128TensorEncoding::q4_h128, sign_seed));

    if (profile.fingerprint.attention_schedule[static_cast<std::size_t>(layer)] ==
        qwen35x::AttentionBlock::linear) {
      tensors.push_back(make_tensor(
        base + "linear_attn.in_proj_qkv.weight", {linear_conv_channels, hidden},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.in_proj_z.weight", {linear_v_dim, hidden},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.in_proj_b.weight",
        {static_cast<std::uint64_t>(config.linear_num_value_heads), hidden},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.in_proj_a.weight",
        {static_cast<std::uint64_t>(config.linear_num_value_heads), hidden},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.conv1d.weight",
        {linear_conv_channels, 1,
         static_cast<std::uint64_t>(config.linear_conv_kernel_dim)},
        Q4H128TensorEncoding::f32, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.out_proj.weight", {hidden, linear_v_dim},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.norm.weight",
        {static_cast<std::uint64_t>(config.linear_value_head_dim)},
        Q4H128TensorEncoding::f32, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.A_log",
        {static_cast<std::uint64_t>(config.linear_num_value_heads)},
        Q4H128TensorEncoding::f32, sign_seed));
      tensors.push_back(make_tensor(
        base + "linear_attn.dt_bias",
        {static_cast<std::uint64_t>(config.linear_num_value_heads)},
        Q4H128TensorEncoding::f32, sign_seed));
    } else {
      tensors.push_back(make_tensor(
        base + "self_attn.q_proj.weight", {full_q_out, hidden},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "self_attn.k_proj.weight", {full_kv_out, hidden},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "self_attn.v_proj.weight", {full_kv_out, hidden},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "self_attn.o_proj.weight", {hidden, full_o_in},
        Q4H128TensorEncoding::q4_h128, sign_seed));
      tensors.push_back(make_tensor(
        base + "self_attn.q_norm.weight", {head_dim},
        Q4H128TensorEncoding::f32, sign_seed));
      tensors.push_back(make_tensor(
        base + "self_attn.k_norm.weight", {head_dim},
        Q4H128TensorEncoding::f32, sign_seed));
    }
  }
  return true;
}

bool shape_matches(
  const std::vector<std::int64_t> & actual,
  const std::vector<std::uint64_t> & expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] <= 0 || static_cast<std::uint64_t>(actual[index]) != expected[index]) {
      return false;
    }
  }
  return true;
}

bool convert_tensor(
  const std::string & model_dir,
  const Q4H128TensorInfo & info,
  qwen35x::Q4H128ArtifactWriter & writer,
  std::string & error) {
  qwen35x::SafetensorTensorF32 tensor;
  if (!qwen35x::SafetensorLoader::read_tensor_f32(
        model_dir, info.name, tensor, error)) {
    return false;
  }
  if (!shape_matches(tensor.shape, info.shape)) {
    error = "Safetensors shape mismatch for '" + info.name + "'.";
    return false;
  }

  if (info.encoding == Q4H128TensorEncoding::f32) {
    return writer.write_tensor(
      info.name, tensor.data.data(), tensor.data.size() * sizeof(float), error);
  }

  const std::size_t block_count =
    tensor.data.size() / qwen35x::cpu::q4_0_values_per_block;
  std::vector<qwen35x::cpu::Q4_0Block> blocks(block_count);
  if (info.encoding == Q4H128TensorEncoding::q4_h128) {
    if (info.shape.size() != 2 ||
        !qwen35x::cpu::q4_h128_quantize_matrix(
          tensor.data.data(), blocks.data(),
          static_cast<std::size_t>(info.shape[0]),
          static_cast<std::size_t>(info.shape[1]), info.sign_seed)) {
      error = "Q4_H128 projection conversion failed for '" + info.name + "'.";
      return false;
    }
  } else {
    qwen35x::cpu::q4_h128_quantize_transformed(
      tensor.data.data(), blocks.data(), block_count);
  }
  return writer.write_tensor(
    info.name, blocks.data(), blocks.size() * sizeof(qwen35x::cpu::Q4_0Block), error);
}

} // namespace

int main(int argc, char ** argv) {
  std::string model_dir;
  std::string output_path;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--hf-model-dir" && index + 1 < argc) {
      model_dir = argv[++index];
    } else if (argument == "--output" && index + 1 < argc) {
      output_path = argv[++index];
    } else if (argument == "--help") {
      std::cout << "Usage: qwen35x_q4_h128_convert --hf-model-dir <dir> --output <file>\n";
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argument << '\n';
      return 2;
    }
  }
  if (model_dir.empty() || output_path.empty()) {
    std::cerr << "Both --hf-model-dir and --output are required.\n";
    return 2;
  }
  namespace fs = std::filesystem;
  if (fs::exists(output_path)) {
    std::cerr << "Refusing to overwrite existing output: " << output_path << '\n';
    return 2;
  }

  std::string error;
  const auto profile = qwen35x::ProfileLoader::load_from_hf_directory(model_dir, error);
  if (!profile) {
    std::cerr << "Profile load failed: " << error << '\n';
    return 3;
  }
  qwen35x::Q4H128ArtifactMetadata metadata;
  metadata.num_hidden_layers = static_cast<std::uint32_t>(profile->text.num_hidden_layers);
  metadata.hidden_size = static_cast<std::uint32_t>(profile->text.hidden_size);
  metadata.intermediate_size = static_cast<std::uint32_t>(profile->text.intermediate_size);
  metadata.vocabulary_size = static_cast<std::uint32_t>(profile->text.vocab_size);
  metadata.sign_seed = qwen35x::cpu::q4_h128_default_sign_seed;

  std::vector<Q4H128TensorInfo> tensors;
  if (!append_required_tensors(*profile, metadata.sign_seed, tensors, error)) {
    std::cerr << "Tensor plan failed: " << error << '\n';
    return 3;
  }

  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path partial = output_path + ".partial." + std::to_string(nonce);
  qwen35x::Q4H128ArtifactWriter writer;
  if (!writer.open(partial.string(), metadata, tensors, error)) {
    std::cerr << "Artifact creation failed: " << error << '\n';
    return 4;
  }
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto started = std::chrono::steady_clock::now();
    if (!convert_tensor(model_dir, tensors[index], writer, error)) {
      writer.close();
      std::error_code ignored;
      fs::remove(partial, ignored);
      std::cerr << "Conversion failed: " << error << '\n';
      return 5;
    }
    const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    std::cout << '[' << (index + 1) << '/' << tensors.size() << "] "
              << tensors[index].name << " (" << seconds << " s)\n";
  }
  if (!writer.finalize(error)) {
    writer.close();
    std::error_code ignored;
    fs::remove(partial, ignored);
    std::cerr << "Artifact finalization failed: " << error << '\n';
    return 6;
  }
  std::error_code rename_error;
  fs::rename(partial, output_path, rename_error);
  if (rename_error) {
    std::error_code ignored;
    fs::remove(partial, ignored);
    std::cerr << "Could not publish completed artifact: " << rename_error.message() << '\n';
    return 6;
  }

  qwen35x::Q4H128ArtifactReader verifier;
  if (!verifier.open(output_path, error)) {
    std::cerr << "Completed artifact verification failed: " << error << '\n';
    return 7;
  }
  std::vector<std::uint8_t> verification_buffer;
  for (const Q4H128TensorInfo & tensor : verifier.tensors()) {
    if (!verifier.read_tensor_bytes(tensor.name, verification_buffer, error)) {
      std::cerr << "Completed artifact payload verification failed: " << error << '\n';
      return 7;
    }
  }
  std::cout << "Wrote and checksum-verified " << verifier.tensors().size()
            << " tensors in " << output_path << " (" << fs::file_size(output_path)
            << " bytes).\n";
  return 0;
}
