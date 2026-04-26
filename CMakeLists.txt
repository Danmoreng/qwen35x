cmake_minimum_required(VERSION 3.24)
project(qwen35x LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QWEN35X_ENABLE_CUDA "Enable CUDA kernels when a CUDA compiler is available" ON)
set(QWEN35X_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures passed to CMAKE_CUDA_ARCHITECTURES")
set(QWEN35X_KERNEL_BENCH_BLOCK_SIZE "256" CACHE STRING "Compile-time BLOCK_SIZE for qwen35x kernel bench")
set(QWEN35X_KERNEL_BENCH_NUM_BLOCKS "82" CACHE STRING "Compile-time NUM_BLOCKS default for qwen35x kernel bench")
set(QWEN35X_CUDA_VARIANT "0p8b" CACHE STRING "Qwen3.5 CUDA kernel specialization for qwen35x_kernelbench: 0p8b or 4b")
set_property(CACHE QWEN35X_CUDA_VARIANT PROPERTY STRINGS 0p8b 4b)

include(CheckLanguage)
if (QWEN35X_ENABLE_CUDA)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "${QWEN35X_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
  else ()
    message(WARNING "QWEN35X_ENABLE_CUDA is ON but no CUDA compiler was found. Building CPU-only scaffold.")
  endif ()
endif ()

add_library(qwen35x_core
  src/compiler/compiler.cpp
  src/compiler/profile_loader.cpp
  src/kernels/kernel_registry.cpp
  src/runtime/cuda_bench_stub.cpp
  src/runtime/cuda_inference_stub.cpp
  src/runtime/qwen35x_cuda_backend.cpp
  src/runtime/reference_inference.cpp
  src/runtime/runtime.cpp
  src/tokenizer/tokenizer.cpp
  src/weights/modelopt_nvfp4.cpp
  src/weights/safetensors.cpp
)

target_include_directories(qwen35x_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (CMAKE_CUDA_COMPILER)
  if (NOT QWEN35X_CUDA_VARIANT STREQUAL "0p8b" AND NOT QWEN35X_CUDA_VARIANT STREQUAL "4b")
    message(FATAL_ERROR "QWEN35X_CUDA_VARIANT must be one of: 0p8b, 4b")
  endif ()

  function(add_qwen35x_cuda_variant variant_name variant_define)
    add_library(qwen35x_cuda_${variant_name} OBJECT
      src/kernels/cuda/kernel.cu
      src/kernels/cuda/prefill.cu
    )
    target_include_directories(qwen35x_cuda_${variant_name}
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(qwen35x_cuda_${variant_name}
      PRIVATE
        BLOCK_SIZE=${QWEN35X_KERNEL_BENCH_BLOCK_SIZE}
        NUM_BLOCKS=${QWEN35X_KERNEL_BENCH_NUM_BLOCKS}
        ${variant_define}=1
        QWEN35X_CUDA_VARIANT_${variant_name}=1
        LAYER_TYPE=LAYER_TYPE_${variant_name}
        decode_kernel=decode_kernel_${variant_name}
        decode_prefix_mlp_kernel=decode_prefix_mlp_kernel_${variant_name}
        final_norm_kernel=final_norm_kernel_${variant_name}
        lm_head_kernel=lm_head_kernel_${variant_name}
        rmsnorm_redundant=rmsnorm_redundant_${variant_name}
        rmsnorm_from_bf16=rmsnorm_from_bf16_${variant_name}
        matvec_bf16=matvec_bf16_${variant_name}
        matvec_gate_up_silu_bf16=matvec_gate_up_silu_bf16_${variant_name}
        matvec_down_residual_bf16=matvec_down_residual_bf16_${variant_name}
        matvec_o_residual_bf16=matvec_o_residual_bf16_${variant_name}
        full_attention_layer=full_attention_layer_${variant_name}
        deltanet_layer=deltanet_layer_${variant_name}
        pf_embed=pf_embed_${variant_name}
        pf_mark_seen_tokens=pf_mark_seen_tokens_${variant_name}
        pf_rmsnorm=pf_rmsnorm_${variant_name}
        pf_bf16_matvec=pf_bf16_matvec_${variant_name}
        pf_add_residual_bf16=pf_add_residual_bf16_${variant_name}
        pf_silu_mul_bf16=pf_silu_mul_bf16_${variant_name}
        pf_deltanet_recurrence=pf_deltanet_recurrence_${variant_name}
        pf_deltanet_conv_prepare=pf_deltanet_conv_prepare_${variant_name}
        pf_deltanet_prepare_norm_gate=pf_deltanet_prepare_norm_gate_${variant_name}
        pf_deltanet_recurrence_cols=pf_deltanet_recurrence_cols_${variant_name}
        pf_deltanet_post_norm_gate=pf_deltanet_post_norm_gate_${variant_name}
        pf_qk_norm_rope=pf_qk_norm_rope_${variant_name}
        pf_causal_softmax_to_bf16=pf_causal_softmax_to_bf16_${variant_name}
        pf_apply_attention_gate_bf16=pf_apply_attention_gate_bf16_${variant_name}
        pf_final_norm=pf_final_norm_${variant_name}
        pf_lm_head=pf_lm_head_${variant_name}
        pf_lm_reduce=pf_lm_reduce_${variant_name}
        launch_decode=launch_decode_${variant_name}
        launch_decode_prefix_mlp=launch_decode_prefix_mlp_${variant_name}
        launch_decode_final_lm=launch_decode_final_lm_${variant_name}
        launch_prefill_bf16=launch_prefill_bf16_${variant_name}
        set_decode_blocks_override=set_decode_blocks_override_${variant_name}
        query_max_safe_decode_blocks=query_max_safe_decode_blocks_${variant_name}
    )
    set_target_properties(qwen35x_cuda_${variant_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endfunction()

  add_qwen35x_cuda_variant(0p8b QWEN35X_VARIANT_0P8B)
  add_qwen35x_cuda_variant(4b QWEN35X_VARIANT_4B)

  target_sources(
    qwen35x_core
    PRIVATE
      src/kernels/cuda/qwen35x_decode_stub.cu
      $<TARGET_OBJECTS:qwen35x_cuda_0p8b>
      $<TARGET_OBJECTS:qwen35x_cuda_4b>
      src/runtime/cuda_bench.cu
      src/runtime/cuda_inference.cu
  )
  target_compile_definitions(
    qwen35x_core
    PRIVATE
      BLOCK_SIZE=${QWEN35X_KERNEL_BENCH_BLOCK_SIZE}
      NUM_BLOCKS=${QWEN35X_KERNEL_BENCH_NUM_BLOCKS}
  )
  target_link_libraries(qwen35x_core PRIVATE CUDA::cublasLt CUDA::cublas)
  target_compile_definitions(qwen35x_core PUBLIC QWEN35X_HAS_CUDA=1)
  set_target_properties(qwen35x_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  target_compile_definitions(qwen35x_core PUBLIC QWEN35X_HAS_CUDA=0)
endif ()

add_executable(qwen35x src/main.cpp)
target_link_libraries(qwen35x PRIVATE qwen35x_core)

if (CMAKE_CUDA_COMPILER)
  add_executable(
    qwen35x_kernelbench
    src/qwen35x_kernel_bench_main.cpp
  )
  target_include_directories(
    qwen35x_kernelbench
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_compile_options(
    qwen35x_kernelbench
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
  target_compile_definitions(
    qwen35x_kernelbench
    PRIVATE
      BLOCK_SIZE=${QWEN35X_KERNEL_BENCH_BLOCK_SIZE}
      NUM_BLOCKS=${QWEN35X_KERNEL_BENCH_NUM_BLOCKS}
      QWEN35X_CUDA_VARIANT_${QWEN35X_CUDA_VARIANT}=1
  )
  target_link_libraries(qwen35x_kernelbench PRIVATE qwen35x_core)
  set_target_properties(qwen35x_kernelbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
