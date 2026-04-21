cmake_minimum_required(VERSION 3.24)
project(qwen35x LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QWEN35X_ENABLE_CUDA "Enable CUDA kernels when a CUDA compiler is available" ON)
set(QWEN35X_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures passed to CMAKE_CUDA_ARCHITECTURES")

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
  src/runtime/reference_inference.cpp
  src/runtime/runtime.cpp
  src/tokenizer/tokenizer.cpp
  src/weights/safetensors.cpp
)

target_include_directories(qwen35x_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (CMAKE_CUDA_COMPILER)
  target_sources(qwen35x_core PRIVATE src/kernels/cuda/qwen35x_decode_stub.cu src/runtime/cuda_bench.cu src/runtime/cuda_inference.cu)
  target_link_libraries(qwen35x_core PRIVATE CUDA::cublasLt)
  target_compile_definitions(qwen35x_core PUBLIC QWEN35X_HAS_CUDA=1)
  set_target_properties(qwen35x_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
  target_compile_definitions(qwen35x_core PUBLIC QWEN35X_HAS_CUDA=0)
endif ()

add_executable(qwen35x src/main.cpp)
target_link_libraries(qwen35x PRIVATE qwen35x_core)
