# Third-Party Notices

This repository is licensed under MIT for original project code (see [LICENSE](LICENSE)).

The project also includes or references third-party materials with their own licenses.

## Included As Git Submodules

1. `third_party/reference/llama.cpp`  
Upstream: https://github.com/ggml-org/llama.cpp  
License: MIT  
Local license file: `third_party/reference/llama.cpp/LICENSE`  
Usage in this repo: reference-only submodule (not linked into `qwen35x` build)

2. `third_party/reference/tinygrad`  
Upstream: https://github.com/tinygrad/tinygrad  
License: MIT  
Local license file: `third_party/reference/tinygrad/LICENSE`  
Usage in this repo: reference-only submodule (not linked into `qwen35x` build)

3. `third_party/reference/lucebox-hub`  
Upstream: https://github.com/Luce-Org/lucebox-hub  
License: MIT  
Local license file: `third_party/reference/lucebox-hub/LICENSE`  
Usage in this repo: reference submodule; selected CUDA benchmark sources are used by local benchmark tooling

## Vendored Third-Party Sources

1. `third_party/vendor/luce_megakernel/kernel.cu`  
Source provenance: adapted from `third_party/reference/lucebox-hub/megakernel/kernel.cu`  
Upstream: https://github.com/Luce-Org/lucebox-hub  
License: MIT  
Local license file: `third_party/vendor/luce_megakernel/LICENSE.Lucebox`  
Usage in this repo: compiled into `qwen35x_lucebench` for local CUDA benchmark comparisons

## Model Assets

Model files are downloaded locally by user scripts into `models/` and are gitignored by default.
When downloaded from Hugging Face (for example `Qwen/Qwen3.5-0.8B`), they are subject to the
license and terms of the respective model repository.
