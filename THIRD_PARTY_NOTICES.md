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

## Model Assets

Model files are downloaded locally by user scripts into `models/` and are gitignored by default.
When downloaded from Hugging Face (for example `Qwen/Qwen3.5-0.8B`), they are subject to the
license and terms of the respective model repository.
