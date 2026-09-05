# Fixed-token CPU comparison adapter

This executable calls the public API of the pinned, unmodified llama.cpp
submodule. Use `scripts/benchmark-inference-seq.ps1 -Modes cpu-llama-fixed`, or
`scripts/benchmark-cpu-engine-comparison.ps1` for the complete paired matrix.
The adapter deliberately accepts only fixed prompt/continuation IDs, FP16 KV,
a single sequence, and CPU execution. It does not implement another inference
kernel or modify the upstream implementation.

In a Visual Studio x64 developer PowerShell (CMake and Ninja available):

```powershell
cmake -S scripts/bench/llama-fixed-cpu -B build-llama-fixed-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_AVX512_VNNI=ON -DGGML_AVX_VNNI=ON
cmake --build build-llama-fixed-cpu --target llama-fixed-cpu-bench llama-bench llama-quantize -j 16
```

The explicit VNNI flags above match the measured Ryzen 9 9955HX3D. Omit them
on CPUs without those extensions. `GGML_NATIVE` otherwise follows upstream's
native feature detection. GPU backends are disabled in this project.

Convert the same HF source as the H128 artifact, excluding unused MTP weights:

```powershell
.venv-hf-parity/Scripts/python.exe third_party/reference/llama.cpp/convert_hf_to_gguf.py models/qwen3.5-0.8b --outfile models/gguf/qwen3.5-0.8b-review-target-bf16.gguf --outtype bf16 --no-mtp
build-llama-fixed-cpu/bin/llama-quantize.exe --pure models/gguf/qwen3.5-0.8b-review-target-bf16.gguf models/gguf/qwen3.5-0.8b-review-target-Q4_0.gguf Q4_0 12
```

The converter Python environment needs the upstream conversion dependencies.
The BF16 file is a conversion intermediate, not one of the measured formats.

A prefill-only request disables logits for every batch. A decode request
computes the last prompt's logits and then one full-vocabulary output per
continuation step. As in Qwen35x, N requested outputs entail N-1 timed decode
forwards. The summary validates this count and normalizes throughput using it.
Model/context initialization and final finite-logit validation are outside
inference timing. Each engine gets one complete warmup run and three measured
runs from the shared sequential script; each invocation is a new process.

Build Qwen35x and create the H128/DOT4 artifact if not already available:

```powershell
.\scripts\build.ps1 -UseNinja -EnableCuda -Configuration Release -BuildAll
build/qwen35x_q4_h128_convert.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-q4-h128-cpu-dot4.q35h --layout cpu-dot4
```

The H128 converter refuses to overwrite an existing artifact. CUDA support in
the Qwen35x executable is optional; this comparison selects CPU execution only.
The model files are local artifacts and are not committed.
