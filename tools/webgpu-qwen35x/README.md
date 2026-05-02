# qwen35x WebGPU Runtime

This is the custom WebGPU inference path for Qwen3.5 0.8B. It does not execute the model through ONNX Runtime or Transformers.js. Transformers.js is only used by the demo page for tokenization.

## Build Weights

```powershell
node .\scripts\build-webgpu-qwen35x-weights.mjs --model-dir .\models\qwen3.5-0.8b --out-dir .\models\webgpu\qwen3.5-0.8b-q8
```

The converter reads the BF16 safetensors file and writes rowwise-q8 matrices plus f16 vectors into:

```text
models/webgpu/qwen3.5-0.8b-q8/
  manifest.json
  weights.bin
  tokenizer.json
  config.json
```

## Run

```powershell
.\scripts\serve-webgpu-qwen35x.ps1
```

Open:

```text
http://127.0.0.1:8790/
```

## Status

Implemented files include:

- BF16 safetensors to q8/f16 converter
- WebGPU buffer loader
- WGSL kernels for core decode operations
- Custom 24-layer greedy decode loop
- Browser runner with adapter reporting

The first tests should be:

1. Converter completes and reports `weights.bin`.
2. Browser loads `manifest.json` and allocates GPU buffers.
3. Single-token generation runs without device loss.
4. Compare token IDs against native CPU reference for `max_new_tokens=1`.
