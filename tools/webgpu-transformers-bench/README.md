# Qwen3.5 0.8B WebGPU Baseline

This is a minimal browser benchmark for Qwen3.5 0.8B through Transformers.js WebGPU.

It is intended as the baseline before porting any qwen35x CUDA kernels to WGSL/WebGPU.

## Run

From the repository root:

```powershell
.\scripts\serve-webgpu-transformers-bench.ps1
```

Then open the printed localhost URL in a WebGPU-capable Chromium or Edge browser.

The first run downloads the ONNX model into the browser cache. For Qwen3.5 0.8B this can be a large download, depending on the selected dtype.

## Defaults

- Model: `onnx-community/Qwen3.5-0.8B-ONNX`
- Device: `webgpu`
- Dtype: Q4 decoder and embeddings, FP16 vision encoder placeholder
- Prompt: `Tell me a short joke.`
- Max new tokens: `128`

## Notes

Qwen3.5 support currently requires the next Transformers.js release line. The page imports:

```js
https://cdn.jsdelivr.net/npm/@huggingface/transformers@next
```

This benchmark measures browser-side generation throughput only. It is not directly comparable to the native CUDA CSVs unless prompt, context, sampling, warmup, and output token count are kept aligned.
