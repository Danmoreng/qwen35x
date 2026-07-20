# qwen35x WebGPU Browser Converter

This page converts a local Hugging Face Qwen3.5 0.8B model folder into the custom WebGPU runtime format.

Run the existing static server:

```powershell
.\scripts\serve-webgpu-qwen35x.ps1
```

Open:

```text
http://127.0.0.1:8790/tools/webgpu-qwen35x-converter/
```

Use:

1. Select the Hugging Face model folder with the file picker.
2. Choose `q8 rowwise` or `f16`.
3. Click `Choose Output Folder` in Chrome or Edge.
4. Click `Convert`.

The output folder receives:

```text
manifest.json
weights.bin
config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

If the browser does not support the directory picker, the converter downloads the files individually.
