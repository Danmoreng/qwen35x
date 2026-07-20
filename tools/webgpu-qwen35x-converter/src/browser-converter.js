const TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"];

const $ = (id) => document.querySelector(id);
let outputDir = null;

function log(line) {
  const el = $("#log");
  el.textContent += `${line}\n`;
  el.scrollTop = el.scrollHeight;
}

function setProgress(done, total) {
  $("#progress").value = total > 0 ? done / total : 0;
}

function readU64LE(view, offset) {
  return Number(view.getBigUint64(offset, true));
}

function bf16ToF32(bits) {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint32(0, bits << 16, true);
  return view.getFloat32(0, true);
}

function f16ToF32(bits) {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

function f32ToF16(value) {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (abs === 0) return sign;
  if (abs >= 65504) return sign | 0x7bff;
  if (abs < 0.00006103515625) {
    return sign | Math.max(0, Math.min(0x3ff, Math.round(abs / Math.pow(2, -24))));
  }
  const exp = Math.floor(Math.log2(abs));
  const mant = abs / Math.pow(2, exp) - 1;
  return sign | ((exp + 15) << 10) | Math.max(0, Math.min(0x3ff, Math.round(mant * 1024)));
}

function fileKey(file) {
  return (file.webkitRelativePath || file.name).replaceAll("\\", "/");
}

function makeFileMap(files) {
  const byPath = new Map();
  const byName = new Map();
  for (const file of files) {
    const key = fileKey(file);
    byPath.set(key, file);
    byName.set(file.name, file);
  }
  return { byPath, byName };
}

function findFile(fileMap, name) {
  if (fileMap.byName.has(name)) return fileMap.byName.get(name);
  for (const [path, file] of fileMap.byPath) {
    if (path.endsWith(`/${name}`)) return file;
  }
  return null;
}

async function readTextFile(fileMap, name) {
  const file = findFile(fileMap, name);
  if (!file) throw new Error(`Missing ${name}`);
  return file.text();
}

async function readSafetensorsHeader(file) {
  const lenBuffer = await file.slice(0, 8).arrayBuffer();
  const headerLen = readU64LE(new DataView(lenBuffer), 0);
  const headerBuffer = await file.slice(8, 8 + headerLen).arrayBuffer();
  const headerText = new TextDecoder().decode(headerBuffer);
  return { headerLen, header: JSON.parse(headerText) };
}

async function buildTensorFileMap(fileMap) {
  const indexFile = findFile(fileMap, "model.safetensors.index.json");
  if (indexFile) {
    const index = JSON.parse(await indexFile.text());
    const out = new Map();
    for (const [tensor, shard] of Object.entries(index.weight_map || {})) {
      const file = findFile(fileMap, shard.split(/[\\/]/).pop());
      if (!file) throw new Error(`Missing safetensors shard ${shard}`);
      out.set(tensor, file);
    }
    return out;
  }
  const single = [...fileMap.byName.values()].find((file) => file.name.endsWith(".safetensors"));
  if (!single) throw new Error("Missing .safetensors file");
  return { single };
}

function tensorElementCount(shape) {
  return shape.reduce((acc, v) => acc * v, 1);
}

function tensorToF32(rawBuffer, info) {
  const count = tensorElementCount(info.shape);
  const out = new Float32Array(count);
  if (info.dtype === "F32") return new Float32Array(rawBuffer, 0, count).slice();
  const u16 = new Uint16Array(rawBuffer, 0, count);
  for (let i = 0; i < count; ++i) {
    if (info.dtype === "BF16") out[i] = bf16ToF32(u16[i]);
    else if (info.dtype === "F16") out[i] = f16ToF32(u16[i]);
    else throw new Error(`Unsupported dtype ${info.dtype}`);
  }
  return out;
}

function makeTensorPlan(config) {
  const t = config.text_config;
  const layers = t.layer_types;
  const names = new Set(["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]);
  for (let i = 0; i < t.num_hidden_layers; ++i) {
    const base = `model.language_model.layers.${i}.`;
    names.add(`${base}input_layernorm.weight`);
    names.add(`${base}post_attention_layernorm.weight`);
    names.add(`${base}mlp.gate_proj.weight`);
    names.add(`${base}mlp.up_proj.weight`);
    names.add(`${base}mlp.down_proj.weight`);
    if (layers[i] === "linear_attention") {
      names.add(`${base}linear_attn.in_proj_qkv.weight`);
      names.add(`${base}linear_attn.in_proj_z.weight`);
      names.add(`${base}linear_attn.in_proj_b.weight`);
      names.add(`${base}linear_attn.in_proj_a.weight`);
      names.add(`${base}linear_attn.conv1d.weight`);
      names.add(`${base}linear_attn.out_proj.weight`);
      names.add(`${base}linear_attn.norm.weight`);
      names.add(`${base}linear_attn.A_log`);
      names.add(`${base}linear_attn.dt_bias`);
    } else {
      names.add(`${base}self_attn.q_proj.weight`);
      names.add(`${base}self_attn.k_proj.weight`);
      names.add(`${base}self_attn.v_proj.weight`);
      names.add(`${base}self_attn.o_proj.weight`);
      names.add(`${base}self_attn.q_norm.weight`);
      names.add(`${base}self_attn.k_norm.weight`);
    }
  }
  return [...names];
}

function shouldQuantize(name, shape) {
  if (shape.length !== 2) return false;
  if (name.endsWith("conv1d.weight")) return false;
  return shape[0] * shape[1] >= 1024;
}

function writeF16(values) {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; ++i) out[i] = f32ToF16(values[i]);
  return new Uint8Array(out.buffer);
}

function quantizeQ8Rowwise(values, rows, cols) {
  const quant = new Uint8Array(Math.ceil((rows * cols) / 4) * 4);
  const quantView = new DataView(quant.buffer);
  const scales = new Float32Array(rows);
  for (let r = 0; r < rows; ++r) {
    let maxAbs = 0;
    const rowBase = r * cols;
    for (let c = 0; c < cols; ++c) maxAbs = Math.max(maxAbs, Math.abs(values[rowBase + c]));
    const scale = maxAbs > 0 ? maxAbs / 127 : 1;
    scales[r] = scale;
    for (let c = 0; c < cols; ++c) {
      let q = Math.round(values[rowBase + c] / scale);
      q = Math.max(-127, Math.min(127, q));
      quantView.setInt8(rowBase + c, q);
    }
  }
  return { quant, scaleBytes: new Uint8Array(scales.buffer) };
}

async function createOutputWriter(name) {
  if (!outputDir) return new MemoryWriter(name);
  const handle = await outputDir.getFileHandle(name, { create: true });
  const stream = await handle.createWritable();
  return new StreamWriter(stream);
}

class StreamWriter {
  constructor(stream) {
    this.stream = stream;
    this.total = 0;
  }

  async writeAligned(bytes, align = 16) {
    let offset = this.total;
    const padding = (align - (offset % align)) % align;
    if (padding) {
      await this.stream.write(new Uint8Array(padding));
      this.total += padding;
      offset += padding;
    }
    await this.stream.write(bytes);
    this.total += bytes.byteLength;
    return { offset, byteLength: bytes.byteLength };
  }

  async close() {
    await this.stream.close();
  }
}

class MemoryWriter {
  constructor(name) {
    this.name = name;
    this.parts = [];
    this.total = 0;
  }

  async writeAligned(bytes, align = 16) {
    let offset = this.total;
    const padding = (align - (offset % align)) % align;
    if (padding) {
      this.parts.push(new Uint8Array(padding));
      this.total += padding;
      offset += padding;
    }
    this.parts.push(bytes);
    this.total += bytes.byteLength;
    return { offset, byteLength: bytes.byteLength };
  }

  async close() {
    downloadBlob(this.name, new Blob(this.parts));
  }
}

function downloadBlob(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function writeOutputFile(name, data) {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  if (outputDir) {
    const handle = await outputDir.getFileHandle(name, { create: true });
    const stream = await handle.createWritable();
    await stream.write(bytes);
    await stream.close();
  } else {
    downloadBlob(name, new Blob([bytes]));
  }
}

async function readTensorRaw(file, headerLen, info) {
  const [start, end] = info.data_offsets;
  return file.slice(8 + headerLen + start, 8 + headerLen + end).arrayBuffer();
}

async function convert() {
  $("#log").textContent = "";
  setProgress(0, 1);
  const files = [...$("#files").files];
  if (!files.length) throw new Error("Choose the HF model folder first.");
  const matrixStorage = $("#storage").value;
  const fileMap = makeFileMap(files);
  const config = JSON.parse(await readTextFile(fileMap, "config.json"));
  const tensorFileMap = await buildTensorFileMap(fileMap);
  const headers = new Map();
  const getHeader = async (file) => {
    if (!headers.has(file)) headers.set(file, await readSafetensorsHeader(file));
    return headers.get(file);
  };

  const tensorNames = makeTensorPlan(config);
  const weights = await createOutputWriter("weights.bin");
  const manifest = {
    format: matrixStorage === "q8" ? "qwen35x-webgpu-q8-v1" : "qwen35x-webgpu-f16-v1",
    source_model_dir: "browser-file-picker",
    weight_file: "weights.bin",
    block_size: 64,
    text_config: config.text_config,
    tensors: {},
  };

  log(`Converting ${tensorNames.length} tensors to ${matrixStorage}`);
  for (let i = 0; i < tensorNames.length; ++i) {
    const name = tensorNames[i];
    const file = tensorFileMap.single || tensorFileMap.get(name);
    if (!file) throw new Error(`No safetensors shard for ${name}`);
    const { headerLen, header } = await getHeader(file);
    const info = header[name];
    if (!info) throw new Error(`Missing tensor in safetensors: ${name}`);
    log(`${i + 1}/${tensorNames.length} ${name} ${info.dtype} [${info.shape.join(",")}]`);
    const raw = await readTensorRaw(file, headerLen, info);
    const f32 = tensorToF32(raw, info);
    const entry = { shape: info.shape, source_dtype: info.dtype };
    if (matrixStorage === "q8" && shouldQuantize(name, info.shape)) {
      const [rows, cols] = info.shape;
      const q = quantizeQ8Rowwise(f32, rows, cols);
      entry.storage = "q8_rowwise";
      entry.data = await weights.writeAligned(q.quant, 16);
      entry.scales = await weights.writeAligned(q.scaleBytes, 16);
    } else {
      entry.storage = "f16";
      if (name.endsWith("linear_attn.conv1d.weight") && info.shape.length === 3) {
        const [channels, one, kernel] = info.shape;
        if (one !== 1) throw new Error(`Unexpected conv1d shape for ${name}: [${info.shape.join(",")}]`);
        entry.shape = [channels, kernel];
      }
      entry.data = await weights.writeAligned(writeF16(f32), 16);
    }
    manifest.tensors[name] = entry;
    setProgress(i + 1, tensorNames.length);
  }

  await weights.close();
  await writeOutputFile("manifest.json", JSON.stringify(manifest, null, 2));
  for (const fileName of TOKENIZER_FILES) {
    const file = findFile(fileMap, fileName);
    if (!file) throw new Error(`Missing ${fileName}`);
    await writeOutputFile(fileName, new Uint8Array(await file.arrayBuffer()));
  }
  log(`Done. weights.bin ${(weights.total / (1024 * 1024)).toFixed(2)} MiB`);
  if (!outputDir) log("Files were downloaded individually because no output folder was selected.");
}

$("#chooseOut").addEventListener("click", async () => {
  if (!window.showDirectoryPicker) {
    log("Directory picker is unavailable in this browser. Convert will download files individually.");
    return;
  }
  outputDir = await window.showDirectoryPicker({ mode: "readwrite" });
  log("Output folder selected.");
});

$("#convert").addEventListener("click", async () => {
  $("#convert").disabled = true;
  try {
    await convert();
  } catch (error) {
    log(`ERROR: ${error.stack || error.message || error}`);
  } finally {
    $("#convert").disabled = false;
  }
});
