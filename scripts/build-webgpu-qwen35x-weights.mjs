#!/usr/bin/env node
import { mkdir, readFile, writeFile, copyFile } from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..").replace(/^\/([A-Za-z]:)/, "$1");
const defaultModelDir = path.join(repoRoot, "models", "qwen3.5-0.8b");
const defaultOutDir = path.join(repoRoot, "models", "webgpu", "qwen3.5-0.8b-q8");

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i], process.argv[i + 1]);
}

const modelDir = path.resolve(args.get("--model-dir") || defaultModelDir);
const outDir = path.resolve(args.get("--out-dir") || defaultOutDir);
const blockSize = Number.parseInt(args.get("--block-size") || "64", 10);
const matrixStorage = args.get("--matrix-storage") || "q8";

function usage() {
  console.log("Usage: node scripts/build-webgpu-qwen35x-weights.mjs [--model-dir models/qwen3.5-0.8b] [--out-dir models/webgpu/qwen3.5-0.8b-q8] [--block-size 64] [--matrix-storage q8|f16]");
}

if (process.argv.includes("--help")) {
  usage();
  process.exit(0);
}

function readU64LE(buffer, offset) {
  return Number(buffer.readBigUInt64LE(offset));
}

function bf16ToF32(bits) {
  const out = new ArrayBuffer(4);
  const view = new DataView(out);
  view.setUint32(0, bits << 16, true);
  return view.getFloat32(0, true);
}

function f16ToF32(bits) {
  const sign = (bits >> 15) & 1;
  let exp = (bits >> 10) & 0x1f;
  let frac = bits & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    let value = frac / 1024;
    return (sign ? -1 : 1) * Math.pow(2, -14) * value;
  }
  if (exp === 0x1f) {
    return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
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

async function readSafetensorsHeader(file) {
  const fd = await (await import("node:fs/promises")).open(file, "r");
  try {
    const lenBuffer = Buffer.alloc(8);
    await fd.read(lenBuffer, 0, 8, 0);
    const headerLen = readU64LE(lenBuffer, 0);
    const headerBuffer = Buffer.alloc(headerLen);
    await fd.read(headerBuffer, 0, headerLen, 8);
    return { headerLen, header: JSON.parse(headerBuffer.toString("utf8")) };
  } finally {
    await fd.close();
  }
}

async function findSafetensorsFile(modelDir) {
  const indexPath = path.join(modelDir, "model.safetensors.index.json");
  try {
    const index = JSON.parse(await readFile(indexPath, "utf8"));
    const first = Object.values(index.weight_map)[0];
    return path.join(modelDir, first);
  } catch {
    return path.join(modelDir, "model.safetensors-00001-of-00001.safetensors");
  }
}

async function readTensorRaw(file, headerLen, info) {
  const [start, end] = info.data_offsets;
  const absoluteStart = 8 + headerLen + start;
  const byteLength = end - start;
  const chunks = [];
  let read = 0;
  await new Promise((resolve, reject) => {
    createReadStream(file, { start: absoluteStart, end: absoluteStart + byteLength - 1 })
      .on("data", (chunk) => {
        chunks.push(chunk);
        read += chunk.length;
      })
      .on("error", reject)
      .on("end", resolve);
  });
  if (read !== byteLength) throw new Error(`Short tensor read: expected ${byteLength}, got ${read}`);
  return Buffer.concat(chunks, byteLength);
}

function tensorElementCount(shape) {
  return shape.reduce((acc, v) => acc * v, 1);
}

function tensorToF32(raw, info) {
  const count = tensorElementCount(info.shape);
  const out = new Float32Array(count);
  if (info.dtype === "F32") {
    return new Float32Array(raw.buffer, raw.byteOffset, count).slice();
  }
  const u16 = new Uint16Array(raw.buffer, raw.byteOffset, count);
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
  const names = new Set([
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
  ]);
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
  return Buffer.from(out.buffer);
}

function quantizeQ8Rowwise(values, rows, cols) {
  const quant = Buffer.alloc(Math.ceil((rows * cols) / 4) * 4);
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
      quant.writeInt8(q, rowBase + c);
    }
  }
  return { quant, scaleBytes: Buffer.from(scales.buffer) };
}

function appendBuffer(parts, buffer, align = 16) {
  let offset = parts.total;
  const padding = (align - (offset % align)) % align;
  if (padding) {
    parts.items.push(Buffer.alloc(padding));
    parts.total += padding;
    offset += padding;
  }
  parts.items.push(buffer);
  parts.total += buffer.length;
  return { offset, byteLength: buffer.length };
}

async function main() {
  const config = JSON.parse(await readFile(path.join(modelDir, "config.json"), "utf8"));
  const tensorFile = await findSafetensorsFile(modelDir);
  const { headerLen, header } = await readSafetensorsHeader(tensorFile);
  const tensorNames = makeTensorPlan(config);
  const parts = { items: [], total: 0 };
  const manifest = {
    format: "qwen35x-webgpu-q8-v1",
    source_model_dir: modelDir,
    weight_file: "weights.bin",
    block_size: blockSize,
    text_config: config.text_config,
    tensors: {},
  };

  await mkdir(outDir, { recursive: true });

  for (const name of tensorNames) {
    const info = header[name];
    if (!info) throw new Error(`Missing tensor in safetensors: ${name}`);
    process.stdout.write(`convert ${name} ${info.dtype} [${info.shape.join(",")}]\n`);
    const raw = await readTensorRaw(tensorFile, headerLen, info);
    const f32 = tensorToF32(raw, info);
    const entry = { shape: info.shape, source_dtype: info.dtype };

    if (matrixStorage === "q8" && shouldQuantize(name, info.shape)) {
      const [rows, cols] = info.shape;
      const q = quantizeQ8Rowwise(f32, rows, cols);
      entry.storage = "q8_rowwise";
      entry.data = appendBuffer(parts, q.quant, 16);
      entry.scales = appendBuffer(parts, q.scaleBytes, 16);
    } else if (matrixStorage === "q8" || matrixStorage === "f16") {
      entry.storage = "f16";
      if (name.endsWith("linear_attn.conv1d.weight") && info.shape.length === 3) {
        const [channels, one, kernel] = info.shape;
        if (one !== 1) throw new Error(`Unexpected conv1d shape for ${name}: [${info.shape.join(",")}]`);
        entry.shape = [channels, kernel];
      }
      entry.data = appendBuffer(parts, writeF16(f32), 16);
    } else {
      throw new Error(`Unsupported --matrix-storage ${matrixStorage}`);
    }
    manifest.tensors[name] = entry;
  }

  await writeFile(path.join(outDir, "weights.bin"), Buffer.concat(parts.items, parts.total));
  await writeFile(path.join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));
  for (const file of ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]) {
    await copyFile(path.join(modelDir, file), path.join(outDir, file));
  }
  console.log(`Wrote ${outDir}`);
  console.log(`weights.bin ${(parts.total / (1024 * 1024)).toFixed(2)} MiB`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
