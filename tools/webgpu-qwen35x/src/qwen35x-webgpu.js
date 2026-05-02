import { attentionShaderSource, linearAttentionShaderSource, matvecShaderSource, normShaderSource, samplingShaderSource, shaderSource } from "./kernels.js";

const HIDDEN = 1024;
const INTERMEDIATE = 3584;

function align4(n) {
  return (n + 3) & ~3;
}

function makeUniformBuffer(device, byteSize = 32) {
  return device.createBuffer({
    size: align4(byteSize),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function uploadUniformBuffer(device, data) {
  const bytes = data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const buffer = device.createBuffer({
    size: align4(bytes.byteLength),
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  return buffer;
}

function storageBuffer(device, byteLength, usage = 0) {
  return device.createBuffer({
    size: align4(byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | usage,
  });
}

function uploadBuffer(device, data, usage = GPUBufferUsage.STORAGE) {
  const buffer = device.createBuffer({
    size: align4(data.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer || data, data.byteOffset || 0, data.byteLength));
  buffer.unmap();
  return buffer;
}

function ceilDiv(a, b) {
  return Math.ceil(a / b);
}

function clearBuffer(device, encoder, buffer, byteSize) {
  encoder.clearBuffer(buffer, 0, align4(byteSize));
}

function dispatchRows2d(program, pass, bindings, rows) {
  const maxX = 65535;
  const x = Math.min(rows, maxX);
  const y = Math.ceil(rows / maxX);
  program.dispatch(pass, bindings, x, y);
}

class Program {
  constructor(device, code, entryPoint, layout) {
    this.device = device;
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code }),
        entryPoint,
      },
    });
    this.layout = layout;
  }

  dispatch(pass, bindings, x, y = 1, z = 1) {
    const entries = bindings.map((item, index) => {
      if (Array.isArray(item)) {
        return { binding: item[0], resource: { buffer: item[1] } };
      }
      return { binding: index, resource: { buffer: item } };
    });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y, z);
  }
}

export class Qwen35xWebGpu {
  static async create({ baseUrl = "/models/webgpu/qwen3.5-0.8b-q8/" } = {}) {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter.");
    if (!adapter.features.has("shader-f16")) {
      throw new Error("This WebGPU adapter does not expose shader-f16, which this runtime currently requires.");
    }
    const requiredLimits = {};
    for (const key of ["maxBufferSize", "maxStorageBufferBindingSize"]) {
      if (adapter.limits[key] > 268435456) {
        requiredLimits[key] = adapter.limits[key];
      }
    }
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"],
      requiredLimits,
    });
    const runtime = new Qwen35xWebGpu(device, adapter, baseUrl);
    await runtime.load();
    return runtime;
  }

  constructor(device, adapter, baseUrl) {
    this.device = device;
    this.adapter = adapter;
    this.baseUrl = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
    this.tensors = new Map();
    this.layers = [];
    this.maxContext = 256;
    this.debugLayerLimit = Number.POSITIVE_INFINITY;
    this.maskTokenIds = [0, 248044, 248045];
  }

  async load() {
    const [manifest, weightsBuffer] = await Promise.all([
      fetch(`${this.baseUrl}manifest.json`).then((r) => r.json()),
      fetch(`${this.baseUrl}weights.bin`).then((r) => r.arrayBuffer()),
    ]);
    this.manifest = manifest;
    this.config = manifest.text_config;
    this.weightsRaw = weightsBuffer;
    this.makePrograms();
    this.uploadTensors();
    this.allocateWorkspace();
    this.bindLayerViews();
  }

  makePrograms() {
    const d = this.device;
    this.programs = {
      embed: new Program(d, shaderSource, "embed_gather"),
      addF32SrcDst: new Program(d, shaderSource, "add_f32_src_dst"),
      siluMul: new Program(d, shaderSource, "silu_mul"),
      matvec: new Program(d, matvecShaderSource, "q8_matvec"),
      embedQ8: new Program(d, matvecShaderSource, "q8_embed_gather"),
      tiedLmHeadQ8: new Program(d, matvecShaderSource, "q8_tied_lm_head"),
      matvecF16: new Program(d, matvecShaderSource, "f16_matvec"),
      tiedLmHeadF16: new Program(d, matvecShaderSource, "f16_tied_lm_head"),
      rmsNorm: new Program(d, normShaderSource, "rms_norm"),
      rmsNormHeads: new Program(d, normShaderSource, "rms_norm_heads"),
      splitQGate: new Program(d, attentionShaderSource, "split_q_gate"),
      writeKvCache: new Program(d, attentionShaderSource, "write_kv_cache"),
      ropeQ: new Program(d, attentionShaderSource, "rope_q"),
      ropeK: new Program(d, attentionShaderSource, "rope_k"),
      fullAttention: new Program(d, attentionShaderSource, "full_attention"),
      linearConvUpdate: new Program(d, linearAttentionShaderSource, "linear_conv_update"),
      linearL2NormQk: new Program(d, linearAttentionShaderSource, "l2_norm_qk"),
      deltanetUpdate: new Program(d, linearAttentionShaderSource, "deltanet_update"),
      linearGatedRms: new Program(d, linearAttentionShaderSource, "gated_rms"),
      maskTokens: new Program(d, samplingShaderSource, "mask_tokens"),
    };
  }

  uploadTensors() {
    const bytes = new Uint8Array(this.weightsRaw);
    for (const [name, meta] of Object.entries(this.manifest.tensors)) {
      const tensor = { name, ...meta };
      const dataSlice = bytes.slice(meta.data.offset, meta.data.offset + meta.data.byteLength);
      if (meta.storage === "q8_rowwise") {
        const scalesSlice = bytes.slice(meta.scales.offset, meta.scales.offset + meta.scales.byteLength);
        tensor.weight = uploadBuffer(this.device, dataSlice);
        tensor.scales = uploadBuffer(this.device, scalesSlice);
      } else if (meta.storage === "f16") {
        tensor.buffer = uploadBuffer(this.device, dataSlice);
      } else {
        throw new Error(`Unsupported tensor storage ${meta.storage}`);
      }
      this.tensors.set(name, tensor);
    }
  }

  allocateWorkspace() {
    const d = this.device;
    const f32 = Float32Array.BYTES_PER_ELEMENT;
    this.hiddenA = storageBuffer(d, HIDDEN * f32);
    this.hiddenB = storageBuffer(d, HIDDEN * f32);
    this.norm = storageBuffer(d, HIDDEN * f32);
    this.proj = storageBuffer(d, 8192 * f32);
    this.proj2 = storageBuffer(d, 8192 * f32);
    this.intermediateA = storageBuffer(d, INTERMEDIATE * f32);
    this.intermediateB = storageBuffer(d, INTERMEDIATE * f32);
    this.linearConv = storageBuffer(d, 6144 * f32);
    this.linearConvOut = storageBuffer(d, 6144 * f32);
    this.linearZ = storageBuffer(d, 2048 * f32);
    this.linearB = storageBuffer(d, 16 * f32);
    this.linearA = storageBuffer(d, 16 * f32);
    this.linearGated = storageBuffer(d, 2048 * f32);
    this.fullQ = storageBuffer(d, 2048 * f32);
    this.fullQNorm = storageBuffer(d, 2048 * f32);
    this.fullKNorm = storageBuffer(d, 512 * f32);
    this.fullGate = storageBuffer(d, 2048 * f32);
    this.fullAttn = storageBuffer(d, 2048 * f32);
    this.logits = storageBuffer(d, this.config.vocab_size * f32);
    this.readback = d.createBuffer({
      size: align4(this.config.vocab_size * f32),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.uVec = makeUniformBuffer(d, 32);
    this.uNorm = makeUniformBuffer(d, 32);
    this.uMatvec = makeUniformBuffer(d, 16);
    this.uAttention = makeUniformBuffer(d, 32);
    this.uLinear = makeUniformBuffer(d, 32);
    this.uMask = makeUniformBuffer(d, 16);

    const fullKv = this.maxContext * this.config.num_key_value_heads * this.config.head_dim * f32;
    this.fullStates = [];
    this.linearStates = [];
    for (let i = 0; i < this.config.num_hidden_layers; ++i) {
      if (this.config.layer_types[i] === "full_attention") {
        this.fullStates[i] = {
          k: storageBuffer(d, fullKv),
          v: storageBuffer(d, fullKv),
        };
      } else {
        this.linearStates[i] = {
          conv: storageBuffer(d, 3 * 6144 * f32),
          recurrent: storageBuffer(d, 16 * 128 * 128 * f32),
        };
      }
    }
  }

  bindLayerViews() {
    for (let i = 0; i < this.config.num_hidden_layers; ++i) {
      const base = `model.language_model.layers.${i}.`;
      const layer = {
        type: this.config.layer_types[i],
        inputNorm: this.t(`${base}input_layernorm.weight`),
        postNorm: this.t(`${base}post_attention_layernorm.weight`),
        mlpGate: this.t(`${base}mlp.gate_proj.weight`),
        mlpUp: this.t(`${base}mlp.up_proj.weight`),
        mlpDown: this.t(`${base}mlp.down_proj.weight`),
      };
      if (layer.type === "linear_attention") {
        Object.assign(layer, {
          inQkv: this.t(`${base}linear_attn.in_proj_qkv.weight`),
          inZ: this.t(`${base}linear_attn.in_proj_z.weight`),
          inB: this.t(`${base}linear_attn.in_proj_b.weight`),
          inA: this.t(`${base}linear_attn.in_proj_a.weight`),
          conv1d: this.t(`${base}linear_attn.conv1d.weight`),
          norm: this.t(`${base}linear_attn.norm.weight`),
          aLog: this.t(`${base}linear_attn.A_log`),
          dtBias: this.t(`${base}linear_attn.dt_bias`),
          outProj: this.t(`${base}linear_attn.out_proj.weight`),
        });
      } else {
        Object.assign(layer, {
          qProj: this.t(`${base}self_attn.q_proj.weight`),
          kProj: this.t(`${base}self_attn.k_proj.weight`),
          vProj: this.t(`${base}self_attn.v_proj.weight`),
          oProj: this.t(`${base}self_attn.o_proj.weight`),
          qNorm: this.t(`${base}self_attn.q_norm.weight`),
          kNorm: this.t(`${base}self_attn.k_norm.weight`),
        });
      }
      this.layers.push(layer);
    }
    this.embed = this.t("model.language_model.embed_tokens.weight");
    this.finalNorm = this.t("model.language_model.norm.weight");
  }

  t(name) {
    const tensor = this.tensors.get(name);
    if (!tensor) throw new Error(`Missing tensor ${name}`);
    return tensor;
  }

  writeVecParams({ n = 0, rows = 0, cols = 0, offset = 0, eps = 0, value = 0, position = 0, seqLen = 0 }) {
    const data = new ArrayBuffer(32);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    u[0] = n; u[1] = rows; u[2] = cols; u[3] = offset;
    f[4] = eps; f[5] = value; u[6] = position; u[7] = seqLen;
    return uploadUniformBuffer(this.device, data);
  }

  writeNormParams({ n, heads = 1, headDim = n, xOffset = 0, yOffset = 0, eps = 1e-6 }) {
    const data = new ArrayBuffer(32);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    u[0] = n; u[1] = heads; u[2] = headDim; u[3] = xOffset; u[4] = yOffset; f[5] = eps;
    return uploadUniformBuffer(this.device, data);
  }

  writeMatvecParams({ rows, cols, xOffset = 0, yOffset = 0 }) {
    return uploadUniformBuffer(this.device, new Uint32Array([rows, cols, xOffset, yOffset]));
  }

  writeAttentionParams(position, seqLen) {
    const data = new ArrayBuffer(32);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    u[0] = this.config.num_attention_heads;
    u[1] = this.config.num_key_value_heads;
    u[2] = this.config.head_dim;
    u[3] = this.ropeDim();
    u[4] = position;
    u[5] = seqLen;
    f[6] = this.config.rope_parameters?.rope_theta || 10000000;
    return uploadUniformBuffer(this.device, data);
  }

  writeLinearParams() {
    const data = new ArrayBuffer(32);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    u[0] = 6144;
    u[1] = 2048;
    u[2] = 2048;
    u[3] = 16;
    u[4] = 128;
    u[5] = 4;
    f[6] = this.config.rms_norm_eps;
    f[7] = 1 / Math.sqrt(128);
    return uploadUniformBuffer(this.device, data);
  }

  matvec(pass, tensor, x, y, xOffset = 0, yOffset = 0) {
    const [rows, cols] = tensor.shape;
    const params = this.writeMatvecParams({ rows, cols, xOffset, yOffset });
    if (tensor.storage === "q8_rowwise") {
      dispatchRows2d(this.programs.matvec, pass, [tensor.weight, tensor.scales, x, y, params], rows);
    } else {
      dispatchRows2d(this.programs.matvecF16, pass, [[2, x], [3, y], [4, params], [5, tensor.buffer]], rows);
    }
  }

  tiedLmHead(pass, x, y) {
    const [rows, cols] = this.embed.shape;
    const params = this.writeMatvecParams({ rows, cols, xOffset: 0, yOffset: 0 });
    if (this.embed.storage === "q8_rowwise") {
      dispatchRows2d(this.programs.tiedLmHeadQ8, pass, [this.embed.weight, this.embed.scales, x, y, params], rows);
    } else {
      dispatchRows2d(this.programs.tiedLmHeadF16, pass, [[2, x], [3, y], [4, params], [5, this.embed.buffer]], rows);
    }
  }

  ropeDim() {
    const factor = this.config.rope_parameters?.partial_rotary_factor ?? this.config.partial_rotary_factor ?? 0.25;
    return Math.floor(this.config.head_dim * factor) & ~1;
  }

  gatherEmbedding(pass, tokenId, y) {
    if (this.embed.storage === "q8_rowwise") {
      const [, cols] = this.embed.shape;
      const params = this.writeMatvecParams({ rows: 1, cols, xOffset: tokenId, yOffset: 0 });
      this.programs.embedQ8.dispatch(
        pass,
        [[0, this.embed.weight], [1, this.embed.scales], [3, y], [4, params]],
        ceilDiv(cols, 256),
      );
      return;
    }
    const params = this.writeVecParams({ n: HIDDEN, offset: tokenId });
    this.programs.embed.dispatch(pass, [[0, this.embed.buffer], [2, y], [3, params]], ceilDiv(HIDDEN, 256));
  }

  rms(pass, x, weight, y, n = HIDDEN, xOffset = 0, yOffset = 0) {
    const params = this.writeNormParams({ n, xOffset, yOffset, eps: this.config.rms_norm_eps });
    this.programs.rmsNorm.dispatch(pass, [x, weight.buffer, y, params], 1);
  }

  rmsHeads(pass, x, weight, y, heads, headDim, xOffset = 0, yOffset = 0) {
    const params = this.writeNormParams({ n: heads * headDim, heads, headDim, xOffset, yOffset, eps: this.config.rms_norm_eps });
    this.programs.rmsNormHeads.dispatch(pass, [x, weight.buffer, y, params], heads);
  }

  async generate(inputIds, { maxNewTokens = 1, onToken, repetitionPenalty = 1.05 } = {}) {
    if (!inputIds.length) throw new Error("No input tokens.");
    this.resetState();
    const generated = [];
    const seenTokenIds = new Set(inputIds);
    for (let i = 0; i < inputIds.length; ++i) {
      await this.forwardToken(inputIds[i], i, false);
    }
    let next = await this.sampleGreedy({ seenTokenIds, repetitionPenalty });
    for (let step = 0; step < maxNewTokens; ++step) {
      generated.push(next);
      seenTokenIds.add(next);
      onToken?.(next);
      if (step + 1 >= maxNewTokens) break;
      const pos = inputIds.length + step;
      await this.forwardToken(next, pos, true);
      next = await this.sampleGreedy({ seenTokenIds, repetitionPenalty });
    }
    return generated;
  }

  resetState() {
    const encoder = this.device.createCommandEncoder();
    const f32 = Float32Array.BYTES_PER_ELEMENT;
    for (const state of this.fullStates) {
      if (!state) continue;
      clearBuffer(
        this.device,
        encoder,
        state.k,
        this.maxContext * this.config.num_key_value_heads * this.config.head_dim * f32,
      );
      clearBuffer(
        this.device,
        encoder,
        state.v,
        this.maxContext * this.config.num_key_value_heads * this.config.head_dim * f32,
      );
    }
    for (const state of this.linearStates) {
      if (!state) continue;
      clearBuffer(this.device, encoder, state.conv, 3 * 6144 * f32);
      clearBuffer(this.device, encoder, state.recurrent, 16 * 128 * 128 * f32);
    }
    this.device.queue.submit([encoder.finish()]);
  }

  setDebugOptions({ layerLimit = Number.POSITIVE_INFINITY } = {}) {
    this.debugLayerLimit = layerLimit;
  }

  writeMaskParams() {
    const tokens = this.maskTokenIds;
    return uploadUniformBuffer(
      this.device,
      new Uint32Array([
        this.config.vocab_size,
        tokens[0] ?? 0xffffffff,
        tokens[1] ?? 0xffffffff,
        tokens[2] ?? 0xffffffff,
      ]),
    );
  }

  async forwardToken(tokenId, position) {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.gatherEmbedding(pass, tokenId, this.hiddenA);

    const layerCount = Math.min(this.layers.length, this.debugLayerLimit);
    for (let i = 0; i < layerCount; ++i) {
      this.runLayer(pass, this.layers[i], i, position);
    }

    this.rms(pass, this.hiddenA, this.finalNorm, this.norm);
    this.tiedLmHead(pass, this.norm, this.logits);
    const maskParams = this.writeMaskParams();
    this.programs.maskTokens.dispatch(pass, [[0, this.logits], [1, maskParams]], 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  runLayer(pass, layer, layerIndex, position) {
    this.rms(pass, this.hiddenA, layer.inputNorm, this.norm);
    if (layer.type === "linear_attention") {
      this.runLinearLayerApprox(pass, layer, layerIndex);
    } else {
      this.runFullLayer(pass, layer, layerIndex, position);
    }
    this.addResidual(pass, this.hiddenA, this.hiddenB);
    this.swapHidden();
    this.rms(pass, this.hiddenA, layer.postNorm, this.norm);
    this.matvec(pass, layer.mlpGate, this.norm, this.intermediateA);
    this.matvec(pass, layer.mlpUp, this.norm, this.intermediateB);
    const siluParams = this.writeVecParams({ n: INTERMEDIATE, offset: 0 });
    this.programs.siluMul.dispatch(pass, [[2, this.intermediateA], [3, siluParams], [4, this.intermediateB]], ceilDiv(INTERMEDIATE, 256));
    this.matvec(pass, layer.mlpDown, this.intermediateA, this.hiddenB);
    this.addResidual(pass, this.hiddenA, this.hiddenB);
    this.swapHidden();
  }

  addResidual(pass, residual, target) {
    const params = this.writeVecParams({ n: HIDDEN });
    this.programs.addF32SrcDst.dispatch(pass, [[2, residual], [3, params], [4, target]], ceilDiv(HIDDEN, 256));
  }

  runLinearLayerApprox(pass, layer, layerIndex) {
    this.matvec(pass, layer.inQkv, this.norm, this.linearConv);
    this.matvec(pass, layer.inZ, this.norm, this.linearZ);
    this.matvec(pass, layer.inB, this.norm, this.linearB);
    this.matvec(pass, layer.inA, this.norm, this.linearA);
    const linearParams = this.writeLinearParams();
    const state = this.linearStates[layerIndex];
    this.programs.linearConvUpdate.dispatch(
      pass,
      [
        [0, this.linearConv],
        [4, layer.conv1d.buffer],
        [5, state.conv],
        [6, this.linearConvOut],
        [12, linearParams],
      ],
      ceilDiv(6144, 256),
    );
    this.programs.linearL2NormQk.dispatch(
      pass,
      [[6, this.linearConvOut], [12, linearParams]],
      16,
      2,
    );
    this.programs.deltanetUpdate.dispatch(
      pass,
      [
        [2, this.linearB],
        [3, this.linearA],
        [6, this.linearConvOut],
        [8, layer.dtBias.buffer],
        [9, layer.aLog.buffer],
        [10, state.recurrent],
        [11, this.linearGated],
        [12, linearParams],
      ],
      16,
      128,
    );
    this.programs.linearGatedRms.dispatch(
      pass,
      [
        [1, this.linearZ],
        [7, layer.norm.buffer],
        [11, this.linearGated],
        [12, linearParams],
      ],
      16,
    );
    this.matvec(pass, layer.outProj, this.linearGated, this.hiddenB);
  }

  runFullLayer(pass, layer, layerIndex, position) {
    const headDim = this.config.head_dim;
    const nHeads = this.config.num_attention_heads;
    const nKv = this.config.num_key_value_heads;
    const qTotal = nHeads * headDim;
    const kvTotal = nKv * headDim;
    this.matvec(pass, layer.qProj, this.norm, this.proj);
    const attentionParams = this.writeAttentionParams(position, position + 1);
    this.programs.splitQGate.dispatch(
      pass,
      [[0, this.fullQ], [1, this.fullGate], [6, this.proj], [7, attentionParams]],
      ceilDiv(qTotal, 256),
    );
    this.rmsHeads(pass, this.fullQ, layer.qNorm, this.fullQNorm, nHeads, headDim);
    this.matvec(pass, layer.kProj, this.norm, this.proj2);
    this.rmsHeads(pass, this.proj2, layer.kNorm, this.fullKNorm, nKv, headDim);
    this.matvec(pass, layer.vProj, this.norm, this.fullAttn);
    this.programs.writeKvCache.dispatch(
      pass,
      [[2, this.fullStates[layerIndex].k], [3, this.fullStates[layerIndex].v], [4, this.fullKNorm], [5, this.fullAttn], [7, attentionParams]],
      ceilDiv(kvTotal, 256),
    );
    this.programs.ropeQ.dispatch(pass, [[0, this.fullQNorm], [7, attentionParams]], ceilDiv(nHeads * (this.ropeDim() / 2), 256));
    this.programs.ropeK.dispatch(pass, [[2, this.fullStates[layerIndex].k], [7, attentionParams]], ceilDiv(nKv * (this.ropeDim() / 2), 256));
    this.programs.fullAttention.dispatch(
      pass,
      [[0, this.fullQNorm], [1, this.fullGate], [2, this.fullStates[layerIndex].k], [3, this.fullStates[layerIndex].v], [6, this.proj], [7, attentionParams]],
      nHeads,
    );
    this.matvec(pass, layer.oProj, this.proj, this.hiddenB);
  }

  swapHidden() {
    const tmp = this.hiddenA;
    this.hiddenA = this.hiddenB;
    this.hiddenB = tmp;
  }

  async sampleGreedy({ seenTokenIds = new Set(), repetitionPenalty = 1.0 } = {}) {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.logits, 0, this.readback, 0, this.config.vocab_size * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(this.readback.getMappedRange());
    let best = 0;
    let bestValue = -Infinity;
    for (let i = 0; i < this.config.vocab_size; ++i) {
      let value = logits[i];
      if (repetitionPenalty > 1.0 && seenTokenIds.has(i)) {
        value = value > 0 ? value / repetitionPenalty : value * repetitionPenalty;
      }
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    this.readback.unmap();
    return best;
  }

  async debugTopLogits(k = 8) {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.logits, 0, this.readback, 0, this.config.vocab_size * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(this.readback.getMappedRange());
    const top = [];
    for (let i = 0; i < this.config.vocab_size; ++i) {
      const logit = logits[i];
      if (!Number.isFinite(logit)) continue;
      if (top.length < k || logit > top[top.length - 1].logit) {
        top.push({ id: i, logit });
        top.sort((a, b) => b.logit - a.logit);
        top.length = Math.min(top.length, k);
      }
    }
    this.readback.unmap();
    return top;
  }

  adapterInfo() {
    return this.adapter.info || {};
  }
}
