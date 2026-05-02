export const shaderSource = /* wgsl */ `
enable f16;

struct VecParams {
  n: u32,
  rows: u32,
  cols: u32,
  offset: u32,
  eps: f32,
  value: f32,
  position: u32,
  seq_len: u32,
};

@group(0) @binding(0) var<storage, read> a_f16: array<f16>;
@group(0) @binding(1) var<storage, read> b_f16: array<f16>;
@group(0) @binding(2) var<storage, read_write> out_f32: array<f32>;
@group(0) @binding(3) var<uniform> p: VecParams;
@group(0) @binding(4) var<storage, read_write> dst_f32: array<f32>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

@compute @workgroup_size(256)
fn embed_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  out_f32[i] = f32(a_f16[p.offset * p.n + i]);
}

@compute @workgroup_size(256)
fn f16_to_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  out_f32[i] = f32(a_f16[i]);
}

@compute @workgroup_size(256)
fn add_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  out_f32[i] = out_f32[i] + f32(a_f16[i]);
}

@compute @workgroup_size(256)
fn add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  out_f32[i] = out_f32[i] + out_f32[p.offset + i];
}

@compute @workgroup_size(256)
fn add_f32_src_dst(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  dst_f32[i] = dst_f32[i] + out_f32[i];
}

@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  out_f32[i] = silu(out_f32[i]) * dst_f32[i];
}
`;

export const matvecShaderSource = /* wgsl */ `
enable f16;

struct MatvecParams {
  rows: u32,
  cols: u32,
  x_offset: u32,
  y_offset: u32,
};

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: MatvecParams;
@group(0) @binding(5) var<storage, read> weights_f16: array<f16>;

var<workgroup> partial: array<f32, 256>;

fn unpack_i8(packed: u32, byte_index: u32) -> f32 {
  let byte = (packed >> (byte_index * 8u)) & 0xffu;
  if (byte >= 128u) {
    return f32(i32(byte) - 256);
  }
  return f32(byte);
}

@compute @workgroup_size(256)
fn q8_matvec(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x + wid.y * 65535u;
  let lane = lid.x;
  if (row >= p.rows) { return; }
  var sum = 0.0;
  var c = lane;
  loop {
    if (c >= p.cols) { break; }
    let element_index = row * p.cols + c;
    let q = unpack_i8(weights[element_index / 4u], element_index % 4u);
    sum = sum + q * x[p.x_offset + c];
    c = c + 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    y[p.y_offset + row] = partial[0] * scales[row];
  }
}

@compute @workgroup_size(256)
fn q8_embed_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= p.cols) { return; }
  let element_index = p.x_offset * p.cols + c;
  let q = unpack_i8(weights[element_index / 4u], element_index % 4u);
  y[p.y_offset + c] = q * scales[p.x_offset];
}

@compute @workgroup_size(256)
fn q8_tied_lm_head(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x + wid.y * 65535u;
  let lane = lid.x;
  if (row >= p.rows) { return; }
  var sum = 0.0;
  var c = lane;
  loop {
    if (c >= p.cols) { break; }
    let element_index = row * p.cols + c;
    let q = unpack_i8(weights[element_index / 4u], element_index % 4u);
    sum = sum + q * x[p.x_offset + c];
    c = c + 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    y[p.y_offset + row] = partial[0] * scales[row];
  }
}

@compute @workgroup_size(256)
fn f16_matvec(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x + wid.y * 65535u;
  let lane = lid.x;
  if (row >= p.rows) { return; }
  var sum = 0.0;
  var c = lane;
  loop {
    if (c >= p.cols) { break; }
    sum = sum + f32(weights_f16[row * p.cols + c]) * x[p.x_offset + c];
    c = c + 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    y[p.y_offset + row] = partial[0];
  }
}

@compute @workgroup_size(256)
fn f16_tied_lm_head(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x + wid.y * 65535u;
  let lane = lid.x;
  if (row >= p.rows) { return; }
  var sum = 0.0;
  var c = lane;
  loop {
    if (c >= p.cols) { break; }
    sum = sum + f32(weights_f16[row * p.cols + c]) * x[p.x_offset + c];
    c = c + 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    y[p.y_offset + row] = partial[0];
  }
}
`;

export const normShaderSource = /* wgsl */ `
enable f16;

struct NormParams {
  n: u32,
  heads: u32,
  head_dim: u32,
  x_offset: u32,
  y_offset: u32,
  eps: f32,
};

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: NormParams;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn rms_norm(@builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  var sum = 0.0;
  var i = lane;
  loop {
    if (i >= p.n) { break; }
    let v = x[p.x_offset + i];
    sum = sum + v * v;
    i = i + 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(partial[0] / f32(p.n) + p.eps);
  i = lane;
  loop {
    if (i >= p.n) { break; }
    y[p.y_offset + i] = x[p.x_offset + i] * inv * (1.0 + f32(weight[i]));
    i = i + 256u;
  }
}

@compute @workgroup_size(256)
fn rms_norm_heads(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wid.x;
  let lane = lid.x;
  if (head >= p.heads) { return; }
  let base = head * p.head_dim;
  var sum = 0.0;
  var d = lane;
  loop {
    if (d >= p.head_dim) { break; }
    let v = x[p.x_offset + base + d];
    sum = sum + v * v;
    d = d + 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(partial[0] / f32(p.head_dim) + p.eps);
  d = lane;
  loop {
    if (d >= p.head_dim) { break; }
    y[p.y_offset + base + d] = x[p.x_offset + base + d] * inv * (1.0 + f32(weight[d]));
    d = d + 256u;
  }
}
`;

export const attentionShaderSource = /* wgsl */ `
struct AttentionParams {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  rope_dim: u32,
  position: u32,
  seq_len: u32,
  rope_theta: f32,
};

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> k_now: array<f32>;
@group(0) @binding(5) var<storage, read_write> v_now: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> p: AttentionParams;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn split_q_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = p.n_heads * p.head_dim;
  if (i >= total) { return; }
  let h = i / p.head_dim;
  let d = i % p.head_dim;
  let src = h * p.head_dim * 2u + d;
  q[i] = out[src];
  gate[i] = out[src + p.head_dim];
}

@compute @workgroup_size(256)
fn write_kv_cache(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = p.n_kv_heads * p.head_dim;
  if (i >= total) { return; }
  let off = p.position * total + i;
  k_cache[off] = k_now[i];
  v_cache[off] = v_now[i];
}

@compute @workgroup_size(256)
fn rope_q(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair = gid.x;
  let half = p.rope_dim / 2u;
  if (pair >= p.n_heads * half) { return; }
  let h = pair / half;
  let i = pair % half;
  let base = h * p.head_dim;
  let inv_freq = pow(p.rope_theta, -f32(2u * i) / f32(p.rope_dim));
  let angle = f32(p.position) * inv_freq;
  let c = cos(angle);
  let s = sin(angle);
  let i0 = base + i;
  let i1 = base + i + half;
  let x0 = q[i0];
  let x1 = q[i1];
  q[i0] = x0 * c - x1 * s;
  q[i1] = x1 * c + x0 * s;
}

@compute @workgroup_size(256)
fn rope_k(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair = gid.x;
  let half = p.rope_dim / 2u;
  if (pair >= p.n_kv_heads * half) { return; }
  let h = pair / half;
  let i = pair % half;
  let base = p.position * p.n_kv_heads * p.head_dim + h * p.head_dim;
  let inv_freq = pow(p.rope_theta, -f32(2u * i) / f32(p.rope_dim));
  let angle = f32(p.position) * inv_freq;
  let c = cos(angle);
  let s = sin(angle);
  let i0 = base + i;
  let i1 = base + i + half;
  let x0 = k_cache[i0];
  let x1 = k_cache[i1];
  k_cache[i0] = x0 * c - x1 * s;
  k_cache[i1] = x1 * c + x0 * s;
}

@compute @workgroup_size(256)
fn full_attention(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let d = lid.x;
  if (h >= p.n_heads || d >= p.head_dim) { return; }
  let n_rep = p.n_heads / p.n_kv_heads;
  let kvh = h / n_rep;
  let q_base = h * p.head_dim;
  var max_score = -3.402823e38;
  let scale = inverseSqrt(f32(p.head_dim));
  var t = 0u;
  loop {
    if (t >= p.seq_len) { break; }
    let k_base = t * p.n_kv_heads * p.head_dim + kvh * p.head_dim;
    var dot = 0.0;
    var j = 0u;
    loop {
      if (j >= p.head_dim) { break; }
      dot = dot + q[q_base + j] * k_cache[k_base + j];
      j = j + 1u;
    }
    max_score = max(max_score, dot * scale);
    t = t + 1u;
  }
  var denom = 0.0;
  var acc = 0.0;
  t = 0u;
  loop {
    if (t >= p.seq_len) { break; }
    let k_base = t * p.n_kv_heads * p.head_dim + kvh * p.head_dim;
    var dot = 0.0;
    var j = 0u;
    loop {
      if (j >= p.head_dim) { break; }
      dot = dot + q[q_base + j] * k_cache[k_base + j];
      j = j + 1u;
    }
    let e = exp(dot * scale - max_score);
    denom = denom + e;
    let v_base = t * p.n_kv_heads * p.head_dim + kvh * p.head_dim;
    acc = acc + e * v_cache[v_base + d];
    t = t + 1u;
  }
  out[q_base + d] = (acc / max(denom, 1.0e-20)) * sigmoid(gate[q_base + d]);
}
`;

export const linearAttentionShaderSource = /* wgsl */ `
enable f16;

struct LinearParams {
  conv_channels: u32,
  q_dim: u32,
  v_dim: u32,
  heads: u32,
  head_dim: u32,
  kernel: u32,
  eps: f32,
  q_scale: f32,
};

@group(0) @binding(0) var<storage, read_write> mixed: array<f32>;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> a: array<f32>;
@group(0) @binding(4) var<storage, read> conv_w: array<f16>;
@group(0) @binding(5) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(6) var<storage, read_write> conv_out: array<f32>;
@group(0) @binding(7) var<storage, read> norm_w: array<f16>;
@group(0) @binding(8) var<storage, read> dt_bias: array<f16>;
@group(0) @binding(9) var<storage, read> a_log: array<f16>;
@group(0) @binding(10) var<storage, read_write> recurrent: array<f32>;
@group(0) @binding(11) var<storage, read_write> core_out: array<f32>;
@group(0) @binding(12) var<uniform> p: LinearParams;

var<workgroup> partial: array<f32, 128>;

fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

fn silu(x: f32) -> f32 {
  return x * sigmoid(x);
}

fn softplus(x: f32) -> f32 {
  if (x > 20.0) { return x; }
  if (x < -20.0) { return exp(x); }
  return log(1.0 + exp(x));
}

@compute @workgroup_size(256)
fn linear_conv_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= p.conv_channels) { return; }
  let s0 = conv_state[c];
  let s1 = conv_state[p.conv_channels + c];
  let s2 = conv_state[2u * p.conv_channels + c];
  let current = mixed[c];
  var sum = 0.0;
  sum = sum + s0 * f32(conv_w[c * p.kernel + 0u]);
  sum = sum + s1 * f32(conv_w[c * p.kernel + 1u]);
  sum = sum + s2 * f32(conv_w[c * p.kernel + 2u]);
  sum = sum + current * f32(conv_w[c * p.kernel + 3u]);
  conv_out[c] = silu(sum);
  conv_state[c] = s1;
  conv_state[p.conv_channels + c] = s2;
  conv_state[2u * p.conv_channels + c] = current;
}

@compute @workgroup_size(128)
fn l2_norm_qk(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wid.x;
  let which = wid.y;
  let lane = lid.x;
  if (head >= p.heads) { return; }
  let base = select(p.q_dim, 0u, which == 0u) + head * p.head_dim;
  var sum = 0.0;
  if (lane < p.head_dim) {
    let v = conv_out[base + lane];
    sum = v * v;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 64u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(partial[0] + p.eps);
  if (lane < p.head_dim) {
    var v = conv_out[base + lane] * inv;
    if (which == 0u) {
      v = v * p.q_scale;
    }
    conv_out[base + lane] = v;
  }
}

@compute @workgroup_size(128)
fn deltanet_update(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let i = wid.y;
  let lane = lid.x;
  if (h >= p.heads || i >= p.head_dim) { return; }

  let q_base = h * p.head_dim;
  let k_base = p.q_dim + h * p.head_dim;
  let v_base = 2u * p.q_dim + h * p.head_dim;
  let state_base = h * p.head_dim * p.head_dim + i * p.head_dim;

  let beta = sigmoid(b[h]);
  let alpha = exp(softplus(a[h] + f32(dt_bias[h])) * -exp(f32(a_log[h])));

  var sk_part = 0.0;
  if (lane < p.head_dim) {
    sk_part = recurrent[state_base + lane] * alpha * conv_out[k_base + lane];
  }
  partial[lane] = sk_part;
  workgroupBarrier();
  var stride = 64u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let sk = partial[0];
  let delta = (conv_out[v_base + i] - sk) * beta;

  var out_part = 0.0;
  if (lane < p.head_dim) {
    let updated = recurrent[state_base + lane] * alpha + delta * conv_out[k_base + lane];
    recurrent[state_base + lane] = updated;
    out_part = updated * conv_out[q_base + lane];
  }
  partial[lane] = out_part;
  workgroupBarrier();
  stride = 64u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lane == 0u) {
    core_out[h * p.head_dim + i] = partial[0];
  }
}

@compute @workgroup_size(128)
fn gated_rms(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let lane = lid.x;
  if (h >= p.heads) { return; }
  let base = h * p.head_dim;
  var sum = 0.0;
  if (lane < p.head_dim) {
    let v = core_out[base + lane];
    sum = v * v;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 64u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      partial[lane] = partial[lane] + partial[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(partial[0] / f32(p.head_dim) + p.eps);
  if (lane < p.head_dim) {
    let idx = base + lane;
    core_out[idx] = core_out[idx] * inv * f32(norm_w[lane]) * silu(z[idx]);
  }
}
`;

export const samplingShaderSource = /* wgsl */ `
struct MaskParams {
  n: u32,
  token0: u32,
  token1: u32,
  token2: u32,
};

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> p: MaskParams;

@compute @workgroup_size(1)
fn mask_tokens() {
  if (p.token0 < p.n) { logits[p.token0] = -3.402823e38; }
  if (p.token1 < p.n) { logits[p.token1] = -3.402823e38; }
  if (p.token2 < p.n) { logits[p.token2] = -3.402823e38; }
}
`;
