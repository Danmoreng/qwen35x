#pragma once

// Compile-time specialization for Qwen3.5-0.8B.
// Keep descriptor-derived runtime validation separate from these constants until
// additional compiled variants exist.

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;

// Full Attention
constexpr int FA_NUM_Q_HEADS = 8;
constexpr int FA_NUM_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA_RATIO = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
constexpr int FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_GATE_SIZE = FA_Q_SIZE;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE + FA_GATE_SIZE;
constexpr int FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROTARY_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_NUM_HEADS = 16;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 82
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int ATTENTION_WARP_SCRATCH = FA_GQA_RATIO * NUM_WARPS * FA_HEAD_DIM;
constexpr int BASE_MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;
constexpr int MAX_ACT_DIM = (BASE_MAX_ACT_DIM > ATTENTION_WARP_SCRATCH) ? BASE_MAX_ACT_DIM : ATTENTION_WARP_SCRATCH;
constexpr int MAX_DECODE_BLOCKS = 1024;
constexpr int MIN_DECODE_BLOCKS = DN_NUM_HEADS;

#ifndef LM_NUM_BLOCKS
#define LM_NUM_BLOCKS 512
#endif
#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif

#define QWEN35X_0P8B_LAYER_TYPE_VALUES \
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1

constexpr int QWEN35X_LAYER_TYPE_HOST[NUM_LAYERS] = { QWEN35X_0P8B_LAYER_TYPE_VALUES };

// Compatibility aliases for the prefill code. These are intentionally kept in
// the variant header so the next split can move kernels without renaming every
// dimension at once.
constexpr int HIDDEN = HIDDEN_SIZE;
constexpr int INTER = INTERMEDIATE_SIZE;
constexpr int VOCAB = VOCAB_SIZE;
constexpr int FA_Q_HEADS = FA_NUM_Q_HEADS;
constexpr int FA_KV_HEADS = FA_NUM_KV_HEADS;
constexpr int FA_GQA = FA_GQA_RATIO;
constexpr int FA_ROT_DIM = FA_ROTARY_DIM;
constexpr int DN_HEADS = DN_NUM_HEADS;
constexpr int DN_KEY = DN_KEY_DIM;
constexpr int DN_VAL = DN_VALUE_DIM;
constexpr int DN_CONV_K = DN_CONV_KERNEL;
constexpr int DN_CONV_CH = DN_CONV_CHANNELS;
