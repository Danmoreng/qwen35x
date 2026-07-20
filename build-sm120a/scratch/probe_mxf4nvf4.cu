extern "C" __global__ void probe(unsigned *out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  unsigned a0=0,a1=0,a2=0,a3=0;
  unsigned b0=0,b1=0;
  float c0=0.0f,c1=0.0f,c2=0.0f,c3=0.0f;
  unsigned d0=0,d1=0,d2=0,d3=0;
  unsigned scale_a=0, scale_b=0;
  unsigned short bid=0, tid=0;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13}, "
    "%14, {%16, %17}, "
    "%15, {%16, %17};\n"
    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(scale_a), "r"(scale_b), "h"(bid), "h"(tid));
  if (threadIdx.x == 0) out[0] = d0 + d1 + d2 + d3;
#endif
}
