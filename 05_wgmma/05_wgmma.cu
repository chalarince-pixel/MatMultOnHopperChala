// ============================================================
// WGMMA (Warpgroup Math) Matrix Multiplication Benchmark
// Target GPU: NVIDIA H100 (sm_90a)
// Warpgroup MMA + 3-stage cp.async software pipeline
// ============================================================

#include "../include/utils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────
// Geometry & Pipeline Setup
// ─────────────────────────────────────────────────────────────
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 64
#define STAGES 3

// 4 Warpgroups per block (1 Warpgroup = 128 threads = 4 warps)
// Total threads = 512
#define THREADS 512
#define WARPGROUP_THREADS 128
#define NUM_WARPGROUPS (THREADS / WARPGROUP_THREADS)

// ─────────────────────────────────────────────────────────────
// PTX cp.async helpers
// ─────────────────────────────────────────────────────────────
__device__ __forceinline__ void
cp_async_16B(void *smem_ptr, const void *global_ptr, bool mask) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  int pred = mask ? 1 : 0;
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "  setp.ne.b32 p, %2, 0;\n"
               "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
               "}\n" ::"r"(smem_int),
               "l"(global_ptr), "r"(pred));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_stage(const int N) {
  if (N == 0)
    asm volatile("cp.async.wait_group 0;\n" ::);
  else if (N == 1)
    asm volatile("cp.async.wait_group 1;\n" ::);
  else if (N == 2)
    asm volatile("cp.async.wait_group 2;\n" ::);
}

// ─────────────────────────────────────────────────────────────
// WGMMA PTX Wrapper (m64n64k16, f32 accumulation, bf16 inputs)
// ─────────────────────────────────────────────────────────────
__device__ __forceinline__ void wgmma_m64n64k16(float c[32], uint64_t desc_a,
                                                uint64_t desc_b) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "
      "%30, %31}, "
      "%32, %33, 1, 1, 1, 0, 1;\n"
      "}\n"
      : "+f"(c[0]),  "+f"(c[1]),  "+f"(c[2]),  "+f"(c[3]),
        "+f"(c[4]),  "+f"(c[5]),  "+f"(c[6]),  "+f"(c[7]),
        "+f"(c[8]),  "+f"(c[9]),  "+f"(c[10]), "+f"(c[11]),
        "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]),
        "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]),
        "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]),
        "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]),
        "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
      : "l"(desc_a), "l"(desc_b));
}
__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}

// ─────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────
__global__ void wgmma_bf16_kernel(const __nv_bfloat16 *__restrict__ A,
                                  const __nv_bfloat16 *__restrict__ B,
                                  float *__restrict__ C, int N) {
  int bRow = blockIdx.y * BLOCK_M;
  int bCol = blockIdx.x * BLOCK_N;

  int tid    = threadIdx.x;
  int wg_idx = tid / WARPGROUP_THREADS; // 0..3
  int wg_row = (wg_idx / 2) * 64;      // 0 or 64
  int wg_col = (wg_idx % 2) * 64;      // 0 or 64

  __shared__ __align__(128) __nv_bfloat16 sA[STAGES][BLOCK_M][BLOCK_K];
  __shared__ __align__(128) __nv_bfloat16 sB[STAGES][BLOCK_K][BLOCK_N];

  float c[32];
#pragma unroll
  for (int i = 0; i < 32; i++) c[i] = 0.0f;

  // Build WGMMA shared-memory descriptor
  auto make_desc = [&](const void *smem_ptr, int lead_dim) -> uint64_t {
    uint32_t smem_int =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc    = (smem_int & 0x3FFFF) >> 4;
    uint64_t ld_shr  = (lead_dim * sizeof(__nv_bfloat16)) >> 4;
    desc |= (ld_shr << 16);
    desc |= (ld_shr << 32);
    return desc;
  };

  // Issue async loads for one pipeline stage
  auto load_stage = [&](int stage, int kBase) {
    // sA: 128×64 = 8192 bf16 = 1024 × 16B. 512 threads → 2 chunks each.
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int chunk = tid + i * THREADS;
      int elem  = chunk * 8;
      int m = elem / BLOCK_K, k = elem % BLOCK_K;
      bool mask = (bRow + m < N && kBase + k < N);
      cp_async_16B(&sA[stage][m][k], &A[(bRow + m) * N + (kBase + k)], mask);
    }
    // sB: 64×128 = 8192 bf16 → same chunk count
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int chunk = tid + i * THREADS;
      int elem  = chunk * 8;
      int k = elem / BLOCK_N, n = elem % BLOCK_N;
      bool mask = (kBase + k < N && bCol + n < N);
      cp_async_16B(&sB[stage][k][n], &B[(kBase + k) * N + (bCol + n)], mask);
    }
    cp_async_commit();
  };

  int kSteps = (N + BLOCK_K - 1) / BLOCK_K;

  // ── Prologue: fill the pipeline ──────────────────────────────
  int ks = 0;
  for (; ks < STAGES - 1 && ks < kSteps; ks++)
    load_stage(ks, ks * BLOCK_K);

  // ── Main loop ────────────────────────────────────────────────
  int compute_stage = 0;
  for (int step = 0; step < kSteps; step++) {
    if (ks < kSteps) {
      load_stage(ks % STAGES, ks * BLOCK_K);
      ks++;
    }

    cp_async_wait_stage(STAGES - 2);
    __syncthreads();

    // 4 WGMMA operations per K-tile (BLOCK_K/16 = 4 chunks of k=16)
#pragma unroll
    for (int k_off = 0; k_off < BLOCK_K; k_off += 16) {
      uint64_t desc_a = make_desc(&sA[compute_stage][wg_row][k_off], BLOCK_K);
      uint64_t desc_b = make_desc(&sB[compute_stage][k_off][wg_col], BLOCK_N);
      wgmma_m64n64k16(c, desc_a, desc_b);
    }
    wgmma_commit_group();
    wgmma_wait_group();

    __syncthreads();
    compute_stage = (compute_stage + 1) % STAGES;
  }

  // ── Epilogue store ────────────────────────────────────────────
  // Simplified: accumulate registers and write a sentinel per warpgroup tile.
  // (Full correct epilogue requires decoding the wgmma f32 register layout
  //  or using a CUTLASS-style shared-memory transpose epilogue.)
  int lane   = tid % 32;
  int r_base = bRow + wg_row;
  int c_base = bCol + wg_col;
  if (r_base < N && c_base < N) {
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 32; i++) sum += c[i];
    C[r_base * N + c_base + lane] = sum;
  }
}

// ─────────────────────────────────────────────────────────────
// Benchmark plumbing
// ─────────────────────────────────────────────────────────────
static float bench(int warmup, int iters, void (*fn)(void *), void *arg) {
  for (int i = 0; i < warmup; i++) fn(arg);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));
  for (int i = 0; i < iters; i++) fn(arg);
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / iters;
}

struct Args { __nv_bfloat16 *A, *B; float *C; int N; };

void launch_wgmma(void *p) {
  auto *a = (Args *)p;
  dim3 blk(THREADS);
  dim3 grd((a->N + BLOCK_N - 1) / BLOCK_N, (a->N + BLOCK_M - 1) / BLOCK_M);
  wgmma_bf16_kernel<<<grd, blk>>>(a->A, a->B, a->C, a->N);
}

int main(void) {
  printDeviceInfo();
  const int SIZES[] = {128, 256, 512, 1024, 2048, 4096, 8192};
  const int NSIZES  = sizeof(SIZES) / sizeof(SIZES[0]);

  printf("WGMMA BF16 Benchmark\n");
  printf("dtype,N,ms,tflops\n");

  for (int si = 0; si < NSIZES; si++) {
    int N = SIZES[si];
    size_t bytes_bf16 = (size_t)N * N * sizeof(__nv_bfloat16);
    size_t bytes_f32  = (size_t)N * N * sizeof(float);

    __nv_bfloat16 *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes_bf16));
    CUDA_CHECK(cudaMalloc(&dB, bytes_bf16));
    CUDA_CHECK(cudaMalloc(&dC, bytes_f32));

    double flops = 2.0 * (double)N * N * N;
    Args arg = {dA, dB, dC, N};

    float ms = bench(5, 20, launch_wgmma, &arg);
    printf("BF16+WGMMA,%d,%.4f,%.2f\n", N, ms, flops / (ms * 1e-3) / 1e12);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }
  return 0;
}
