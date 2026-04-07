// ============================================================
// WGMMA + DSM (Distributed Shared Memory) Cluster Kernel
// Target GPU: NVIDIA H100 (sm_90a)
//
// Experiment: can we improve on the baseline WGMMA kernel by
// using Thread Block Clusters + DSM to halve A-tile global
// memory traffic?
//
// SHORT ANSWER: Not with cp.async + cluster.sync().
// See README_DSM.md for the full explanation and the correct
// path (TMA + mbarrier) that CUTLASS 3.x uses.
// ============================================================

#include "../include/utils.cuh"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────
// Geometry & Pipeline
// ─────────────────────────────────────────────────────────────
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 64
#define STAGES 3

#define THREADS 512
#define WARPGROUP_THREADS 128

// Cluster: 1×2 along X (columns). Adjacent column-tile blocks
// cover the same row range → identical A tiles every K-step.
#define CLUSTER_DIM_X 2
#define CLUSTER_DIM_Y 1

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
               "}\n" ::"r"(smem_int), "l"(global_ptr), "r"(pred));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_stage(const int N) {
  if (N == 0)      asm volatile("cp.async.wait_group 0;\n" ::);
  else if (N == 1) asm volatile("cp.async.wait_group 1;\n" ::);
  else if (N == 2) asm volatile("cp.async.wait_group 2;\n" ::);
}

// ─────────────────────────────────────────────────────────────
// WGMMA PTX Wrapper (m64n64k16)
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
// Standalone WGMMA kernel (no cluster) — baseline reference
// ─────────────────────────────────────────────────────────────
__global__ void wgmma_bf16_kernel(const __nv_bfloat16 *__restrict__ A,
                                  const __nv_bfloat16 *__restrict__ B,
                                  float *__restrict__ C, int N) {
  int bRow = blockIdx.y * BLOCK_M;
  int bCol = blockIdx.x * BLOCK_N;
  int tid  = threadIdx.x;
  int wg_idx = tid / WARPGROUP_THREADS;
  int wg_row = (wg_idx / 2) * 64;
  int wg_col = (wg_idx % 2) * 64;

  __shared__ __align__(128) __nv_bfloat16 sA[STAGES][BLOCK_M][BLOCK_K];
  __shared__ __align__(128) __nv_bfloat16 sB[STAGES][BLOCK_K][BLOCK_N];

  float c[32];
#pragma unroll
  for (int i = 0; i < 32; i++) c[i] = 0.0f;

  auto make_desc = [&](const void *smem_ptr, int lead_dim) -> uint64_t {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc   = (smem_int & 0x3FFFF) >> 4;
    uint64_t ld_shr = (lead_dim * sizeof(__nv_bfloat16)) >> 4;
    desc |= (ld_shr << 16);
    desc |= (ld_shr << 32);
    return desc;
  };

  auto load_stage = [&](int stage, int kBase) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int chunk = tid + i * THREADS, elem = chunk * 8;
      int m = elem / BLOCK_K, k = elem % BLOCK_K;
      cp_async_16B(&sA[stage][m][k], &A[(bRow + m) * N + (kBase + k)],
                   (bRow + m < N && kBase + k < N));
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int chunk = tid + i * THREADS, elem = chunk * 8;
      int k = elem / BLOCK_N, n = elem % BLOCK_N;
      cp_async_16B(&sB[stage][k][n], &B[(kBase + k) * N + (bCol + n)],
                   (kBase + k < N && bCol + n < N));
    }
    cp_async_commit();
  };

  int kSteps = (N + BLOCK_K - 1) / BLOCK_K;
  int ks = 0;
  for (; ks < STAGES - 1 && ks < kSteps; ks++) load_stage(ks, ks * BLOCK_K);

  int compute_stage = 0;
  for (int step = 0; step < kSteps; step++) {
    if (ks < kSteps) { load_stage(ks % STAGES, ks * BLOCK_K); ks++; }
    cp_async_wait_stage(STAGES - 2);
    __syncthreads();
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
// WGMMA + DSM Cluster Kernel  (1×2 cluster, A-tile via DSM)
// ─────────────────────────────────────────────────────────────
//
// Cluster (1×2 along X):
//   rank 0 (even blockIdx.x): loads A from global memory
//   rank 1 (odd  blockIdx.x): reads A via DSM into rank 0's shmem
//   both ranks: load their own B tile (different column ranges)
//
// Cross-block synchronisation uses cluster.sync() to ensure rank 0's
// cp.async writes to sA are visible to rank 1 before WGMMA.
//
// ⚠️  PERFORMANCE NOTE: see README_DSM.md
//     cluster.sync() costs ~660 ns per call. With BLOCK_K=64 and 2 syncs
//     per K-step, the barrier alone is ~170 ms for N=8192 — 75× the 2.25 ms
//     baseline. This is why this variant is SLOWER, not faster.
//     The correct path is TMA + mbarrier (see 06_wgmma_tma/).
// ─────────────────────────────────────────────────────────────
__global__ void wgmma_bf16_cluster_kernel(const __nv_bfloat16 *__restrict__ A,
                                          const __nv_bfloat16 *__restrict__ B,
                                          float *__restrict__ C, int N) {
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterRank  = cluster.block_rank();

  int bRow   = blockIdx.y * BLOCK_M;
  int bCol   = blockIdx.x * BLOCK_N;
  int tid    = threadIdx.x;
  int wg_idx = tid / WARPGROUP_THREADS;
  int wg_row = (wg_idx / 2) * 64;
  int wg_col = (wg_idx % 2) * 64;

  __shared__ __align__(128) __nv_bfloat16 sA[STAGES][BLOCK_M][BLOCK_K];
  __shared__ __align__(128) __nv_bfloat16 sB[STAGES][BLOCK_K][BLOCK_N];

  float c[32];
#pragma unroll
  for (int i = 0; i < 32; i++) c[i] = 0.0f;

  auto make_desc = [&](const void *smem_ptr, int lead_dim) -> uint64_t {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc   = (smem_int & 0x3FFFF) >> 4;
    uint64_t ld_shr = (lead_dim * sizeof(__nv_bfloat16)) >> 4;
    desc |= (ld_shr << 16);
    desc |= (ld_shr << 32);
    return desc;
  };

  // Rank 1's DSM window into rank 0's sA array
  using sA_stage_t = __nv_bfloat16[BLOCK_M][BLOCK_K];
  sA_stage_t *sA_r0 = (sA_stage_t *)cluster.map_shared_rank(&sA[0], 0);

  auto load_stage = [&](int slot, int kBase) {
    // Only rank 0 loads A
    if (clusterRank == 0) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int chunk = tid + i * THREADS, elem = chunk * 8;
        int m = elem / BLOCK_K, k = elem % BLOCK_K;
        cp_async_16B(&sA[slot][m][k], &A[(bRow + m) * N + (kBase + k)],
                     (bRow + m < N && kBase + k < N));
      }
    }
    // Both ranks load their own B column tile
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int chunk = tid + i * THREADS, elem = chunk * 8;
      int k = elem / BLOCK_N, n = elem % BLOCK_N;
      cp_async_16B(&sB[slot][k][n], &B[(kBase + k) * N + (bCol + n)],
                   (kBase + k < N && bCol + n < N));
    }
    cp_async_commit();
  };

  int kSteps = (N + BLOCK_K - 1) / BLOCK_K;
  int ks = 0;
  for (; ks < STAGES - 1 && ks < kSteps; ks++) load_stage(ks, ks * BLOCK_K);

  int compute_stage = 0;
  for (int step = 0; step < kSteps; step++) {
    if (ks < kSteps) { load_stage(ks % STAGES, ks * BLOCK_K); ks++; }

    cp_async_wait_stage(STAGES - 2);
    __syncthreads();

    // ⚡ 660 ns stall — ensures rank 0's sA is visible to rank 1 via DSM.
    // This call dominates runtime. See README_DSM.md for why.
    cluster.sync();

    // rank 0 → local sA;  rank 1 → DSM pointer into rank 0's sA
    const __nv_bfloat16 (*sA_cur)[BLOCK_K] =
        (clusterRank == 0) ? sA[compute_stage] : sA_r0[compute_stage];

#pragma unroll
    for (int k_off = 0; k_off < BLOCK_K; k_off += 16) {
      uint64_t desc_a = make_desc(&sA_cur[wg_row][k_off], BLOCK_K);
      uint64_t desc_b = make_desc(&sB[compute_stage][k_off][wg_col], BLOCK_N);
      wgmma_m64n64k16(c, desc_a, desc_b);
    }
    wgmma_commit_group();
    wgmma_wait_group();
    __syncthreads();

    // ⚡ Second 660 ns stall — prevents rank 0 from overwriting sA while
    // rank 1 may still reference it. Eliminates any pipelining benefit.
    cluster.sync();

    compute_stage = (compute_stage + 1) % STAGES;
  }

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

void launch_baseline(void *p) {
  auto *a = (Args *)p;
  dim3 blk(THREADS);
  dim3 grd((a->N + BLOCK_N - 1) / BLOCK_N, (a->N + BLOCK_M - 1) / BLOCK_M);
  wgmma_bf16_kernel<<<grd, blk>>>(a->A, a->B, a->C, a->N);
}

void launch_cluster(void *p) {
  auto *a = (Args *)p;
  dim3 blk(THREADS);
  dim3 grd((a->N + BLOCK_N - 1) / BLOCK_N, (a->N + BLOCK_M - 1) / BLOCK_M);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grd; cfg.blockDim = blk;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {CLUSTER_DIM_X, CLUSTER_DIM_Y, 1};
  cfg.attrs = &attr; cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, wgmma_bf16_cluster_kernel,
                                a->A, a->B, a->C, a->N));
}

int main(void) {
  printDeviceInfo();
  // Min N = BLOCK_N × CLUSTER_DIM_X = 256
  const int SIZES[] = {256, 512, 1024, 2048, 4096, 8192};
  const int NSIZES  = sizeof(SIZES) / sizeof(SIZES[0]);

  printf("WGMMA + DSM Cluster Experiment  (1×2 cluster, cp.async sync)\n");
  printf("See README_DSM.md for why DSM is slower here.\n\n");
  printf("dtype,N,ms_baseline,tflops_baseline,ms_dsm,tflops_dsm,slowdown\n");

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

    float ms_b = bench(5, 20, launch_baseline, &arg);
    float ms_d = bench(5, 20, launch_cluster,  &arg);

    printf("BF16,%d,%.4f,%.2f,%.4f,%.2f,%.2f×\n",
           N,
           ms_b, flops / (ms_b * 1e-3) / 1e12,
           ms_d, flops / (ms_d * 1e-3) / 1e12,
           ms_d / ms_b);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }
  return 0;
}
