// ============================================================
// WMMA Tensor Core matrix multiplication benchmark
// BF16 path : Optimized 128x128 double-buffered cp.async pipeline
// TF32 path : A/B = float (→tf32), acc = float, tile 16×16×8
// Target GPU: NVIDIA H100 (sm_90)
// ============================================================

#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// ─────────────────────────────────────────────────────────────
// Shared Kernel configuration
// ─────────────────────────────────────────────────────────────
#define WARP_SIZE 32

// BF16 WMMA tile
#define BF16_WMMA_M 16
#define BF16_WMMA_N 16
#define BF16_WMMA_K 16

// TF32 WMMA tile
#define TF32_WMMA_M 16
#define TF32_WMMA_N 16
#define TF32_WMMA_K 8

// ==== BF16 OPTIMIZED PARAMS ====
// Block output tile: 128 × 128
#define BF16_BLOCK_M 128
#define BF16_BLOCK_N 128
#define BF16_BLOCK_K 32 // Chunk loaded per step
#define BF16_WARPS_M 4
#define BF16_WARPS_N 2
#define BF16_THREADS (BF16_WARPS_M * BF16_WARPS_N * WARP_SIZE) // 256 threads (8 warps)

// ==== TF32 PARAMS ====
// Block output tile: 64 × 64 (keep original)
#define TF32_BLOCK_M 64
#define TF32_BLOCK_N 64
#define TF32_BLOCK_K 32
#define TF32_WARPS_M 4
#define TF32_WARPS_N 4
#define TF32_THREADS (TF32_WARPS_M * TF32_WARPS_N * WARP_SIZE) // 512 threads (16 warps)

// ─────────────────────────────────────────────────────────────
// PTX cp.async helpers
// ─────────────────────────────────────────────────────────────
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* global_ptr, bool mask) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int pred = mask ? 1 : 0;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n" ::"r"(smem_int), "l"(global_ptr), "r"(pred));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// ─────────────────────────────────────────────────────────────
// BF16 WMMA Kernel (Optimized Pipeline)
// ─────────────────────────────────────────────────────────────
__global__ void wmma_bf16(const __nv_bfloat16 *__restrict__ A,
                          const __nv_bfloat16 *__restrict__ B,
                          float *__restrict__ C, int N) {
  const int bRow = blockIdx.y * BF16_BLOCK_M;
  const int bCol = blockIdx.x * BF16_BLOCK_N;

  const int wid = threadIdx.x / WARP_SIZE;
  const int wRow = wid / BF16_WARPS_N; // 0..3
  const int wCol = wid % BF16_WARPS_N; // 0..1

  // Accumulators (2x4 array per warp to cover 32x64 output)
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  #pragma unroll
  for(int i=0; i<2; i++) {
    #pragma unroll
    for(int j=0; j<4; j++) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  // Double-buffered shared memory
  __shared__ __align__(16) __nv_bfloat16 sA[2][BF16_BLOCK_M][BF16_BLOCK_K];
  __shared__ __align__(16) __nv_bfloat16 sB[2][BF16_BLOCK_K][BF16_BLOCK_N];

  const int tid = threadIdx.x;
  
  // Helper to load async stage into shared mem
  auto load_stage = [&](int stage, int kBase) {
      // Load A (128x32 = 4096 elements = 512 chunks of 16 bytes). 
      // Threads = 256, so 2 chunks per thread.
      #pragma unroll
      for (int i = 0; i < 2; i++) {
          int chunk_idx = tid + i * BF16_THREADS;
          int elem_idx = chunk_idx * 8; // 8 x BF16 = 16 bytes
          int lm = elem_idx / BF16_BLOCK_K;
          int lk = elem_idx % BF16_BLOCK_K;
          int gm = bRow + lm;
          int gk = kBase + lk;
          bool mask = (gm < N && gk < N);
          cp_async_16B(&sA[stage][lm][lk], &A[gm * N + gk], mask);
      }
      
      // Load B (32x128 = 4096 elements). Same, 2 chunks per thread.
      #pragma unroll
      for (int i = 0; i < 2; i++) {
          int chunk_idx = tid + i * BF16_THREADS;
          int elem_idx = chunk_idx * 8;
          int lk = elem_idx / BF16_BLOCK_N;
          int ln = elem_idx % BF16_BLOCK_N;
          int gk = kBase + lk;
          int gn = bCol + ln;
          bool mask = (gk < N && gn < N);
          cp_async_16B(&sB[stage][lk][ln], &B[gk * N + gn], mask);
      }
      cp_async_commit();
  };

  const int kSteps = (N + BF16_BLOCK_K - 1) / BF16_BLOCK_K;
  if (kSteps == 0) return;

  // 1. Prologue: Launch loads for Stage 0
  load_stage(0, 0);

  // 2. Main compute loop
  int compute_stage = 0;
  for (int ks = 0; ks < kSteps; ks++) {
      int next_stage = compute_stage ^ 1;
      
      // Request next stage load
      if (ks + 1 < kSteps) {
          load_stage(next_stage, (ks + 1) * BF16_BLOCK_K);
      }
      
      // Wait for CURRENT stage loads to complete
      cp_async_wait();
      __syncthreads();
      
      // Compute over current stage
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> aFrag[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bFrag[4];

      #pragma unroll
      for (int ki = 0; ki < BF16_BLOCK_K; ki += 16) {
          // Load 2 rows of A fragments for this warp
          wmma::load_matrix_sync(aFrag[0], &sA[compute_stage][wRow * 32 + 0][ki], BF16_BLOCK_K);
          wmma::load_matrix_sync(aFrag[1], &sA[compute_stage][wRow * 32 + 16][ki], BF16_BLOCK_K);
          
          // Load 4 columns of B fragments for this warp
          wmma::load_matrix_sync(bFrag[0], &sB[compute_stage][ki][wCol * 64 + 0], BF16_BLOCK_N);
          wmma::load_matrix_sync(bFrag[1], &sB[compute_stage][ki][wCol * 64 + 16], BF16_BLOCK_N);
          wmma::load_matrix_sync(bFrag[2], &sB[compute_stage][ki][wCol * 64 + 32], BF16_BLOCK_N);
          wmma::load_matrix_sync(bFrag[3], &sB[compute_stage][ki][wCol * 64 + 48], BF16_BLOCK_N);

          // Multiply-accumulate outer product for the 2x4 tiles
          #pragma unroll
          for (int i = 0; i < 2; i++) {
              #pragma unroll
              for (int j = 0; j < 4; j++) {
                  wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
              }
          }
      }
      
      // Ensure compute finishes consuming shared mem before next loop overwrites it
      __syncthreads(); 
      compute_stage = next_stage;
  }

  // 3. Epilogue: Store accumulators to global memory
  #pragma unroll
  for(int i=0; i<2; i++) {
      #pragma unroll
      for(int j=0; j<4; j++) {
          int cRow = bRow + wRow * 32 + i * 16;
          int cCol = bCol + wCol * 64 + j * 16;
          if (cRow < N && cCol < N) {
              wmma::store_matrix_sync(C + cRow * N + cCol, acc[i][j], N, wmma::mem_row_major);
          }
      }
  }
}

// ─────────────────────────────────────────────────────────────
// TF32 WMMA Kernel  (float input, TF32 Tensor Core compute)
// ─────────────────────────────────────────────────────────────
__global__ void wmma_tf32(const float *__restrict__ A,
                          const float *__restrict__ B, float *__restrict__ C,
                          int N) {
  const int bRow = blockIdx.y * TF32_BLOCK_M;
  const int bCol = blockIdx.x * TF32_BLOCK_N;

  const int wid = threadIdx.x / WARP_SIZE;
  const int wRow = wid / TF32_WARPS_N;
  const int wCol = wid % TF32_WARPS_N;

  wmma::fragment<wmma::accumulator, TF32_WMMA_M, TF32_WMMA_N, TF32_WMMA_K,
                 float>
      acc;
  wmma::fill_fragment(acc, 0.0f);

  __shared__ float sA[TF32_BLOCK_M][TF32_BLOCK_K]; // 64×32 floats = 8 KB
  __shared__ float sB[TF32_BLOCK_K][TF32_BLOCK_N]; // 32×64 floats = 8 KB

  const int tid = threadIdx.x;
  const int kSteps = (N + TF32_BLOCK_K - 1) / TF32_BLOCK_K;

  for (int ks = 0; ks < kSteps; ks++) {
    const int kBase = ks * TF32_BLOCK_K;

    for (int i = tid; i < TF32_BLOCK_M * TF32_BLOCK_K; i += TF32_THREADS) {
      int lm = i / TF32_BLOCK_K, lk = i % TF32_BLOCK_K;
      int gm = bRow + lm, gk = kBase + lk;
      sA[lm][lk] = (gm < N && gk < N) ? A[gm * N + gk] : 0.0f;
    }

    for (int i = tid; i < TF32_BLOCK_K * TF32_BLOCK_N; i += TF32_THREADS) {
      int lk = i / TF32_BLOCK_N, ln = i % TF32_BLOCK_N;
      int gk = kBase + lk, gn = bCol + ln;
      sB[lk][ln] = (gk < N && gn < N) ? B[gk * N + gn] : 0.0f;
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, TF32_WMMA_M, TF32_WMMA_N, TF32_WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, TF32_WMMA_M, TF32_WMMA_N, TF32_WMMA_K,
                   wmma::precision::tf32, wmma::row_major>
        bFrag;

#pragma unroll
    for (int ki = 0; ki < TF32_BLOCK_K; ki += TF32_WMMA_K) {
      wmma::load_matrix_sync(aFrag, &sA[wRow * TF32_WMMA_M][ki], TF32_BLOCK_K);
      wmma::load_matrix_sync(bFrag, &sB[ki][wCol * TF32_WMMA_N], TF32_BLOCK_N);
      wmma::mma_sync(acc, aFrag, bFrag, acc);
    }

    __syncthreads();
  }

  const int cRow = bRow + wRow * TF32_WMMA_M;
  const int cCol = bCol + wCol * TF32_WMMA_N;
  if (cRow < N && cCol < N)
    wmma::store_matrix_sync(C + cRow * N + cCol, acc, N, wmma::mem_row_major);
}

// ─────────────────────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────────────────────
static float bench(int warmup, int iters, void (*fn)(void *), void *arg) {
  for (int i = 0; i < warmup; i++)
    fn(arg);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));
  for (int i = 0; i < iters; i++)
    fn(arg);
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return ms / iters;
}

struct BF16Args {
  __nv_bfloat16 *A, *B;
  float *C;
  int N;
};

static void launch_bf16(void *p) {
  auto *a = (BF16Args *)p;
  dim3 blk(BF16_THREADS);
  dim3 grd((a->N + BF16_BLOCK_N - 1) / BF16_BLOCK_N, (a->N + BF16_BLOCK_M - 1) / BF16_BLOCK_M);
  wmma_bf16<<<grd, blk>>>(a->A, a->B, a->C, a->N);
}

struct TF32Args {
  float *A, *B, *C;
  int N;
};

static void launch_tf32(void *p) {
  auto *a = (TF32Args *)p;
  dim3 blk(TF32_THREADS);
  dim3 grd((a->N + TF32_BLOCK_N - 1) / TF32_BLOCK_N, (a->N + TF32_BLOCK_M - 1) / TF32_BLOCK_M);
  wmma_tf32<<<grd, blk>>>(a->A, a->B, a->C, a->N);
}

int main(void) {
  printDeviceInfo();

  const int SIZES[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
  const int NSIZES = (int)(sizeof(SIZES) / sizeof(SIZES[0]));
  const int WARMUP = 5;
  const int ITERS = 20;

  printf("dtype,N,ms,tflops\n");

  for (int si = 0; si < NSIZES; si++) {
    int N = SIZES[si];
    size_t bytes_bf16 = (size_t)N * N * sizeof(__nv_bfloat16);
    size_t bytes_f32 = (size_t)N * N * sizeof(float);

    __nv_bfloat16 *dA16, *dB16;
    float *dC16;
    float *dA32, *dB32, *dC32;

    CUDA_CHECK(cudaMalloc(&dA16, bytes_bf16));
    CUDA_CHECK(cudaMalloc(&dB16, bytes_bf16));
    CUDA_CHECK(cudaMalloc(&dC16, bytes_f32));
    CUDA_CHECK(cudaMalloc(&dA32, bytes_f32));
    CUDA_CHECK(cudaMalloc(&dB32, bytes_f32));
    CUDA_CHECK(cudaMalloc(&dC32, bytes_f32));

    {
      __nv_bfloat16 *hA16 = (__nv_bfloat16 *)malloc(bytes_bf16);
      __nv_bfloat16 *hB16 = (__nv_bfloat16 *)malloc(bytes_bf16);
      float *hA32 = (float *)malloc(bytes_f32);
      float *hB32 = (float *)malloc(bytes_f32);
      for (int i = 0; i < N * N; i++) {
        float v = (float)(rand() % 10) / 10.0f;
        hA16[i] = __float2bfloat16(v);
        hB16[i] = __float2bfloat16(v);
        hA32[i] = v;
        hB32[i] = v;
      }
      CUDA_CHECK(cudaMemcpy(dA16, hA16, bytes_bf16, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB16, hB16, bytes_bf16, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dA32, hA32, bytes_f32, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB32, hB32, bytes_f32, cudaMemcpyHostToDevice));
      free(hA16);
      free(hB16);
      free(hA32);
      free(hB32);
    }

    double flops = 2.0 * (double)N * N * N;

    {
      BF16Args arg = {dA16, dB16, dC16, N};
      float ms = bench(WARMUP, ITERS, launch_bf16, &arg);
      printf("BF16,%d,%.4f,%.2f\n", N, ms, flops / (ms * 1e-3) / 1e12);
    }

    {
      TF32Args arg = {dA32, dB32, dC32, N};
      float ms = bench(WARMUP, ITERS, launch_tf32, &arg);
      printf("FP32,%d,%.4f,%.2f\n", N, ms, flops / (ms * 1e-3) / 1e12);
    }

    CUDA_CHECK(cudaFree(dA16));
    CUDA_CHECK(cudaFree(dB16));
    CUDA_CHECK(cudaFree(dC16));
    CUDA_CHECK(cudaFree(dA32));
    CUDA_CHECK(cudaFree(dB32));
    CUDA_CHECK(cudaFree(dC32));
  }

  return 0;
}
