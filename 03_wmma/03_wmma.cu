// ============================================================
// WMMA Tensor Core matrix multiplication benchmark
// BF16 path : A/B = __nv_bfloat16, acc = float, tile 16×16×16
// TF32 path : A/B = float (→tf32),  acc = float, tile 16×16×8
// Target GPU: NVIDIA H100 (sm_90)
// ============================================================

#include "utils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// ─────────────────────────────────────────────────────────────
// Kernel configuration
// ─────────────────────────────────────────────────────────────
#define WARP_SIZE 32

// BF16 WMMA tile: 16×16×16
#define BF16_WMMA_M 16
#define BF16_WMMA_N 16
#define BF16_WMMA_K 16

// TF32 WMMA tile: 16×16×8
#define TF32_WMMA_M 16
#define TF32_WMMA_N 16
#define TF32_WMMA_K 8

// Block warp layout: 4 rows × 4 cols = 16 warps = 512 threads
// Block output tile: 64 × 64
#define WARPS_M 4
#define WARPS_N 4
#define BLOCK_M (WARPS_M * 16)                  // 64
#define BLOCK_N (WARPS_N * 16)                  // 64
#define THREADS (WARPS_M * WARPS_N * WARP_SIZE) // 512

// K chunks loaded into shmem per outer loop step
#define BF16_BLOCK_K 64 // 4 × 16
#define TF32_BLOCK_K 32 // 4 × 8

// ─────────────────────────────────────────────────────────────
// BF16 WMMA Kernel
// ─────────────────────────────────────────────────────────────
__global__ void wmma_bf16(const __nv_bfloat16 *__restrict__ A,
                          const __nv_bfloat16 *__restrict__ B,
                          float *__restrict__ C, int N) {
  const int bRow = blockIdx.y * BLOCK_M;
  const int bCol = blockIdx.x * BLOCK_N;

  const int wid = threadIdx.x / WARP_SIZE;
  const int wRow = wid / WARPS_N;
  const int wCol = wid % WARPS_N;

  // Warp accumulator
  wmma::fragment<wmma::accumulator, BF16_WMMA_M, BF16_WMMA_N, BF16_WMMA_K,
                 float>
      acc;
  wmma::fill_fragment(acc, 0.0f);

  // Shared memory tiles — single buffer (16 KB total)
  __shared__ __nv_bfloat16 sA[BLOCK_M][BF16_BLOCK_K];
  __shared__ __nv_bfloat16 sB[BF16_BLOCK_K][BLOCK_N];

  const int tid = threadIdx.x;
  const int kSteps = (N + BF16_BLOCK_K - 1) / BF16_BLOCK_K;

  for (int ks = 0; ks < kSteps; ks++) {
    const int kBase = ks * BF16_BLOCK_K;

    // ── Cooperatively load A tile [BLOCK_M × BF16_BLOCK_K] ──
    for (int i = tid; i < BLOCK_M * BF16_BLOCK_K; i += THREADS) {
      int lm = i / BF16_BLOCK_K, lk = i % BF16_BLOCK_K;
      int gm = bRow + lm, gk = kBase + lk;
      sA[lm][lk] = (gm < N && gk < N) ? A[gm * N + gk] : __float2bfloat16(0.0f);
    }

    // ── Cooperatively load B tile [BF16_BLOCK_K × BLOCK_N] ──
    for (int i = tid; i < BF16_BLOCK_K * BLOCK_N; i += THREADS) {
      int lk = i / BLOCK_N, ln = i % BLOCK_N;
      int gk = kBase + lk, gn = bCol + ln;
      sB[lk][ln] = (gk < N && gn < N) ? B[gk * N + gn] : __float2bfloat16(0.0f);
    }

    __syncthreads();

    // ── Inner K loop: 4 WMMA ops per warp over loaded tile ──
    wmma::fragment<wmma::matrix_a, BF16_WMMA_M, BF16_WMMA_N, BF16_WMMA_K,
                   __nv_bfloat16, wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, BF16_WMMA_M, BF16_WMMA_N, BF16_WMMA_K,
                   __nv_bfloat16, wmma::row_major>
        bFrag;

#pragma unroll
    for (int ki = 0; ki < BF16_BLOCK_K; ki += BF16_WMMA_K) {
      wmma::load_matrix_sync(aFrag, &sA[wRow * BF16_WMMA_M][ki], BF16_BLOCK_K);
      wmma::load_matrix_sync(bFrag, &sB[ki][wCol * BF16_WMMA_N], BLOCK_N);
      wmma::mma_sync(acc, aFrag, bFrag, acc);
    }

    __syncthreads();
  }

  // ── Store accumulator ─────────────────────────────────────
  const int cRow = bRow + wRow * BF16_WMMA_M;
  const int cCol = bCol + wCol * BF16_WMMA_N;
  if (cRow < N && cCol < N)
    wmma::store_matrix_sync(C + cRow * N + cCol, acc, N, wmma::mem_row_major);
}

// ─────────────────────────────────────────────────────────────
// TF32 WMMA Kernel  (float input, TF32 Tensor Core compute)
// ─────────────────────────────────────────────────────────────
__global__ void wmma_tf32(const float *__restrict__ A,
                          const float *__restrict__ B, float *__restrict__ C,
                          int N) {
  const int bRow = blockIdx.y * BLOCK_M;
  const int bCol = blockIdx.x * BLOCK_N;

  const int wid = threadIdx.x / WARP_SIZE;
  const int wRow = wid / WARPS_N;
  const int wCol = wid % WARPS_N;

  wmma::fragment<wmma::accumulator, TF32_WMMA_M, TF32_WMMA_N, TF32_WMMA_K,
                 float>
      acc;
  wmma::fill_fragment(acc, 0.0f);

  __shared__ float sA[BLOCK_M][TF32_BLOCK_K]; // 64×32 floats = 8 KB
  __shared__ float sB[TF32_BLOCK_K][BLOCK_N]; // 32×64 floats = 8 KB

  const int tid = threadIdx.x;
  const int kSteps = (N + TF32_BLOCK_K - 1) / TF32_BLOCK_K;

  for (int ks = 0; ks < kSteps; ks++) {
    const int kBase = ks * TF32_BLOCK_K;

    for (int i = tid; i < BLOCK_M * TF32_BLOCK_K; i += THREADS) {
      int lm = i / TF32_BLOCK_K, lk = i % TF32_BLOCK_K;
      int gm = bRow + lm, gk = kBase + lk;
      sA[lm][lk] = (gm < N && gk < N) ? A[gm * N + gk] : 0.0f;
    }

    for (int i = tid; i < TF32_BLOCK_K * BLOCK_N; i += THREADS) {
      int lk = i / BLOCK_N, ln = i % BLOCK_N;
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
      wmma::load_matrix_sync(bFrag, &sB[ki][wCol * TF32_WMMA_N], BLOCK_N);
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
// Benchmark helper — returns average ms over `iters` launches
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

// ── BF16 launch wrapper ──────────────────────────────────────
struct BF16Args {
  __nv_bfloat16 *A, *B;
  float *C;
  int N;
};

static void launch_bf16(void *p) {
  auto *a = (BF16Args *)p;
  dim3 blk(THREADS);
  dim3 grd((a->N + BLOCK_N - 1) / BLOCK_N, (a->N + BLOCK_M - 1) / BLOCK_M);
  wmma_bf16<<<grd, blk>>>(a->A, a->B, a->C, a->N);
}

// ── TF32 launch wrapper ──────────────────────────────────────
struct TF32Args {
  float *A, *B, *C;
  int N;
};

static void launch_tf32(void *p) {
  auto *a = (TF32Args *)p;
  dim3 blk(THREADS);
  dim3 grd((a->N + BLOCK_N - 1) / BLOCK_N, (a->N + BLOCK_M - 1) / BLOCK_M);
  wmma_tf32<<<grd, blk>>>(a->A, a->B, a->C, a->N);
}

// ─────────────────────────────────────────────────────────────
// main — sweep N, print CSV
// ─────────────────────────────────────────────────────────────
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

    // Initialise with random data
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

    // ── BF16 benchmark ─────────────────────────────────────
    {
      BF16Args arg = {dA16, dB16, dC16, N};
      float ms = bench(WARMUP, ITERS, launch_bf16, &arg);
      printf("BF16,%d,%.4f,%.2f\n", N, ms, flops / (ms * 1e-3) / 1e12);
    }

    // ── TF32 benchmark ─────────────────────────────────────
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
