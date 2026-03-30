// ============================================================
// cuBLAS matrix multiplication benchmark
// BF16 path : cublasGemmEx, CUBLAS_COMPUTE_32F + CUDA_R_16BF
// FP32 path : cublasGemmEx, CUBLAS_COMPUTE_32F_FAST_TF32
// Trick: call cuBLAS with (B, A) instead of (A, B) to handle
//        row-major storage (equivalent to C^T = B^T × A^T).
// Target GPU: NVIDIA H100 (sm_90)
// ============================================================

#include "utils.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t _s = (call);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "[cuBLAS ERR] %s:%d  code=%d\n", __FILE__, __LINE__,     \
              (int)_s);                                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

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

// ─────────────────────────────────────────────────────────────
// Launch wrappers
// ─────────────────────────────────────────────────────────────
struct BF16Args {
  cublasHandle_t handle;
  __nv_bfloat16 *A, *B;
  float *C;
  int N;
};

static void run_bf16(void *p) {
  auto *a = (BF16Args *)p;
  const float alpha = 1.f, beta = 0.f;
  // Row-major C = A×B  →  col-major C^T = B^T × A^T
  // Pass (B, A) to cuBLAS with CUBLAS_OP_N, CUBLAS_OP_N
  CUBLAS_CHECK(cublasGemmEx(a->handle, CUBLAS_OP_N, CUBLAS_OP_N, a->N, a->N,
                            a->N, &alpha, a->B, CUDA_R_16BF, a->N, a->A,
                            CUDA_R_16BF, a->N, &beta, a->C, CUDA_R_32F, a->N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

struct FP32Args {
  cublasHandle_t handle;
  float *A, *B, *C;
  int N;
};

static void run_fp32(void *p) {
  auto *a = (FP32Args *)p;
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(
      a->handle, CUBLAS_OP_N, CUBLAS_OP_N, a->N, a->N, a->N, &alpha, a->B,
      CUDA_R_32F, a->N, a->A, CUDA_R_32F, a->N, &beta, a->C, CUDA_R_32F, a->N,
      CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ─────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────
int main(void) {
  printDeviceInfo();

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const int SIZES[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
  const int NSIZES = (int)(sizeof(SIZES) / sizeof(SIZES[0]));
  const int WARMUP = 5;
  const int ITERS = 20;

  printf("dtype,N,ms,tflops\n");

  for (int si = 0; si < NSIZES; si++) {
    int N = SIZES[si];
    size_t bytes16 = (size_t)N * N * sizeof(__nv_bfloat16);
    size_t bytes32 = (size_t)N * N * sizeof(float);

    __nv_bfloat16 *dA16, *dB16;
    float *dC16;
    float *dA32, *dB32, *dC32;

    CUDA_CHECK(cudaMalloc(&dA16, bytes16));
    CUDA_CHECK(cudaMalloc(&dB16, bytes16));
    CUDA_CHECK(cudaMalloc(&dC16, bytes32));
    CUDA_CHECK(cudaMalloc(&dA32, bytes32));
    CUDA_CHECK(cudaMalloc(&dB32, bytes32));
    CUDA_CHECK(cudaMalloc(&dC32, bytes32));

    // Initialise on device via fill (simple, avoids large host allocs)
    CUDA_CHECK(cudaMemset(dA16, 0x3C, bytes16)); // ~1.0 in BF16
    CUDA_CHECK(cudaMemset(dB16, 0x3C, bytes16));
    CUDA_CHECK(cudaMemset(dC16, 0, bytes32));
    CUDA_CHECK(cudaMemset(dA32, 0x3F, bytes32)); // ~1.0 in float
    CUDA_CHECK(cudaMemset(dB32, 0x3F, bytes32));
    CUDA_CHECK(cudaMemset(dC32, 0, bytes32));

    double flops = 2.0 * (double)N * N * N;

    // ── BF16 ───────────────────────────────────────────────
    {
      BF16Args arg = {handle, dA16, dB16, dC16, N};
      float ms = bench(WARMUP, ITERS, run_bf16, &arg);
      printf("BF16,%d,%.4f,%.2f\n", N, ms, flops / (ms * 1e-3) / 1e12);
    }

    // ── FP32 (TF32 compute) ────────────────────────────────
    {
      FP32Args arg = {handle, dA32, dB32, dC32, N};
      float ms = bench(WARMUP, ITERS, run_fp32, &arg);
      printf("FP32,%d,%.4f,%.2f\n", N, ms, flops / (ms * 1e-3) / 1e12);
    }

    CUDA_CHECK(cudaFree(dA16));
    CUDA_CHECK(cudaFree(dB16));
    CUDA_CHECK(cudaFree(dC16));
    CUDA_CHECK(cudaFree(dA32));
    CUDA_CHECK(cudaFree(dB32));
    CUDA_CHECK(cudaFree(dC32));
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
