// ============================================================
// Tiled matrix multiplication — CUDA Programming Guide example
// Target GPU : NVIDIA H100 (sm_90, Compute Capability 9.0)
// ============================================================

#include "utils.cuh" // CUDA_CHECK, GpuTimer, printDeviceInfo
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

// Thread-block tile size (16×16 = 256 threads, well within H100's 1024 limit)
#define BLOCK_SIZE 16

// Default matrix dimension — must be a multiple of BLOCK_SIZE.
// 2048×2048 gives a meaningful workload on an H100 without being too slow.
#define N 2048

// ─────────────────────────────────────────────────────────────
// Matrix type
// Matrices are stored in row-major order:
//   M(row, col) = *(M.elements + row * M.stride + col)
// ─────────────────────────────────────────────────────────────
typedef struct {
  int width;
  int height;
  int stride;
  float *elements;
} Matrix;

// ─────────────────────────────────────────────────────────────
// Device helper functions
// ─────────────────────────────────────────────────────────────

// Get a matrix element
__device__ float GetElement(const Matrix M, int row, int col) {
  return M.elements[row * M.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix M, int row, int col, float value) {
  M.elements[row * M.stride + col] = value;
}

// Get the BLOCK_SIZE×BLOCK_SIZE sub-matrix Asub of A that is located
// col sub-matrices to the right and row sub-matrices down from the
// upper-left corner of A.
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

// ─────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────

// Matrix multiplication kernel called by MatMul().
// Each thread block computes one BLOCK_SIZE×BLOCK_SIZE tile of C.
// Shared memory is used to reduce global-memory bandwidth.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  // Tile of A and B cached in shared memory.
  // Declared OUTSIDE the loop so the compiler can place them once.
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  // Block and thread indices
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

  // Accumulator for this thread's output element
  float Cvalue = 0.0f;

  // Loop over all BLOCK_SIZE×BLOCK_SIZE tiles of A and B needed for Csub
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // Sub-matrices of A and B for this iteration
    Matrix Asub = GetSubMatrix(A, blockRow, m);
    Matrix Bsub = GetSubMatrix(B, m, blockCol);

    // Cooperatively load the tile into shared memory —
    // each thread loads exactly one element of As and one of Bs.
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);

    // Wait until every thread has finished loading before computing
    __syncthreads();

    // Multiply this pair of tiles and accumulate
    for (int e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += As[row][e] * Bs[e][col];

    // Wait until every thread has finished computing before loading
    // the next tile (prevents overwriting As/Bs while still in use)
    __syncthreads();
  }

  // Write the final accumulated value to the output matrix
  SetElement(Csub, row, col, Cvalue);
}

// ─────────────────────────────────────────────────────────────
// Host driver — allocates device memory, runs kernel, copies back
// ─────────────────────────────────────────────────────────────

// Matrix dimensions must be multiples of BLOCK_SIZE.
void MatMul(const Matrix A, const Matrix B, Matrix C) {

  // ── Allocate and copy A to device ──────────────────────────
  Matrix d_A;
  d_A.width = d_A.stride = A.width;
  d_A.height = A.height;
  size_t sizeA = (size_t)A.width * A.height * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_A.elements, sizeA));
  CUDA_CHECK(
      cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice));

  // ── Allocate and copy B to device ──────────────────────────
  Matrix d_B;
  d_B.width = d_B.stride = B.width;
  d_B.height = B.height;
  size_t sizeB = (size_t)B.width * B.height * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_B.elements, sizeB));
  CUDA_CHECK(
      cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice));

  // ── Allocate C on device (no initialisation needed) ────────
  Matrix d_C;
  d_C.width = d_C.stride = C.width;
  d_C.height = C.height;
  size_t sizeC = (size_t)C.width * C.height * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_C.elements, sizeC));

  // ── Launch kernel ───────────────────────────────────────────
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / BLOCK_SIZE, A.height / BLOCK_SIZE);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

  // Check for any asynchronous kernel errors
  CUDA_CHECK(cudaGetLastError());

  // ── Copy result back to host ────────────────────────────────
  // cudaMemcpy implicitly synchronises; any kernel fault surfaces here.
  CUDA_CHECK(
      cudaMemcpy(C.elements, d_C.elements, sizeC, cudaMemcpyDeviceToHost));

  // ── Release device memory ───────────────────────────────────
  CUDA_CHECK(cudaFree(d_A.elements));
  CUDA_CHECK(cudaFree(d_B.elements));
  CUDA_CHECK(cudaFree(d_C.elements));
}

// ─────────────────────────────────────────────────────────────
// CPU reference  (used only for correctness verification)
// ─────────────────────────────────────────────────────────────
static void cpuMatMul(const Matrix A, const Matrix B, Matrix C) {
  for (int row = 0; row < A.height; ++row)
    for (int col = 0; col < B.width; ++col) {
      float sum = 0.0f;
      for (int e = 0; e < A.width; ++e)
        sum += A.elements[row * A.stride + e] * B.elements[e * B.stride + col];
      C.elements[row * C.stride + col] = sum;
    }
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
static void initMatrix(Matrix *M, int height, int width) {
  M->height = height;
  M->width = width;
  M->stride = width;
  M->elements = (float *)malloc((size_t)height * width * sizeof(float));
  if (!M->elements) {
    fprintf(stderr, "malloc failed\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < height * width; ++i)
    M->elements[i] = (float)(rand() % 10);
}

static void freeMatrix(Matrix *M) {
  free(M->elements);
  M->elements = NULL;
}

// ─────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────
int main(void) {
  srand(42);

  // Print device information (name, compute capability, memory, SMs)
  printDeviceInfo();

  printf("Matrix size : %d x %d\n", N, N);
  printf("Block size  : %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
  printf("Grid size   : %d x %d\n\n", N / BLOCK_SIZE, N / BLOCK_SIZE);

  // Allocate host matrices
  Matrix A, B, C, Cref;
  initMatrix(&A, N, N);
  initMatrix(&B, N, N);
  initMatrix(&C, N, N);
  initMatrix(&Cref, N, N);

  // ── Timed GPU multiplication ────────────────────────────────
  GpuTimer timer;
  timerStart(&timer);
  MatMul(A, B, C);
  float gpuMs = timerStop(&timer);

  // FLOPs: each output element requires N multiply-adds = 2N FLOPs.
  double flops = 2.0 * (double)N * N * N;
  double gflops = flops / (gpuMs * 1e-3) / 1e9;
  printf("GPU time    : %.3f ms\n", gpuMs);
  printf("Performance : %.2f GFlop/s\n\n", gflops);

  // ── CPU reference (only for small N; skip for N > 512) ─────
#define VERIFY_LIMIT 512
  if (N <= VERIFY_LIMIT) {
    cpuMatMul(A, B, Cref);

    float maxErr = 0.0f;
    for (int i = 0; i < N * N; ++i) {
      float err = fabsf(C.elements[i] - Cref.elements[i]);
      if (err > maxErr)
        maxErr = err;
    }
    printf("Max error vs CPU reference : %e\n", maxErr);
    printf("Correctness check          : %s\n\n",
           maxErr < 1e-2f ? "PASSED ✅" : "FAILED ❌");
  } else {
    printf("(Skipping CPU reference for N=%d — would take too long)\n\n", N);
  }

  freeMatrix(&A);
  freeMatrix(&B);
  freeMatrix(&C);
  freeMatrix(&Cref);
  return 0;
}