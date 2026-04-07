// ============================================================
// Matrix multiplication using Thread Block Clusters (Hopper / sm_90)
// Target GPU : NVIDIA H100 (sm_90, Compute Capability 9.0)
// ============================================================
//
// Thread Block Clusters are a new level of the CUDA thread hierarchy
// introduced in the Hopper architecture (sm_90). A Cluster groups
// multiple Thread Blocks that are co-scheduled on the same GPC
// (Graphics Processing Cluster) and can share data through
// Distributed Shared Memory (DSM) without going to global memory.
//
// Hierarchy recap on H100:
//   Thread → Warp (32 threads)
//             → Thread Block (up to 1024 threads, 1 SM)
//               → Thread Block Cluster (up to 16 blocks, 1 GPC)  ← NEW
//                 → Grid
//
// Key concepts used:
//
//   Distributed Shared Memory (DSM):
//     Each block in a cluster can read/write the shared memory of
//     any other block in the same cluster via cluster.map_shared_rank().
//     This avoids a round-trip through L2 / global memory for inter-block
//     communication.
//
//   Cluster-wide synchronisation:
//     cluster.sync() — barrier across all blocks in the cluster
//
// Compile flag: -arch=sm_90a (already set in Makefile)
//
// Approach:
//   We use a 1×2 cluster (2 blocks along the N-column direction).
//   Each block loads its own TILE_SIZE×TILE_SIZE tile of A and B into
//   shared memory.  After a cluster.sync(), each block can read the
//   neighbouring block's B-tile via DSM, so each block computes TWO
//   output sub-tiles of C — doubling the effective B-tile width
//   without doubling per-block shared memory.
//
//   For the FP32 path we operate on float tiles directly.
//   For the BF16 path we load __nv_bfloat16 tiles and accumulate in float.
//
// Reference:
//   CUDA C++ Programming Guide §K — Thread Block Clusters
//   https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters
// ============================================================

#include "utils.cuh"
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────
// cuBLAS error checking macro
// ─────────────────────────────────────────────────────────────
#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "[cuBLAS ERR] %s:%d  code=%d\n", __FILE__, __LINE__,     \
              (int)status);                                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────
#define TILE_SIZE 16    // Per-block tile dimension
#define CLUSTER_DIM_X 1 // Cluster width  (along M / row direction)
#define CLUSTER_DIM_Y 2 // Cluster height (along N / col direction)

// ─────────────────────────────────────────────────────────────
// FP32 Cluster Kernel
// ─────────────────────────────────────────────────────────────
// Each block in the cluster computes TILE_SIZE rows of C.
// Via DSM each block accesses its own B-tile AND the neighbor's
// B-tile, producing 2 sub-tiles of C (2 × TILE_SIZE columns).
//
// Grid is launched with half the columns (the cluster doubles output
// width):
//   gridDim.x = matrixSize / TILE_SIZE                          (rows)
//   gridDim.y = matrixSize / (TILE_SIZE * CLUSTER_DIM_Y)        (column-pairs)
// ─────────────────────────────────────────────────────────────
__global__ void __cluster_dims__(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1)
    clusterMatMulFP32(const float *__restrict__ inputMatrixA,
                      const float *__restrict__ inputMatrixB,
                      float *__restrict__ outputMatrixC, int matrixSize) {
  cg::cluster_group clusterGroup = cg::this_cluster();
  unsigned int clusterBlockRank = clusterGroup.block_rank();

  // Block's row-tile and column-tile indices in the output matrix
  int tileRowIndex = blockIdx.x; // which TILE_SIZE-row stripe

  // The grid's y-dimension covers column-pairs.
  // clusterBlockRank (0 or 1) selects the sub-tile within the pair.
  int tileColumnIndex = blockIdx.y * CLUSTER_DIM_Y + clusterBlockRank;

  int threadRow = threadIdx.y;
  int threadColumn = threadIdx.x;

  // Shared memory for this block's tiles
  __shared__ float sharedTileA[TILE_SIZE][TILE_SIZE];
  __shared__ float sharedTileB[TILE_SIZE][TILE_SIZE];

  // Accumulator for THIS block's own output element
  float accumulatorLocal = 0.0f;

  // Also accumulate for the NEIGHBOR's B-tile (DSM access)
  float accumulatorNeighbor = 0.0f;

  int numberOfTiles = matrixSize / TILE_SIZE;

  for (int tileIndex = 0; tileIndex < numberOfTiles; ++tileIndex) {
    // Load A tile: row is from tileRowIndex, col iterates over k-tiles
    int globalRowA = tileRowIndex * TILE_SIZE + threadRow;
    int globalColumnA = tileIndex * TILE_SIZE + threadColumn;
    sharedTileA[threadRow][threadColumn] =
        inputMatrixA[globalRowA * matrixSize + globalColumnA];

    // Load B tile: row iterates over k-tiles, col is from this block
    int globalRowB = tileIndex * TILE_SIZE + threadRow;
    int globalColumnB = tileColumnIndex * TILE_SIZE + threadColumn;
    sharedTileB[threadRow][threadColumn] =
        inputMatrixB[globalRowB * matrixSize + globalColumnB];

    // Ensure all loads in the cluster are visible
    __syncthreads();
    clusterGroup.sync();

    // ── Compute with THIS block's B-tile ─────────────────
#pragma unroll
    for (int kElement = 0; kElement < TILE_SIZE; ++kElement)
      accumulatorLocal += sharedTileA[threadRow][kElement] *
                          sharedTileB[kElement][threadColumn];

    // ── Compute with NEIGHBOR's B-tile via DSM ───────────
    unsigned int neighborBlockRank = clusterBlockRank ^ 1; // flip 0↔1
    float *neighborSharedTileB =
        clusterGroup.map_shared_rank(sharedTileB, neighborBlockRank);
#pragma unroll
    for (int kElement = 0; kElement < TILE_SIZE; ++kElement)
      accumulatorNeighbor += sharedTileA[threadRow][kElement] *
                             neighborSharedTileB[kElement][threadColumn];

    __syncthreads();
    clusterGroup.sync();
  }

  // Write THIS block's result
  int outputRow = tileRowIndex * TILE_SIZE + threadRow;
  int outputColumn = tileColumnIndex * TILE_SIZE + threadColumn;
  if (outputRow < matrixSize && outputColumn < matrixSize)
    outputMatrixC[outputRow * matrixSize + outputColumn] = accumulatorLocal;

  // Write NEIGHBOR's result (the column offset uses neighbor's tileColumnIndex)
  unsigned int neighborBlockRank = clusterBlockRank ^ 1;
  int neighborTileColumnIndex = blockIdx.y * CLUSTER_DIM_Y + neighborBlockRank;
  int neighborOutputColumn = neighborTileColumnIndex * TILE_SIZE + threadColumn;
  if (outputRow < matrixSize && neighborOutputColumn < matrixSize)
    outputMatrixC[outputRow * matrixSize + neighborOutputColumn] =
        accumulatorNeighbor;
}

// ─────────────────────────────────────────────────────────────
// BF16 Cluster Kernel  (loads BF16, accumulates in FP32)
// ─────────────────────────────────────────────────────────────
__global__ void __cluster_dims__(CLUSTER_DIM_X, CLUSTER_DIM_Y, 1)
    clusterMatMulBF16(const __nv_bfloat16 *__restrict__ inputMatrixA,
                      const __nv_bfloat16 *__restrict__ inputMatrixB,
                      float *__restrict__ outputMatrixC, int matrixSize) {
  cg::cluster_group clusterGroup = cg::this_cluster();
  unsigned int clusterBlockRank = clusterGroup.block_rank();

  int tileRowIndex = blockIdx.x;
  int tileColumnIndex = blockIdx.y * CLUSTER_DIM_Y + clusterBlockRank;

  int threadRow = threadIdx.y;
  int threadColumn = threadIdx.x;

  __shared__ __nv_bfloat16 sharedTileA[TILE_SIZE][TILE_SIZE];
  __shared__ __nv_bfloat16 sharedTileB[TILE_SIZE][TILE_SIZE];

  float accumulatorLocal = 0.0f;
  float accumulatorNeighbor = 0.0f;

  int numberOfTiles = matrixSize / TILE_SIZE;

  for (int tileIndex = 0; tileIndex < numberOfTiles; ++tileIndex) {
    int globalRowA = tileRowIndex * TILE_SIZE + threadRow;
    int globalColumnA = tileIndex * TILE_SIZE + threadColumn;
    sharedTileA[threadRow][threadColumn] =
        inputMatrixA[globalRowA * matrixSize + globalColumnA];

    int globalRowB = tileIndex * TILE_SIZE + threadRow;
    int globalColumnB = tileColumnIndex * TILE_SIZE + threadColumn;
    sharedTileB[threadRow][threadColumn] =
        inputMatrixB[globalRowB * matrixSize + globalColumnB];

    __syncthreads();
    clusterGroup.sync();

#pragma unroll
    for (int kElement = 0; kElement < TILE_SIZE; ++kElement)
      accumulatorLocal += __bfloat162float(sharedTileA[threadRow][kElement]) *
                          __bfloat162float(sharedTileB[kElement][threadColumn]);

    unsigned int neighborBlockRank = clusterBlockRank ^ 1;
    __nv_bfloat16 *neighborSharedTileB =
        clusterGroup.map_shared_rank(sharedTileB, neighborBlockRank);
#pragma unroll
    for (int kElement = 0; kElement < TILE_SIZE; ++kElement)
      accumulatorNeighbor +=
          __bfloat162float(sharedTileA[threadRow][kElement]) *
          __bfloat162float(neighborSharedTileB[kElement][threadColumn]);

    __syncthreads();
    clusterGroup.sync();
  }

  int outputRow = tileRowIndex * TILE_SIZE + threadRow;
  int outputColumn = tileColumnIndex * TILE_SIZE + threadColumn;
  if (outputRow < matrixSize && outputColumn < matrixSize)
    outputMatrixC[outputRow * matrixSize + outputColumn] = accumulatorLocal;

  unsigned int neighborBlockRank = clusterBlockRank ^ 1;
  int neighborTileColumnIndex = blockIdx.y * CLUSTER_DIM_Y + neighborBlockRank;
  int neighborOutputColumn = neighborTileColumnIndex * TILE_SIZE + threadColumn;
  if (outputRow < matrixSize && neighborOutputColumn < matrixSize)
    outputMatrixC[outputRow * matrixSize + neighborOutputColumn] =
        accumulatorNeighbor;
}

// ─────────────────────────────────────────────────────────────
// Benchmark helper — returns average milliseconds over
// `benchmarkIterations` launches
// ─────────────────────────────────────────────────────────────
static float runBenchmark(int warmupIterations, int benchmarkIterations,
                          void (*kernelLauncher)(void *),
                          void *kernelArguments) {
  for (int iteration = 0; iteration < warmupIterations; iteration++)
    kernelLauncher(kernelArguments);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t eventStart, eventStop;
  CUDA_CHECK(cudaEventCreate(&eventStart));
  CUDA_CHECK(cudaEventCreate(&eventStop));
  CUDA_CHECK(cudaEventRecord(eventStart));
  for (int iteration = 0; iteration < benchmarkIterations; iteration++)
    kernelLauncher(kernelArguments);
  CUDA_CHECK(cudaEventRecord(eventStop));
  CUDA_CHECK(cudaEventSynchronize(eventStop));
  float elapsedMilliseconds = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, eventStart, eventStop));
  cudaEventDestroy(eventStart);
  cudaEventDestroy(eventStop);
  return elapsedMilliseconds / benchmarkIterations;
}

// ─────────────────────────────────────────────────────────────
// Cluster kernel launch wrappers (using cudaLaunchKernelEx)
// ─────────────────────────────────────────────────────────────

struct ClusterFP32Arguments {
  float *inputA, *inputB, *outputC;
  int matrixSize;
};

static void launchClusterFP32Kernel(void *parameters) {
  auto *arguments = (ClusterFP32Arguments *)parameters;
  int matrixSize = arguments->matrixSize;

  dim3 blockDimensions(TILE_SIZE, TILE_SIZE);
  // Grid: x = rows / TILE_SIZE,  y = cols / (TILE_SIZE * CLUSTER_DIM_Y)
  dim3 gridDimensions(matrixSize / TILE_SIZE,
                      matrixSize / (TILE_SIZE * CLUSTER_DIM_Y));

  // Configure cluster launch
  cudaLaunchConfig_t launchConfig = {};
  launchConfig.gridDim = gridDimensions;
  launchConfig.blockDim = blockDimensions;
  launchConfig.dynamicSmemBytes = 0;

  cudaLaunchAttribute clusterAttribute;
  clusterAttribute.id = cudaLaunchAttributeClusterDimension;
  clusterAttribute.val.clusterDim = {CLUSTER_DIM_X, CLUSTER_DIM_Y, 1};
  launchConfig.attrs = &clusterAttribute;
  launchConfig.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&launchConfig, clusterMatMulFP32,
                                arguments->inputA, arguments->inputB,
                                arguments->outputC, matrixSize));
}

struct ClusterBF16Arguments {
  __nv_bfloat16 *inputA, *inputB;
  float *outputC;
  int matrixSize;
};

static void launchClusterBF16Kernel(void *parameters) {
  auto *arguments = (ClusterBF16Arguments *)parameters;
  int matrixSize = arguments->matrixSize;

  dim3 blockDimensions(TILE_SIZE, TILE_SIZE);
  dim3 gridDimensions(matrixSize / TILE_SIZE,
                      matrixSize / (TILE_SIZE * CLUSTER_DIM_Y));

  cudaLaunchConfig_t launchConfig = {};
  launchConfig.gridDim = gridDimensions;
  launchConfig.blockDim = blockDimensions;
  launchConfig.dynamicSmemBytes = 0;

  cudaLaunchAttribute clusterAttribute;
  clusterAttribute.id = cudaLaunchAttributeClusterDimension;
  clusterAttribute.val.clusterDim = {CLUSTER_DIM_X, CLUSTER_DIM_Y, 1};
  launchConfig.attrs = &clusterAttribute;
  launchConfig.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&launchConfig, clusterMatMulBF16,
                                arguments->inputA, arguments->inputB,
                                arguments->outputC, matrixSize));
}

// ─────────────────────────────────────────────────────────────
// cuBLAS reference launch wrappers
// ─────────────────────────────────────────────────────────────
struct CuBlasBF16Arguments {
  cublasHandle_t cublasHandle;
  __nv_bfloat16 *inputA, *inputB;
  float *outputC;
  int matrixSize;
};

static void launchCuBlasBF16Kernel(void *parameters) {
  auto *arguments = (CuBlasBF16Arguments *)parameters;
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(
      arguments->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, arguments->matrixSize,
      arguments->matrixSize, arguments->matrixSize, &alpha, arguments->inputB,
      CUDA_R_16BF, arguments->matrixSize, arguments->inputA, CUDA_R_16BF,
      arguments->matrixSize, &beta, arguments->outputC, CUDA_R_32F,
      arguments->matrixSize, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

struct CuBlasFP32Arguments {
  cublasHandle_t cublasHandle;
  float *inputA, *inputB, *outputC;
  int matrixSize;
};

static void launchCuBlasFP32Kernel(void *parameters) {
  auto *arguments = (CuBlasFP32Arguments *)parameters;
  const float alpha = 1.f, beta = 0.f;
  CUBLAS_CHECK(cublasGemmEx(
      arguments->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, arguments->matrixSize,
      arguments->matrixSize, arguments->matrixSize, &alpha, arguments->inputB,
      CUDA_R_32F, arguments->matrixSize, arguments->inputA, CUDA_R_32F,
      arguments->matrixSize, &beta, arguments->outputC, CUDA_R_32F,
      arguments->matrixSize, CUBLAS_COMPUTE_32F_FAST_TF32,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ─────────────────────────────────────────────────────────────
// Correctness verification (CPU reference, small sizes only)
// ─────────────────────────────────────────────────────────────
static void cpuMatrixMultiplyFP32(const float *inputMatrixA,
                                  const float *inputMatrixB,
                                  float *outputMatrixC, int matrixSize) {
  for (int rowIndex = 0; rowIndex < matrixSize; ++rowIndex)
    for (int columnIndex = 0; columnIndex < matrixSize; ++columnIndex) {
      float accumulator = 0.f;
      for (int kIndex = 0; kIndex < matrixSize; ++kIndex)
        accumulator += inputMatrixA[rowIndex * matrixSize + kIndex] *
                       inputMatrixB[kIndex * matrixSize + columnIndex];
      outputMatrixC[rowIndex * matrixSize + columnIndex] = accumulator;
    }
}

static bool verifyAgainstCpuReference(const float *gpuOutputMatrix,
                                      const float *cpuReferenceMatrix,
                                      int matrixSize, float tolerance,
                                      const char *verificationLabel) {
  float maximumError = 0.f;
  for (int elementIndex = 0; elementIndex < matrixSize * matrixSize;
       ++elementIndex) {
    float absoluteError =
        fabsf(gpuOutputMatrix[elementIndex] - cpuReferenceMatrix[elementIndex]);
    if (absoluteError > maximumError)
      maximumError = absoluteError;
  }
  bool passed = (maximumError < tolerance);
  printf("  [%s] max error = %e  %s\n", verificationLabel, maximumError,
         passed ? "PASSED ✅" : "FAILED ❌");
  return passed;
}

// ─────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────
int main(void) {
  printDeviceInfo();

  cublasHandle_t cublasHandle;
  CUBLAS_CHECK(cublasCreate(&cublasHandle));

  const int testMatrixSizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
  const int numberOfTestSizes =
      (int)(sizeof(testMatrixSizes) / sizeof(testMatrixSizes[0]));
  const int warmupIterations = 5;
  const int benchmarkIterations = 20;
  const int verificationSizeLimit = 512;

  printf("============================================================\n");
  printf("Thread Block Cluster MatMul  vs  cuBLAS\n");
  printf("Cluster dims: %d × %d,  Tile: %d × %d\n", CLUSTER_DIM_X,
         CLUSTER_DIM_Y, TILE_SIZE, TILE_SIZE);
  printf("============================================================\n\n");

  // CSV header
  printf("impl,dtype,N,ms,tflops\n");

  for (int sizeIndex = 0; sizeIndex < numberOfTestSizes; sizeIndex++) {
    int matrixSize = testMatrixSizes[sizeIndex];
    size_t memorySizeBFloat16 =
        (size_t)matrixSize * matrixSize * sizeof(__nv_bfloat16);
    size_t memorySizeFloat32 = (size_t)matrixSize * matrixSize * sizeof(float);
    double floatingPointOperations =
        2.0 * (double)matrixSize * matrixSize * matrixSize;

    // ── Allocate device memory ────────────────────────────
    __nv_bfloat16 *deviceInputA16, *deviceInputB16;
    float *deviceOutputC16Cluster, *deviceOutputC16Cublas;
    float *deviceInputA32, *deviceInputB32;
    float *deviceOutputC32Cluster, *deviceOutputC32Cublas;

    CUDA_CHECK(cudaMalloc(&deviceInputA16, memorySizeBFloat16));
    CUDA_CHECK(cudaMalloc(&deviceInputB16, memorySizeBFloat16));
    CUDA_CHECK(cudaMalloc(&deviceOutputC16Cluster, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceOutputC16Cublas, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceInputA32, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceInputB32, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceOutputC32Cluster, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceOutputC32Cublas, memorySizeFloat32));

    // ── Initialise with random data ──────────────────────
    {
      __nv_bfloat16 *hostInputA16 = (__nv_bfloat16 *)malloc(memorySizeBFloat16);
      __nv_bfloat16 *hostInputB16 = (__nv_bfloat16 *)malloc(memorySizeBFloat16);
      float *hostInputA32 = (float *)malloc(memorySizeFloat32);
      float *hostInputB32 = (float *)malloc(memorySizeFloat32);
      srand(42);
      for (int elementIndex = 0; elementIndex < matrixSize * matrixSize;
           elementIndex++) {
        float randomValue = (float)(rand() % 10) / 10.0f;
        hostInputA16[elementIndex] = __float2bfloat16(randomValue);
        hostInputB16[elementIndex] = __float2bfloat16(randomValue);
        hostInputA32[elementIndex] = randomValue;
        hostInputB32[elementIndex] = randomValue;
      }
      CUDA_CHECK(cudaMemcpy(deviceInputA16, hostInputA16, memorySizeBFloat16,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(deviceInputB16, hostInputB16, memorySizeBFloat16,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(deviceInputA32, hostInputA32, memorySizeFloat32,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(deviceInputB32, hostInputB32, memorySizeFloat32,
                            cudaMemcpyHostToDevice));
      free(hostInputA16);
      free(hostInputB16);
      free(hostInputA32);
      free(hostInputB32);
    }

    // ── BF16 — Cluster kernel ─────────────────────────────
    {
      CUDA_CHECK(cudaMemset(deviceOutputC16Cluster, 0, memorySizeFloat32));
      ClusterBF16Arguments kernelArguments = {
          deviceInputA16, deviceInputB16, deviceOutputC16Cluster, matrixSize};
      float elapsedMilliseconds =
          runBenchmark(warmupIterations, benchmarkIterations,
                       launchClusterBF16Kernel, &kernelArguments);
      printf("cluster,BF16,%d,%.4f,%.2f\n", matrixSize, elapsedMilliseconds,
             floatingPointOperations / (elapsedMilliseconds * 1e-3) / 1e12);
    }

    // ── BF16 — cuBLAS reference ───────────────────────────
    {
      CUDA_CHECK(cudaMemset(deviceOutputC16Cublas, 0, memorySizeFloat32));
      CuBlasBF16Arguments kernelArguments = {cublasHandle, deviceInputA16,
                                             deviceInputB16,
                                             deviceOutputC16Cublas, matrixSize};
      float elapsedMilliseconds =
          runBenchmark(warmupIterations, benchmarkIterations,
                       launchCuBlasBF16Kernel, &kernelArguments);
      printf("cublas,BF16,%d,%.4f,%.2f\n", matrixSize, elapsedMilliseconds,
             floatingPointOperations / (elapsedMilliseconds * 1e-3) / 1e12);
    }

    // ── FP32 — Cluster kernel ─────────────────────────────
    {
      CUDA_CHECK(cudaMemset(deviceOutputC32Cluster, 0, memorySizeFloat32));
      ClusterFP32Arguments kernelArguments = {
          deviceInputA32, deviceInputB32, deviceOutputC32Cluster, matrixSize};
      float elapsedMilliseconds =
          runBenchmark(warmupIterations, benchmarkIterations,
                       launchClusterFP32Kernel, &kernelArguments);
      printf("cluster,FP32,%d,%.4f,%.2f\n", matrixSize, elapsedMilliseconds,
             floatingPointOperations / (elapsedMilliseconds * 1e-3) / 1e12);
    }

    // ── FP32 — cuBLAS reference ───────────────────────────
    {
      CUDA_CHECK(cudaMemset(deviceOutputC32Cublas, 0, memorySizeFloat32));
      CuBlasFP32Arguments kernelArguments = {cublasHandle, deviceInputA32,
                                             deviceInputB32,
                                             deviceOutputC32Cublas, matrixSize};
      float elapsedMilliseconds =
          runBenchmark(warmupIterations, benchmarkIterations,
                       launchCuBlasFP32Kernel, &kernelArguments);
      printf("cublas,FP32,%d,%.4f,%.2f\n", matrixSize, elapsedMilliseconds,
             floatingPointOperations / (elapsedMilliseconds * 1e-3) / 1e12);
    }

    // ── Correctness verification (small sizes) ────────────
    if (matrixSize <= verificationSizeLimit) {
      float *hostInputA = (float *)malloc(memorySizeFloat32);
      float *hostInputB = (float *)malloc(memorySizeFloat32);
      float *hostOutputCluster = (float *)malloc(memorySizeFloat32);
      float *hostOutputReference = (float *)malloc(memorySizeFloat32);

      CUDA_CHECK(cudaMemcpy(hostInputA, deviceInputA32, memorySizeFloat32,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hostInputB, deviceInputB32, memorySizeFloat32,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hostOutputCluster, deviceOutputC32Cluster,
                            memorySizeFloat32, cudaMemcpyDeviceToHost));

      cpuMatrixMultiplyFP32(hostInputA, hostInputB, hostOutputReference,
                            matrixSize);
      verifyAgainstCpuReference(hostOutputCluster, hostOutputReference,
                                matrixSize, 1e-2f, "FP32 cluster vs CPU");

      free(hostInputA);
      free(hostInputB);
      free(hostOutputCluster);
      free(hostOutputReference);
    }

    // ── Cleanup ───────────────────────────────────────────
    CUDA_CHECK(cudaFree(deviceInputA16));
    CUDA_CHECK(cudaFree(deviceInputB16));
    CUDA_CHECK(cudaFree(deviceOutputC16Cluster));
    CUDA_CHECK(cudaFree(deviceOutputC16Cublas));
    CUDA_CHECK(cudaFree(deviceInputA32));
    CUDA_CHECK(cudaFree(deviceInputB32));
    CUDA_CHECK(cudaFree(deviceOutputC32Cluster));
    CUDA_CHECK(cudaFree(deviceOutputC32Cublas));
  }

  CUBLAS_CHECK(cublasDestroy(cublasHandle));
  printf("\n✅  All benchmarks complete.\n");
  return 0;
}
