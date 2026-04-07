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
//   Both blocks in a cluster share the same row of tiles in A.
//   Only block rank 0 loads the A-tile from global memory; rank 1
//   reads it via Distributed Shared Memory — cutting A-matrix
//   global memory traffic in half.
//
//   Each thread computes a THREAD_TILE_M × THREAD_TILE_N sub-tile of
//   the output by performing THREAD_TILE_M × THREAD_TILE_N FMA ops
//   per K-element — dramatically increasing arithmetic intensity.
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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

// Block-level tile loaded into shared memory
#define BLOCK_TILE_M 128 // rows of C per block
#define BLOCK_TILE_N 128 // cols of C per block
#define BLOCK_TILE_K 16  // K-dimension tile width

// Thread-level tile — each thread computes TM × TN output elements
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8

// Thread block dimensions: (BLOCK_TILE_N / TN) × (BLOCK_TILE_M / TM)
// = (128/8) × (128/8) = 16 × 16 = 256 threads
#define BLOCK_DIM_X (BLOCK_TILE_N / THREAD_TILE_N) // 16
#define BLOCK_DIM_Y (BLOCK_TILE_M / THREAD_TILE_M) // 16

// Cluster configuration
#define CLUSTER_DIM_X 1
#define CLUSTER_DIM_Y 2

// ─────────────────────────────────────────────────────────────
// FP32 Cluster Kernel — high-performance thread-tiled version
// ─────────────────────────────────────────────────────────────
//
// Each thread computes an 8×8 sub-tile of C.
// Block tile: 128×128 output, loaded via 128×16 A-tile and 16×128 B-tile.
// Cluster: 1×2 — rank 0 loads A from global, rank 1 reads A via DSM.
// ─────────────────────────────────────────────────────────────
__global__ void
    clusterMatMulFP32(const float *__restrict__ inputMatrixA,
                      const float *__restrict__ inputMatrixB,
                      float *__restrict__ outputMatrixC, int matrixSize) {
  cg::cluster_group clusterGroup = cg::this_cluster();
  unsigned int clusterBlockRank = clusterGroup.block_rank();

  // Which 128×128 output tile does this block compute?
  int blockTileRow = blockIdx.x; // row tile index
  int blockTileCol = blockIdx.y; // col tile index

  // Thread's position within the block
  int threadId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;

  // Shared memory tiles
  __shared__ float sharedA[BLOCK_TILE_M][BLOCK_TILE_K]; // 128 × 16
  __shared__ float sharedB[BLOCK_TILE_K][BLOCK_TILE_N]; // 16 × 128

  // Thread-local accumulators: TM × TN = 8 × 8 = 64 registers
  float accumulator[THREAD_TILE_M][THREAD_TILE_N] = {};

  // Which sub-tile of the output does this thread compute?
  int threadTileRow = threadIdx.y * THREAD_TILE_M; // row offset within block tile
  int threadTileCol = threadIdx.x * THREAD_TILE_N; // col offset within block tile

  // Number of K-tiles
  int numberOfKTiles = matrixSize / BLOCK_TILE_K;

  // Precompute loading assignments — each of 256 threads loads
  // multiple elements to fill the shared memory tiles.
  // sharedA: 128×16 = 2048 floats, 256 threads → 8 floats/thread
  // sharedB: 16×128 = 2048 floats, 256 threads → 8 floats/thread

  for (int kTile = 0; kTile < numberOfKTiles; ++kTile) {
    // ── Load A tile into shared memory ─────────────────────
    // Only rank 0 loads A; rank 1 will read via DSM.
    if (clusterBlockRank == 0) {
      // Each thread loads 8 elements of A
      int globalRowBase = blockTileRow * BLOCK_TILE_M;
      int globalColBase = kTile * BLOCK_TILE_K;
#pragma unroll
      for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_K) / (BLOCK_DIM_X * BLOCK_DIM_Y); ++i) {
        int linearIdx = threadId + i * (BLOCK_DIM_X * BLOCK_DIM_Y);
        int localRow = linearIdx / BLOCK_TILE_K;
        int localCol = linearIdx % BLOCK_TILE_K;
        sharedA[localRow][localCol] =
            inputMatrixA[(globalRowBase + localRow) * matrixSize +
                         globalColBase + localCol];
      }
    }

    // ── Load B tile into shared memory ─────────────────────
    // Each block loads its own B tile (different column ranges).
    {
      int globalRowBase = kTile * BLOCK_TILE_K;
      int globalColBase = blockTileCol * BLOCK_TILE_N;
#pragma unroll
      for (int i = 0; i < (BLOCK_TILE_K * BLOCK_TILE_N) / (BLOCK_DIM_X * BLOCK_DIM_Y); ++i) {
        int linearIdx = threadId + i * (BLOCK_DIM_X * BLOCK_DIM_Y);
        int localRow = linearIdx / BLOCK_TILE_N;
        int localCol = linearIdx % BLOCK_TILE_N;
        sharedB[localRow][localCol] =
            inputMatrixB[(globalRowBase + localRow) * matrixSize +
                         globalColBase + localCol];
      }
    }

    __syncthreads();
    clusterGroup.sync();

    // ── Get A tile pointer (DSM for rank 1) ────────────────
    const float (*tileA)[BLOCK_TILE_K];
    if (clusterBlockRank == 0) {
      tileA = sharedA;
    } else {
      tileA = clusterGroup.map_shared_rank(sharedA, 0);
    }

    // ── Compute: each thread does TM × TN FMAs per K element ──
#pragma unroll
    for (int k = 0; k < BLOCK_TILE_K; ++k) {
      // Load A column and B row into registers
      float regA[THREAD_TILE_M];
      float regB[THREAD_TILE_N];
#pragma unroll
      for (int m = 0; m < THREAD_TILE_M; ++m)
        regA[m] = tileA[threadTileRow + m][k];
#pragma unroll
      for (int n = 0; n < THREAD_TILE_N; ++n)
        regB[n] = sharedB[k][threadTileCol + n];
      // Outer product accumulation
#pragma unroll
      for (int m = 0; m < THREAD_TILE_M; ++m)
#pragma unroll
        for (int n = 0; n < THREAD_TILE_N; ++n)
          accumulator[m][n] += regA[m] * regB[n];
    }

    __syncthreads();
    clusterGroup.sync();
  }

  // ── Write results to global memory ─────────────────────────
  int globalRowBase = blockTileRow * BLOCK_TILE_M + threadTileRow;
  int globalColBase = blockTileCol * BLOCK_TILE_N + threadTileCol;
#pragma unroll
  for (int m = 0; m < THREAD_TILE_M; ++m) {
    int row = globalRowBase + m;
    if (row < matrixSize) {
#pragma unroll
      for (int n = 0; n < THREAD_TILE_N; ++n) {
        int col = globalColBase + n;
        if (col < matrixSize)
          outputMatrixC[row * matrixSize + col] = accumulator[m][n];
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────
// BF16 Cluster Kernel  (loads BF16, converts to FP32 into shmem,
//                        accumulates in FP32)
// ─────────────────────────────────────────────────────────────
__global__ void
    clusterMatMulBF16(const __nv_bfloat16 *__restrict__ inputMatrixA,
                      const __nv_bfloat16 *__restrict__ inputMatrixB,
                      float *__restrict__ outputMatrixC, int matrixSize) {
  cg::cluster_group clusterGroup = cg::this_cluster();
  unsigned int clusterBlockRank = clusterGroup.block_rank();

  int blockTileRow = blockIdx.x;
  int blockTileCol = blockIdx.y;
  int threadId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;

  // Store as float in shared memory — convert during load, not during compute
  __shared__ float sharedA[BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ float sharedB[BLOCK_TILE_K][BLOCK_TILE_N];

  float accumulator[THREAD_TILE_M][THREAD_TILE_N] = {};

  int threadTileRow = threadIdx.y * THREAD_TILE_M;
  int threadTileCol = threadIdx.x * THREAD_TILE_N;

  int numberOfKTiles = matrixSize / BLOCK_TILE_K;

  for (int kTile = 0; kTile < numberOfKTiles; ++kTile) {
    // Load A (BF16 → float) — only rank 0
    if (clusterBlockRank == 0) {
      int globalRowBase = blockTileRow * BLOCK_TILE_M;
      int globalColBase = kTile * BLOCK_TILE_K;
#pragma unroll
      for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_K) / (BLOCK_DIM_X * BLOCK_DIM_Y); ++i) {
        int linearIdx = threadId + i * (BLOCK_DIM_X * BLOCK_DIM_Y);
        int localRow = linearIdx / BLOCK_TILE_K;
        int localCol = linearIdx % BLOCK_TILE_K;
        sharedA[localRow][localCol] = __bfloat162float(
            inputMatrixA[(globalRowBase + localRow) * matrixSize +
                         globalColBase + localCol]);
      }
    }

    // Load B (BF16 → float) — each block loads its own
    {
      int globalRowBase = kTile * BLOCK_TILE_K;
      int globalColBase = blockTileCol * BLOCK_TILE_N;
#pragma unroll
      for (int i = 0; i < (BLOCK_TILE_K * BLOCK_TILE_N) / (BLOCK_DIM_X * BLOCK_DIM_Y); ++i) {
        int linearIdx = threadId + i * (BLOCK_DIM_X * BLOCK_DIM_Y);
        int localRow = linearIdx / BLOCK_TILE_N;
        int localCol = linearIdx % BLOCK_TILE_N;
        sharedB[localRow][localCol] = __bfloat162float(
            inputMatrixB[(globalRowBase + localRow) * matrixSize +
                         globalColBase + localCol]);
      }
    }

    __syncthreads();
    clusterGroup.sync();

    // Same as FP32 from here — shared memory is float
    const float (*tileA)[BLOCK_TILE_K];
    if (clusterBlockRank == 0) {
      tileA = sharedA;
    } else {
      tileA = clusterGroup.map_shared_rank(sharedA, 0);
    }

#pragma unroll
    for (int k = 0; k < BLOCK_TILE_K; ++k) {
      float regA[THREAD_TILE_M];
      float regB[THREAD_TILE_N];
#pragma unroll
      for (int m = 0; m < THREAD_TILE_M; ++m)
        regA[m] = tileA[threadTileRow + m][k];
#pragma unroll
      for (int n = 0; n < THREAD_TILE_N; ++n)
        regB[n] = sharedB[k][threadTileCol + n];
#pragma unroll
      for (int m = 0; m < THREAD_TILE_M; ++m)
#pragma unroll
        for (int n = 0; n < THREAD_TILE_N; ++n)
          accumulator[m][n] += regA[m] * regB[n];
    }

    __syncthreads();
    clusterGroup.sync();
  }

  int globalRowBase = blockTileRow * BLOCK_TILE_M + threadTileRow;
  int globalColBase = blockTileCol * BLOCK_TILE_N + threadTileCol;
#pragma unroll
  for (int m = 0; m < THREAD_TILE_M; ++m) {
    int row = globalRowBase + m;
    if (row < matrixSize) {
#pragma unroll
      for (int n = 0; n < THREAD_TILE_N; ++n) {
        int col = globalColBase + n;
        if (col < matrixSize)
          outputMatrixC[row * matrixSize + col] = accumulator[m][n];
      }
    }
  }
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

  dim3 blockDimensions(BLOCK_DIM_X, BLOCK_DIM_Y); // 16 × 16 = 256 threads
  dim3 gridDimensions(matrixSize / BLOCK_TILE_M, matrixSize / BLOCK_TILE_N);

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

  dim3 blockDimensions(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 gridDimensions(matrixSize / BLOCK_TILE_M, matrixSize / BLOCK_TILE_N);

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

  // Minimum: BLOCK_TILE_M * CLUSTER_DIM_Y = 256
  // (need gridDim.y ≥ CLUSTER_DIM_Y and divisible by it)
  const int testMatrixSizes[] = {256, 512, 1024, 2048, 4096};
  const int numberOfTestSizes =
      (int)(sizeof(testMatrixSizes) / sizeof(testMatrixSizes[0]));
  const int warmupIterations = 5;
  const int benchmarkIterations = 20;
  const int verificationSizeLimit = 512;

  printf("============================================================\n");
  printf("Thread Block Cluster MatMul (thread-tiled)\n");
  printf("Block tile: %d×%d,  Thread tile: %d×%d,  K-tile: %d\n",
         BLOCK_TILE_M, BLOCK_TILE_N, THREAD_TILE_M, THREAD_TILE_N,
         BLOCK_TILE_K);
  printf("Cluster dims: %d × %d,  Threads/block: %d\n", CLUSTER_DIM_X,
         CLUSTER_DIM_Y, BLOCK_DIM_X * BLOCK_DIM_Y);
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
    float *deviceOutputC16Cluster;
    float *deviceInputA32, *deviceInputB32;
    float *deviceOutputC32Cluster;

    CUDA_CHECK(cudaMalloc(&deviceInputA16, memorySizeBFloat16));
    CUDA_CHECK(cudaMalloc(&deviceInputB16, memorySizeBFloat16));
    CUDA_CHECK(cudaMalloc(&deviceOutputC16Cluster, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceInputA32, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceInputB32, memorySizeFloat32));
    CUDA_CHECK(cudaMalloc(&deviceOutputC32Cluster, memorySizeFloat32));

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
    CUDA_CHECK(cudaFree(deviceInputA32));
    CUDA_CHECK(cudaFree(deviceInputB32));
    CUDA_CHECK(cudaFree(deviceOutputC32Cluster));
  }

  printf("\n✅  All benchmarks complete.\n");
  return 0;
}
