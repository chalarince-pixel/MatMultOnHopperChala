// ============================================================
// WMMA Tensor Core matrix multiplication benchmark
// BF16 & TF32 paths: Optimized pipeline with cp.async and STAGES
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
#define PIPELINE_STAGES 3

// BF16 WMMA tile
#define BF16_WMMA_M 16
#define BF16_WMMA_N 16
#define BF16_WMMA_K 16

// TF32 WMMA tile
#define TF32_WMMA_M 16
#define TF32_WMMA_N 16
#define TF32_WMMA_K 8

// ==== BF16 OPTIMIZED PARAMS ====
#define BF16_BLOCK_M 128
#define BF16_BLOCK_N 128
#define BF16_BLOCK_K 32 
#define BF16_WARPS_M 4
#define BF16_WARPS_N 2
#define BF16_THREADS (BF16_WARPS_M * BF16_WARPS_N * WARP_SIZE)

// ==== TF32 PARAMS ====
#define TF32_BLOCK_M 64
#define TF32_BLOCK_N 64
#define TF32_BLOCK_K 32
#define TF32_WARPS_M 4
#define TF32_WARPS_N 4
#define TF32_THREADS (TF32_WARPS_M * TF32_WARPS_N * WARP_SIZE)

// ─────────────────────────────────────────────────────────────
// PTX cp.async helpers
// ─────────────────────────────────────────────────────────────
__device__ __forceinline__ void copyAsync16Bytes(void* sharedPtr, const void* globalPtr, bool mask) {
    uint32_t sharedInt = static_cast<uint32_t>(__cvta_generic_to_shared(sharedPtr));
    int predicate = mask ? 1 : 0;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n" ::"r"(sharedInt), "l"(globalPtr), "r"(predicate));
}

__device__ __forceinline__ void copyAsyncCommit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int numGroups> __device__ __forceinline__ void copyAsyncWaitGroup() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(numGroups));
}

// ─────────────────────────────────────────────────────────────
// BF16 WMMA Kernel (Optimized Pipeline)
// ─────────────────────────────────────────────────────────────
__global__ void wmmaBfloat16(const __nv_bfloat16 *__restrict__ ptrA,
                          const __nv_bfloat16 *__restrict__ ptrB,
                          float *__restrict__ ptrC, int matrixSize) {
  // L2 Cache Swizzling (Group 8 blocks together)
  const int cacheSwizzle = 8;
  const int gridWidth = gridDim.x;
  const int linearBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
  const int swizzledIndex = linearBlockIndex / cacheSwizzle;
  const int swizzledRemainder = linearBlockIndex % cacheSwizzle;
  
  const int panelIndex = swizzledIndex * cacheSwizzle;
  const int panelWidth = min(cacheSwizzle, gridWidth - panelIndex);
  
  const int blockColumnIndex = panelIndex + (swizzledRemainder % panelWidth);
  const int blockRowIndex = (swizzledIndex * cacheSwizzle + swizzledRemainder) / gridWidth;

  const int blockRow = blockRowIndex * BF16_BLOCK_M;
  const int blockColumn = blockColumnIndex * BF16_BLOCK_N;

  const int warpId = threadIdx.x / WARP_SIZE;
  const int warpRow = warpId / BF16_WARPS_N; 
  const int warpColumn = warpId % BF16_WARPS_N; 

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[2][4];
  #pragma unroll
  for(int i=0; i<2; i++) {
    #pragma unroll
    for(int j=0; j<4; j++) {
      wmma::fill_fragment(accumulators[i][j], 0.0f);
    }
  }

  // Padding to avoid bank conflicts
  __shared__ __align__(16) __nv_bfloat16 sharedA[PIPELINE_STAGES][BF16_BLOCK_M][BF16_BLOCK_K + 8];
  __shared__ __align__(16) __nv_bfloat16 sharedB[PIPELINE_STAGES][BF16_BLOCK_K][BF16_BLOCK_N + 8];

  const int threadId = threadIdx.x;
  
  auto loadStage = [&](int currentStage, int kBaseOffset) {
      #pragma unroll
      for (int i = 0; i < 2; i++) {
          int chunkIndex = threadId + i * BF16_THREADS;
          int elementIndex = chunkIndex * 8; 
          int localRow = elementIndex / BF16_BLOCK_K;
          int localK = elementIndex % BF16_BLOCK_K;
          int globalRow = blockRow + localRow;
          int globalK = kBaseOffset + localK;
          bool mask = (globalRow < matrixSize && globalK < matrixSize);
          copyAsync16Bytes(&sharedA[currentStage][localRow][localK], &ptrA[globalRow * matrixSize + globalK], mask);
      }
      
      #pragma unroll
      for (int i = 0; i < 2; i++) {
          int chunkIndex = threadId + i * BF16_THREADS;
          int elementIndex = chunkIndex * 8;
          int localK = elementIndex / BF16_BLOCK_N;
          int localColumn = elementIndex % BF16_BLOCK_N;
          int globalK = kBaseOffset + localK;
          int globalColumn = blockColumn + localColumn;
          bool mask = (globalK < matrixSize && globalColumn < matrixSize);
          copyAsync16Bytes(&sharedB[currentStage][localK][localColumn], &ptrB[globalK * matrixSize + globalColumn], mask);
      }
      copyAsyncCommit();
  };

  const int kSteps = (matrixSize + BF16_BLOCK_K - 1) / BF16_BLOCK_K;
  if (kSteps == 0) return;

  // Prologue
  for (int i = 0; i < PIPELINE_STAGES - 1; i++) {
      if (i < kSteps) {
          loadStage(i, i * BF16_BLOCK_K);
      }
  }

  // Main compute loop
  int computeStage = 0;
  for (int step = 0; step < kSteps; step++) {
      int nextFetchStep = step + PIPELINE_STAGES - 1;
      int fetchStage = nextFetchStep % PIPELINE_STAGES;
      
      if (nextFetchStep < kSteps) {
          loadStage(fetchStage, nextFetchStep * BF16_BLOCK_K);
      }
      
      copyAsyncWaitGroup<PIPELINE_STAGES - 2>();
      __syncthreads();
      
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fragmentA[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fragmentB[4];

      #pragma unroll
      for (int kIndex = 0; kIndex < BF16_BLOCK_K; kIndex += 16) {
          wmma::load_matrix_sync(fragmentA[0], &sharedA[computeStage][warpRow * 32 + 0][kIndex], BF16_BLOCK_K + 8);
          wmma::load_matrix_sync(fragmentA[1], &sharedA[computeStage][warpRow * 32 + 16][kIndex], BF16_BLOCK_K + 8);
          
          wmma::load_matrix_sync(fragmentB[0], &sharedB[computeStage][kIndex][warpColumn * 64 + 0], BF16_BLOCK_N + 8);
          wmma::load_matrix_sync(fragmentB[1], &sharedB[computeStage][kIndex][warpColumn * 64 + 16], BF16_BLOCK_N + 8);
          wmma::load_matrix_sync(fragmentB[2], &sharedB[computeStage][kIndex][warpColumn * 64 + 32], BF16_BLOCK_N + 8);
          wmma::load_matrix_sync(fragmentB[3], &sharedB[computeStage][kIndex][warpColumn * 64 + 48], BF16_BLOCK_N + 8);

          #pragma unroll
          for (int i = 0; i < 2; i++) {
              #pragma unroll
              for (int j = 0; j < 4; j++) {
                  wmma::mma_sync(accumulators[i][j], fragmentA[i], fragmentB[j], accumulators[i][j]);
              }
          }
      }
      
      __syncthreads(); 
      computeStage = (computeStage + 1) % PIPELINE_STAGES;
  }

  #pragma unroll
  for(int i=0; i<2; i++) {
      #pragma unroll
      for(int j=0; j<4; j++) {
          int outputRow = blockRow + warpRow * 32 + i * 16;
          int outputColumn = blockColumn + warpColumn * 64 + j * 16;
          if (outputRow < matrixSize && outputColumn < matrixSize) {
              wmma::store_matrix_sync(ptrC + outputRow * matrixSize + outputColumn, accumulators[i][j], matrixSize, wmma::mem_row_major);
          }
      }
  }
}

// ─────────────────────────────────────────────────────────────
// TF32 WMMA Kernel (Optimized Pipeline)
// ─────────────────────────────────────────────────────────────
__global__ void wmmaTensorFloat32(const float *__restrict__ ptrA,
                          const float *__restrict__ ptrB, float *__restrict__ ptrC,
                          int matrixSize) {
  // L2 Cache Swizzling (Group 8 blocks together)
  const int cacheSwizzle = 8;
  const int gridWidth = gridDim.x;
  const int linearBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
  const int swizzledIndex = linearBlockIndex / cacheSwizzle;
  const int swizzledRemainder = linearBlockIndex % cacheSwizzle;

  const int panelIndex = swizzledIndex * cacheSwizzle;
  const int panelWidth = min(cacheSwizzle, gridWidth - panelIndex);

  const int blockColumnIndex = panelIndex + (swizzledRemainder % panelWidth);
  const int blockRowIndex = (swizzledIndex * cacheSwizzle + swizzledRemainder) / gridWidth;

  const int blockRow = blockRowIndex * TF32_BLOCK_M;
  const int blockColumn = blockColumnIndex * TF32_BLOCK_N;

  const int warpId = threadIdx.x / WARP_SIZE;
  const int warpRow = warpId / TF32_WARPS_N;
  const int warpColumn = warpId % TF32_WARPS_N;

  wmma::fragment<wmma::accumulator, TF32_WMMA_M, TF32_WMMA_N, TF32_WMMA_K, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  __shared__ __align__(16) float sharedA[PIPELINE_STAGES][TF32_BLOCK_M][TF32_BLOCK_K + 4]; 
  __shared__ __align__(16) float sharedB[PIPELINE_STAGES][TF32_BLOCK_K][TF32_BLOCK_N + 4]; 

  const int threadId = threadIdx.x;
  
  auto loadStage = [&](int currentStage, int kBaseOffset) {
      // Load A (64x32 = 2048 floats = 512 chunks of 16 bytes). 1 chunk per thread.
      int chunkIndex = threadId;
      int elementIndex = chunkIndex * 4; 
      int localRow = elementIndex / TF32_BLOCK_K;
      int localK = elementIndex % TF32_BLOCK_K;
      int globalRow = blockRow + localRow;
      int globalK = kBaseOffset + localK;
      bool maskA = (globalRow < matrixSize && globalK < matrixSize);
      copyAsync16Bytes(&sharedA[currentStage][localRow][localK], &ptrA[globalRow * matrixSize + globalK], maskA);
      
      // Load B (32x64 = 2048 floats = 512 chunks of 16 bytes). 1 chunk per thread.
      int localColumn = elementIndex % TF32_BLOCK_N;
      int localKForB = elementIndex / TF32_BLOCK_N;
      int globalKForB = kBaseOffset + localKForB;
      int globalColumnForB = blockColumn + localColumn;
      bool maskB = (globalKForB < matrixSize && globalColumnForB < matrixSize);
      copyAsync16Bytes(&sharedB[currentStage][localKForB][localColumn], &ptrB[globalKForB * matrixSize + globalColumnForB], maskB);
      
      copyAsyncCommit();
  };

  const int kSteps = (matrixSize + TF32_BLOCK_K - 1) / TF32_BLOCK_K;
  if(kSteps == 0) return;

  // Prologue
  for (int i = 0; i < PIPELINE_STAGES - 1; i++) {
      if (i < kSteps) loadStage(i, i * TF32_BLOCK_K);
  }

  int computeStage = 0;
  for (int step = 0; step < kSteps; step++) {
    int nextFetchStep = step + PIPELINE_STAGES - 1;
    int fetchStage = nextFetchStep % PIPELINE_STAGES;
      
    if (nextFetchStep < kSteps) {
        loadStage(fetchStage, nextFetchStep * TF32_BLOCK_K);
    }
      
    copyAsyncWaitGroup<PIPELINE_STAGES - 2>();
    __syncthreads();

    wmma::fragment<wmma::matrix_a, TF32_WMMA_M, TF32_WMMA_N, TF32_WMMA_K,
                   wmma::precision::tf32, wmma::row_major> fragmentA;
    wmma::fragment<wmma::matrix_b, TF32_WMMA_M, TF32_WMMA_N, TF32_WMMA_K,
                   wmma::precision::tf32, wmma::row_major> fragmentB;

    #pragma unroll
    for (int kIndex = 0; kIndex < TF32_BLOCK_K; kIndex += TF32_WMMA_K) {
      wmma::load_matrix_sync(fragmentA, &sharedA[computeStage][warpRow * TF32_WMMA_M][kIndex], TF32_BLOCK_K + 4);
      wmma::load_matrix_sync(fragmentB, &sharedB[computeStage][kIndex][warpColumn * TF32_WMMA_N], TF32_BLOCK_N + 4);
      wmma::mma_sync(accumulator, fragmentA, fragmentB, accumulator);
    }

    __syncthreads();
    computeStage = (computeStage + 1) % PIPELINE_STAGES;
  }

  const int outputRow = blockRow + warpRow * TF32_WMMA_M;
  const int outputColumn = blockColumn + warpColumn * TF32_WMMA_N;
  if (outputRow < matrixSize && outputColumn < matrixSize) {
    wmma::store_matrix_sync(ptrC + outputRow * matrixSize + outputColumn, accumulator, matrixSize, wmma::mem_row_major);
  }
}

// ─────────────────────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────────────────────
static float runBenchmark(int warmupIterations, int totalIterations, void (*testFunction)(void *), void *functionArguments) {
  for (int i = 0; i < warmupIterations; i++)
    testFunction(functionArguments);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t eventStart, eventEnd;
  CUDA_CHECK(cudaEventCreate(&eventStart));
  CUDA_CHECK(cudaEventCreate(&eventEnd));
  CUDA_CHECK(cudaEventRecord(eventStart));
  for (int i = 0; i < totalIterations; i++)
    testFunction(functionArguments);
  CUDA_CHECK(cudaEventRecord(eventEnd));
  CUDA_CHECK(cudaEventSynchronize(eventEnd));
  float elapsedMilliseconds = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, eventStart, eventEnd));
  cudaEventDestroy(eventStart);
  cudaEventDestroy(eventEnd);
  return elapsedMilliseconds / totalIterations;
}

struct BFloat16Arguments {
  __nv_bfloat16 *ptrA, *ptrB;
  float *ptrC;
  int matrixSize;
};

static void launchBFloat16Kernel(void *parameters) {
  auto *args = (BFloat16Arguments *)parameters;
  dim3 blockDimensions(BF16_THREADS);
  dim3 gridDimensions((args->matrixSize + BF16_BLOCK_N - 1) / BF16_BLOCK_N, (args->matrixSize + BF16_BLOCK_M - 1) / BF16_BLOCK_M);
  wmmaBfloat16<<<gridDimensions, blockDimensions>>>(args->ptrA, args->ptrB, args->ptrC, args->matrixSize);
}

struct TensorFloat32Arguments {
  float *ptrA, *ptrB, *ptrC;
  int matrixSize;
};

static void launchTensorFloat32Kernel(void *parameters) {
  auto *args = (TensorFloat32Arguments *)parameters;
  dim3 blockDimensions(TF32_THREADS);
  dim3 gridDimensions((args->matrixSize + TF32_BLOCK_N - 1) / TF32_BLOCK_N, (args->matrixSize + TF32_BLOCK_M - 1) / TF32_BLOCK_M);
  wmmaTensorFloat32<<<gridDimensions, blockDimensions>>>(args->ptrA, args->ptrB, args->ptrC, args->matrixSize);
}

int main(void) {
  printDeviceInfo();

  const int testSizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
  const int numSizes = (int)(sizeof(testSizes) / sizeof(testSizes[0]));
  const int warmupRuns = 5;
  const int totalRuns = 20;

  printf("dtype,N,ms,tflops\n");

  for (int sizeIndex = 0; sizeIndex < numSizes; sizeIndex++) {
    int matrixSize = testSizes[sizeIndex];
    size_t memoryBytesBFloat16 = (size_t)matrixSize * matrixSize * sizeof(__nv_bfloat16);
    size_t memoryBytesFloat32 = (size_t)matrixSize * matrixSize * sizeof(float);

    __nv_bfloat16 *deviceA16, *deviceB16;
    float *deviceC16;
    float *deviceA32, *deviceB32, *deviceC32;

    CUDA_CHECK(cudaMalloc(&deviceA16, memoryBytesBFloat16));
    CUDA_CHECK(cudaMalloc(&deviceB16, memoryBytesBFloat16));
    CUDA_CHECK(cudaMalloc(&deviceC16, memoryBytesFloat32));
    CUDA_CHECK(cudaMalloc(&deviceA32, memoryBytesFloat32));
    CUDA_CHECK(cudaMalloc(&deviceB32, memoryBytesFloat32));
    CUDA_CHECK(cudaMalloc(&deviceC32, memoryBytesFloat32));

    {
      __nv_bfloat16 *hostA16 = (__nv_bfloat16 *)malloc(memoryBytesBFloat16);
      __nv_bfloat16 *hostB16 = (__nv_bfloat16 *)malloc(memoryBytesBFloat16);
      float *hostA32 = (float *)malloc(memoryBytesFloat32);
      float *hostB32 = (float *)malloc(memoryBytesFloat32);
      for (int i = 0; i < matrixSize * matrixSize; i++) {
        float randomValue = (float)(rand() % 10) / 10.0f;
        hostA16[i] = __float2bfloat16(randomValue);
        hostB16[i] = __float2bfloat16(randomValue);
        hostA32[i] = randomValue;
        hostB32[i] = randomValue;
      }
      CUDA_CHECK(cudaMemcpy(deviceA16, hostA16, memoryBytesBFloat16, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(deviceB16, hostB16, memoryBytesBFloat16, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(deviceA32, hostA32, memoryBytesFloat32, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(deviceB32, hostB32, memoryBytesFloat32, cudaMemcpyHostToDevice));
      free(hostA16);
      free(hostB16);
      free(hostA32);
      free(hostB32);
    }

    double floatingPointOperations = 2.0 * (double)matrixSize * matrixSize * matrixSize;

    {
      BFloat16Arguments functionArgs = {deviceA16, deviceB16, deviceC16, matrixSize};
      float elapsedMilliseconds = runBenchmark(warmupRuns, totalRuns, launchBFloat16Kernel, &functionArgs);
      printf("BF16,%d,%.4f,%.2f\n", matrixSize, elapsedMilliseconds, floatingPointOperations / (elapsedMilliseconds * 1e-3) / 1e12);
    }

    {
      TensorFloat32Arguments functionArgs = {deviceA32, deviceB32, deviceC32, matrixSize};
      float elapsedMilliseconds = runBenchmark(warmupRuns, totalRuns, launchTensorFloat32Kernel, &functionArgs);
      printf("FP32,%d,%.4f,%.2f\n", matrixSize, elapsedMilliseconds, floatingPointOperations / (elapsedMilliseconds * 1e-3) / 1e12);
    }

    CUDA_CHECK(cudaFree(deviceA16));
    CUDA_CHECK(cudaFree(deviceB16));
    CUDA_CHECK(cudaFree(deviceC16));
    CUDA_CHECK(cudaFree(deviceA32));
    CUDA_CHECK(cudaFree(deviceB32));
    CUDA_CHECK(cudaFree(deviceC32));
  }

  return 0;
}
