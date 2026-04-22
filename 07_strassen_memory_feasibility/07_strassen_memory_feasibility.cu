// ============================================================
// Two-level Strassen memory feasibility model (Hopper/H100)
//
// Purpose:
//   Evaluate whether a hybrid DFS/BFS schedule is feasible strictly
//   from a memory perspective:
//     - HBM/global memory footprint
//     - Cluster DSMEM capacity check
//
// Notes:
//   This sample does NOT run Strassen math kernels.
//   It only estimates and optionally validates allocations.
// ============================================================

#include "utils.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum DataType {
  DATA_FP16,
  DATA_BF16,
  DATA_FP32,
};

struct Estimate {
  long double totalBytes;
  long double workspaceBytes;
};

static long double toGiB(long double bytes) {
  return bytes / (1024.0L * 1024.0L * 1024.0L);
}

static bool toSizeT(long double bytes, size_t *output) {
  if (bytes < 0.0L)
    return false;
  if (bytes > (long double)SIZE_MAX)
    return false;
  *output = (size_t)(bytes + 0.5L);
  return true;
}

static const char *yesNo(bool value) { return value ? "YES" : "NO"; }

static bool parseDataType(const char *text, DataType *dtype) {
  if (strcmp(text, "fp16") == 0) {
    *dtype = DATA_FP16;
    return true;
  }
  if (strcmp(text, "bf16") == 0) {
    *dtype = DATA_BF16;
    return true;
  }
  if (strcmp(text, "fp32") == 0) {
    *dtype = DATA_FP32;
    return true;
  }
  return false;
}

static const char *dataTypeName(DataType dtype) {
  switch (dtype) {
  case DATA_FP16:
    return "FP16";
  case DATA_BF16:
    return "BF16";
  case DATA_FP32:
    return "FP32";
  default:
    return "UNKNOWN";
  }
}

static int bytesPerElement(DataType dtype) {
  switch (dtype) {
  case DATA_FP16:
  case DATA_BF16:
    return 2;
  case DATA_FP32:
    return 4;
  default:
    return 4;
  }
}

static Estimate makeEstimate(long double baseABC, long double extraPerParent,
                             int parentCount) {
  Estimate estimate;
  estimate.workspaceBytes = extraPerParent * (long double)parentCount;
  estimate.totalBytes = baseABC + estimate.workspaceBytes;
  return estimate;
}

static bool tryAllocationPlan(size_t fullMatrixBytes, size_t workspaceBytes) {
  void *a = NULL;
  void *b = NULL;
  void *c = NULL;
  void *workspace = NULL;

  cudaError_t err = cudaMalloc(&a, fullMatrixBytes);
  if (err != cudaSuccess)
    goto fail;

  err = cudaMalloc(&b, fullMatrixBytes);
  if (err != cudaSuccess)
    goto fail;

  err = cudaMalloc(&c, fullMatrixBytes);
  if (err != cudaSuccess)
    goto fail;

  err = cudaMalloc(&workspace, workspaceBytes);
  if (err != cudaSuccess)
    goto fail;

  cudaFree(workspace);
  cudaFree(c);
  cudaFree(b);
  cudaFree(a);
  return true;

fail:
  if (workspace)
    cudaFree(workspace);
  if (c)
    cudaFree(c);
  if (b)
    cudaFree(b);
  if (a)
    cudaFree(a);
  cudaGetLastError();
  return false;
}

static int roundDownToMultipleOf4(int value) { return (value / 4) * 4; }

int main(int argc, char **argv) {
  int n = 32768; // Default matrix dimension
  DataType dtype = DATA_FP32;
  int assumedClusterBlocks = 7;

  if (argc >= 2)
    n = atoi(argv[1]);
  if (argc >= 3) {
    if (!parseDataType(argv[2], &dtype)) {
      fprintf(stderr,
              "Unsupported dtype '%s'. Use one of: fp16, bf16, fp32\n",
              argv[2]);
      return EXIT_FAILURE;
    }
  }
  if (argc >= 4)
    assumedClusterBlocks = atoi(argv[3]);

  if (n <= 0 || (n % 4) != 0) {
    fprintf(stderr, "N must be > 0 and divisible by 4 (got %d).\n", n);
    return EXIT_FAILURE;
  }
  if (assumedClusterBlocks <= 0) {
    fprintf(stderr, "Cluster blocks must be > 0 (got %d).\n",
            assumedClusterBlocks);
    return EXIT_FAILURE;
  }

  printDeviceInfo();

  size_t freeBytes = 0;
  size_t totalBytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  int maxShmemPerBlockOptin = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxShmemPerBlockOptin,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    0));

  long double bytesPerElem = (long double)bytesPerElement(dtype);
  long double nSquared = (long double)n * (long double)n;

  long double matrixBytesN = nSquared * bytesPerElem;          // N x N
  long double matrixBytesN2 = (nSquared / 4.0L) * bytesPerElem;  // N/2 x N/2
  long double matrixBytesN4 =
      (nSquared / 16.0L) * bytesPerElem; // N/4 x N/4

  // Base persistent matrices A, B, C at the full level.
  long double baseABC = 3.0L * matrixBytesN;

  // Per parent (M_i) level-2 BFS assumptions:
  // Optimistic:
  //   - 7 child outputs materialized
  //   - 2 child operand temps reused (streamed)
  //   - parent output (N/2)^2
  long double extraPerParentOptimistic =
      7.0L * matrixBytesN4 + 2.0L * matrixBytesN4 + matrixBytesN2;

  // Conservative:
  //   - 7 child outputs materialized
  //   - 14 child operand temps materialized (all children active)
  //   - parent output (N/2)^2
  long double extraPerParentConservative =
      7.0L * matrixBytesN4 + 14.0L * matrixBytesN4 + matrixBytesN2;

  Estimate dfsOptimistic = makeEstimate(baseABC, extraPerParentOptimistic, 1);
  Estimate dfsConservative =
      makeEstimate(baseABC, extraPerParentConservative, 1);
  Estimate bfsOptimistic = makeEstimate(baseABC, extraPerParentOptimistic, 7);
  Estimate bfsConservative =
      makeEstimate(baseABC, extraPerParentConservative, 7);

  // DSMEM feasibility check (theoretical upper bound).
  long double clusterDsmemBudget =
      (long double)assumedClusterBlocks * (long double)maxShmemPerBlockOptin;
  long double oneChildFullMatrices = 3.0L * matrixBytesN4;       // A, B, C
  long double sevenChildFullMatrices = 7.0L * oneChildFullMatrices;

  // N maxima inferred from currently free memory.
  // totalBytes = coeff * N^2 * bytesPerElem
  const long double coeffDfsOptimistic = 61.0L / 16.0L;
  const long double coeffDfsConservative = 73.0L / 16.0L;
  const long double coeffBfsOptimistic = 139.0L / 16.0L;
  const long double coeffBfsConservative = 223.0L / 16.0L;

  long double freeBytesLD = (long double)freeBytes;
  int maxNDfsOptimistic = roundDownToMultipleOf4(
      (int)floorl(sqrtl(freeBytesLD / (coeffDfsOptimistic * bytesPerElem))));
  int maxNDfsConservative = roundDownToMultipleOf4(
      (int)floorl(sqrtl(freeBytesLD / (coeffDfsConservative * bytesPerElem))));
  int maxNBfsOptimistic = roundDownToMultipleOf4(
      (int)floorl(sqrtl(freeBytesLD / (coeffBfsOptimistic * bytesPerElem))));
  int maxNBfsConservative = roundDownToMultipleOf4(
      (int)floorl(sqrtl(freeBytesLD / (coeffBfsConservative * bytesPerElem))));

  printf("============================================================\n");
  printf("Two-level Strassen Memory Feasibility (memory-only)\n");
  printf("============================================================\n");
  printf("Inputs:\n");
  printf("  N                         : %d\n", n);
  printf("  Data type                 : %s (%d bytes/element)\n",
         dataTypeName(dtype), (int)bytesPerElement(dtype));
  printf("  Assumed cluster blocks    : %d\n", assumedClusterBlocks);
  printf("  Free HBM now              : %.2Lf GiB\n", toGiB(freeBytesLD));
  printf("  Total HBM                 : %.2Lf GiB\n", toGiB((long double)totalBytes));
  printf("\n");

  printf("Model assumptions:\n");
  printf("  Level 1 DFS = one parent M_i at a time\n");
  printf("  Level 1 BFS = seven parent M_i active at once\n");
  printf("  Level 2 BFS = seven child multiplies per parent active\n");
  printf("\n");

  printf("Base allocations:\n");
  printf("  A, B, C full matrices     : %.2Lf GiB\n", toGiB(baseABC));
  printf("\n");

  printf("Estimated total global-memory footprint:\n");
  printf("  DFS + child BFS (optimistic)   : %8.2Lf GiB   fits free HBM: %s\n",
         toGiB(dfsOptimistic.totalBytes),
         yesNo(dfsOptimistic.totalBytes <= freeBytesLD));
  printf("  DFS + child BFS (conservative) : %8.2Lf GiB   fits free HBM: %s\n",
         toGiB(dfsConservative.totalBytes),
         yesNo(dfsConservative.totalBytes <= freeBytesLD));
  printf("  BFS + child BFS (optimistic)   : %8.2Lf GiB   fits free HBM: %s\n",
         toGiB(bfsOptimistic.totalBytes),
         yesNo(bfsOptimistic.totalBytes <= freeBytesLD));
  printf("  BFS + child BFS (conservative) : %8.2Lf GiB   fits free HBM: %s\n",
         toGiB(bfsConservative.totalBytes),
         yesNo(bfsConservative.totalBytes <= freeBytesLD));
  printf("\n");

  printf("DSMEM upper-bound check (theoretical):\n");
  printf("  Max shmem/block (opt-in)   : %.2Lf KiB\n",
         (long double)maxShmemPerBlockOptin / 1024.0L);
  printf("  Assumed cluster DSMEM pool : %.2Lf KiB\n",
         clusterDsmemBudget / 1024.0L);
  printf("  One level-2 child full A+B+C in DSMEM   : %.6Lf GiB   fits: %s\n",
         toGiB(oneChildFullMatrices),
         yesNo(oneChildFullMatrices <= clusterDsmemBudget));
  printf("  Seven level-2 children full A+B+C DSMEM : %.6Lf GiB   fits: %s\n",
         toGiB(sevenChildFullMatrices),
         yesNo(sevenChildFullMatrices <= clusterDsmemBudget));
  printf("\n");

  long double nLimitOneChild =
      4.0L * sqrtl(clusterDsmemBudget / (3.0L * bytesPerElem));
  long double nLimitSevenChildren =
      4.0L * sqrtl(clusterDsmemBudget / (21.0L * bytesPerElem));
  printf("N upper bound for full level-2 matrix storage in DSMEM:\n");
  printf("  One child  (A+B+C)         : N <= %.0Lf\n", nLimitOneChild);
  printf("  Seven children (A+B+C each): N <= %.0Lf\n", nLimitSevenChildren);
  printf("\n");

  printf("Max N from current FREE HBM (same model assumptions):\n");
  printf("  DFS + child BFS (optimistic)   : N <= %d\n", maxNDfsOptimistic);
  printf("  DFS + child BFS (conservative) : N <= %d\n", maxNDfsConservative);
  printf("  BFS + child BFS (optimistic)   : N <= %d\n", maxNBfsOptimistic);
  printf("  BFS + child BFS (conservative) : N <= %d\n", maxNBfsConservative);
  printf("\n");

  // Optional practical verification by real allocations.
  size_t fullMatrixBytesSizeT = 0;
  size_t dfsWorkspaceConservativeSizeT = 0;
  size_t bfsWorkspaceConservativeSizeT = 0;

  bool canConvertFullMatrix = toSizeT(matrixBytesN, &fullMatrixBytesSizeT);
  bool canConvertDfsWorkspace =
      toSizeT(dfsConservative.workspaceBytes, &dfsWorkspaceConservativeSizeT);
  bool canConvertBfsWorkspace =
      toSizeT(bfsConservative.workspaceBytes, &bfsWorkspaceConservativeSizeT);

  printf("Allocation probe (cudaMalloc plan test):\n");
  if (canConvertFullMatrix && canConvertDfsWorkspace) {
    bool dfsAllocOk =
        tryAllocationPlan(fullMatrixBytesSizeT, dfsWorkspaceConservativeSizeT);
    printf("  DFS + child BFS (conservative): %s\n", yesNo(dfsAllocOk));
  } else {
    printf("  DFS + child BFS (conservative): SKIPPED (size conversion overflow)\n");
  }

  if (canConvertFullMatrix && canConvertBfsWorkspace) {
    bool bfsAllocOk =
        tryAllocationPlan(fullMatrixBytesSizeT, bfsWorkspaceConservativeSizeT);
    printf("  BFS + child BFS (conservative): %s\n", yesNo(bfsAllocOk));
  } else {
    printf("  BFS + child BFS (conservative): SKIPPED (size conversion overflow)\n");
  }

  printf("\nDone.\n");
  return 0;
}
