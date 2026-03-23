#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────
// Error checking macro
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────
// Timer utilities
// ─────────────────────────────────────────────
typedef struct {
    cudaEvent_t start, stop;
} GpuTimer;

inline void timerStart(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
    CUDA_CHECK(cudaEventRecord(t->start));
}

inline float timerStop(GpuTimer *t) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(t->stop));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
    return ms;
}

// ─────────────────────────────────────────────
// Device info helper
// ─────────────────────────────────────────────
inline void printDeviceInfo() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== GPU: %s ===\n", prop.name);
    printf("  Compute Capability : %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory      : %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  SMs                : %d\n", prop.multiProcessorCount);
    printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size          : %d\n", prop.warpSize);
    printf("================================\n\n");
}
