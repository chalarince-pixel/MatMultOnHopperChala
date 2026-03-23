#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.cuh"

// ─────────────────────────────────────────────
// Example kernel: vector addition
// Replace with your assignment kernel(s)
// ─────────────────────────────────────────────
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    printDeviceInfo();

    const int N = 1 << 24;  // 16M elements
    size_t bytes = N * sizeof(float);

    // ── Allocate host memory ──────────────────
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ── Allocate device memory ────────────────
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // ── Copy host → device ────────────────────
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ── Launch kernel ─────────────────────────
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    GpuTimer timer;
    timerStart(&timer);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());         // catch launch errors
    CUDA_CHECK(cudaDeviceSynchronize());    // wait + catch runtime errors

    float ms = timerStop(&timer);
    printf("Kernel time: %.3f ms\n", ms);

    // ── Copy device → host ────────────────────
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // ── Verify ───────────────────────────────
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) { ok = false; break; }
    }
    printf("Result: %s\n", ok ? "✅ CORRECT" : "❌ WRONG");

    // ── Free ──────────────────────────────────
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return 0;
}
