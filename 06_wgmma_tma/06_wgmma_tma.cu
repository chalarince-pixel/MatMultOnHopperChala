// ============================================================
// WGMMA + TMA (Tensor Memory Accelerator) Matrix Multiplication
// Target GPU: NVIDIA H100 (sm_90a)
// Phase 2: TMA bulk tensor copies and Swizzled Shared Memory
// ============================================================

#include "../include/utils.cuh"
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#pragma nv_diag_suppress 20013 
#include <cuda/barrier>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────────────────────
// Geometry & Pipeline Setup
// ─────────────────────────────────────────────────────────────
#define BLOCK_M 128
#define BLOCK_N 64
#define BLOCK_K 64
#define STAGES 3

#define THREADS 256
#define WARPGROUP_THREADS 128

// ─────────────────────────────────────────────────────────────
// PTX WGMMA and TMA intrinsics
// ─────────────────────────────────────────────────────────────

__device__ __forceinline__ void wgmma_m64n64k16(
    float c[32], uint64_t desc_a, uint64_t desc_b) 
{
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, 1, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]),
          "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]),
          "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]),
          "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31])
        : "l"(desc_a), "l"(desc_b)
    );
}

__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}

__device__ __forceinline__ void tma_copy_2d(
    const CUtensorMap* tmap, 
    uint32_t smem_int_ptr, 
    int col_idx, 
    int row_idx, 
    uint64_t mbar_ptr) 
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_int_ptr), "l"(tmap), "r"(col_idx), "r"(row_idx), "l"(mbar_ptr)
        : "memory"
    );
}

// ─────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────
__global__ void wgmma_tma_bf16_kernel(
    const CUtensorMap* tmapA,
    const CUtensorMap* tmapB,
    float *__restrict__ C, 
    int N) 
{
    int bRow = blockIdx.y * BLOCK_M;
    int bCol = blockIdx.x * BLOCK_N;

    int tid = threadIdx.x;
    int wg_idx = tid / WARPGROUP_THREADS; 
    int wg_row = (wg_idx / 2) * 64;       
    int wg_col = (wg_idx % 2) * 64;       

    __shared__ __align__(128) __nv_bfloat16 sA[STAGES][BLOCK_M][BLOCK_K];
    __shared__ __align__(128) __nv_bfloat16 sB[STAGES][BLOCK_K][BLOCK_N];

    // Hardware mbarrier per stage using libcu++ native barrier
    __shared__ __align__(8) cuda::barrier<cuda::thread_scope_block> mbar[STAGES];

    if (tid == 0) {
        for (int i = 0; i < STAGES; i++) {
            init(&mbar[i], THREADS); 
        }
    }
    __syncthreads();

    float c[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) c[i] = 0.0f;

    auto make_swizzled_desc = [&](const void* smem_ptr, int lead_dim) -> uint64_t {
        uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
        uint64_t desc = (smem_int & 0x3FFFF) >> 4; 
        uint64_t ld_shr = (lead_dim * sizeof(__nv_bfloat16)) >> 4;
        desc |= (ld_shr << 16);
        desc |= (ld_shr << 32); 
        desc |= (1ull << 47);    // 128B Swizzle
        return desc;
    };

    int kSteps = (N + BLOCK_K - 1) / BLOCK_K;
    bool is_tma_producer = (tid == 0);

    int ks = 0;
    for (; ks < STAGES - 1 && ks < kSteps; ks++) {
        if (is_tma_producer) {
            cuda::device::barrier_expect_tx(mbar[ks], 32768); 
            uint64_t mbar_ptr = static_cast<uint64_t>(__cvta_generic_to_shared(&mbar[ks]));
            tma_copy_2d(tmapA, static_cast<uint32_t>(__cvta_generic_to_shared(&sA[ks])), ks * BLOCK_K, bRow, mbar_ptr);
            tma_copy_2d(tmapB, static_cast<uint32_t>(__cvta_generic_to_shared(&sB[ks])), bCol, ks * BLOCK_K, mbar_ptr);
        }
    }

    int compute_stage = 0;
    for (int step = 0; step < kSteps; step++) {
        
        if (ks < kSteps) {
            if (is_tma_producer) {
                cuda::device::barrier_expect_tx(mbar[ks % STAGES], 32768); 
                uint64_t mbar_ptr = static_cast<uint64_t>(__cvta_generic_to_shared(&mbar[ks % STAGES]));
                tma_copy_2d(tmapA, static_cast<uint32_t>(__cvta_generic_to_shared(&sA[ks % STAGES])), ks * BLOCK_K, bRow, mbar_ptr);
                tma_copy_2d(tmapB, static_cast<uint32_t>(__cvta_generic_to_shared(&sB[ks % STAGES])), bCol, ks * BLOCK_K, mbar_ptr);
            }
            ks++;
        }

        mbar[compute_stage].arrive_and_wait();

        #pragma unroll
        for (int k_off = 0; k_off < BLOCK_K; k_off += 16) {
            uint64_t desc_a = make_swizzled_desc(&sA[compute_stage][wg_row][k_off], BLOCK_K);
            uint64_t desc_b = make_swizzled_desc(&sB[compute_stage][k_off][wg_col], BLOCK_N);
            wgmma_m64n64k16(c, desc_a, desc_b);
        }
        wgmma_commit_group();
        wgmma_wait_group();

        mbar[compute_stage].arrive();
        compute_stage = (compute_stage + 1) % STAGES;
    }

    // Dummy store
    int lane = tid % 32;
    int r_base = bRow + wg_row;
    int c_base = bCol + wg_col;
    if (r_base < N && c_base < N && lane == 0) {
        float sum = 0;
        for (int i = 0; i < 32; i++) sum += c[i];
        C[r_base * N + c_base + lane] = sum; 
    }
}

// ─────────────────────────────────────────────────────────────
// Host Driver API Helpers
// ─────────────────────────────────────────────────────────────
void encode_tmap(CUtensorMap* map, void* global_ptr, int global_rows, int global_cols, int block_rows, int block_cols) {
    uint64_t gdim[2] = {(uint64_t)global_cols, (uint64_t)global_rows}; // [col, row]
    uint64_t gstride[1] = {(uint64_t)global_cols * sizeof(__nv_bfloat16)};
    uint32_t bdim[2] = {(uint32_t)block_cols, (uint32_t)block_rows};
    uint32_t estride[2] = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, global_ptr, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (err != CUDA_SUCCESS) {
        const char *err_str;
        cuGetErrorName(err, &err_str);
        printf("ERROR: cuTensorMapEncodeTiled failed: %s\n", err_str);
        exit(1);
    }
}

static float bench(int warmup, int iters, void (*fn)(void *), void *arg) {
    for (int i = 0; i < warmup; i++) fn(arg);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) fn(arg);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms / iters;
}

struct TMA_Args {
    CUtensorMap* tmapA;
    CUtensorMap* tmapB;
    float *C;
    int N;
};

void launch_wgmma_tma(void *p) {
    auto *a = (TMA_Args *)p;
    dim3 blk(THREADS);
    dim3 grd((a->N + BLOCK_N - 1) / BLOCK_N, (a->N + BLOCK_M - 1) / BLOCK_M);
    wgmma_tma_bf16_kernel<<<grd, blk>>>(a->tmapA, a->tmapB, a->C, a->N);
}

int main(void) {
    cuInit(0);
    cudaFree(0); 
    
    printDeviceInfo();
    const int SIZES[] = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int NSIZES = sizeof(SIZES) / sizeof(SIZES[0]);
    
    printf("Hopper Optimized WGMMA + TMA BF16 Benchmark\n");
    printf("dtype,N,ms,tflops\n");

    for (int si = 0; si < NSIZES; si++) {
        int N = SIZES[si];
        size_t bytes_bf16 = (size_t)N * N * sizeof(__nv_bfloat16);
        size_t bytes_f32 = (size_t)N * N * sizeof(float);

        __nv_bfloat16 *dA, *dB;
        float *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes_bf16));
        CUDA_CHECK(cudaMalloc(&dB, bytes_bf16));
        CUDA_CHECK(cudaMalloc(&dC, bytes_f32));

        CUtensorMap host_tmapA, host_tmapB;
        encode_tmap(&host_tmapA, dA, N, N, BLOCK_M, BLOCK_K);
        encode_tmap(&host_tmapB, dB, N, N, BLOCK_K, BLOCK_N);

        CUtensorMap *dev_tmapA, *dev_tmapB;
        CUDA_CHECK(cudaMalloc(&dev_tmapA, sizeof(CUtensorMap)));
        CUDA_CHECK(cudaMalloc(&dev_tmapB, sizeof(CUtensorMap)));
        CUDA_CHECK(cudaMemcpy(dev_tmapA, &host_tmapA, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_tmapB, &host_tmapB, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

        double flops = 2.0 * (double)N * N * N;
        TMA_Args arg = {dev_tmapA, dev_tmapB, dC, N};
        
        float ms = bench(5, 20, launch_wgmma_tma, &arg);
        printf("BF16+TMA,%d,%.4f,%.2f\n", N, ms, flops / (ms * 1e-3) / 1e12);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dev_tmapA));
        CUDA_CHECK(cudaFree(dev_tmapB));
    }
    return 0;
}
