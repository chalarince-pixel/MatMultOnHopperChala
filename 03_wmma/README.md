# 03 — WMMA Tensor Core Intrinsics

**Approach:** Warp Matrix Multiply-Accumulate (WMMA) API — manual Tensor Core access
**Performance on H100:** ~200–500 TFlop/s (FP16, depends on tuning)
**Rank:** 3 / 4 — below cuBLAS and TBC, above baseline

---

## What this does (High-Level Overview)

This implementation drives the H100 Tensor Cores directly using the **Warp Matrix Multiply-Accumulate (WMMA) API**. At an abstract level, it demonstrates how to build a highly optimized, custom matrix multiplication kernel from the ground up without relying on black-box libraries like cuBLAS.

The code accomplishes this through a heavily optimized pipeline:
1. **Warp-Level Execution:** Each warp (32 threads) collaboratively loads and computes a fixed-size tile of the matrix (e.g., 16x16x16).
2. **Multi-Stage Pipelining (Asynchronous Copies):** Instead of waiting for data to load from global memory, the kernel uses `cp.async` to pre-fetch multiple "stages" of data in the background while the Tensor Cores are busy computing the current stage.
3. **Shared Memory Bank Conflict Padding:** Data loaded into shared memory is strategically padded to ensure that when warps read from it, they don't request data from the same memory banks simultaneously, avoiding severe pipeline stalls.
4. **L2 Cache Swizzling:** The thread blocks are scheduled in a 2D "swizzled" panel pattern (rather than a simple row-by-row sequence). This ensures that blocks computing adjacent output tiles run concurrently, maximizing data reuse in the GPU's L2 cache and drastically reducing memory bandwidth pressure.
5. **Precision Profiling:** It specifically contrasts the performance of **BF16** (2 bytes per element) and **TF32** (4 bytes per element) to demonstrate how bandwidth and shared memory capacity limit Tensor Core throughput.

---

## Key Concepts

### Warp-level Tile Sizes on sm_90

| Precision A/B | Accumulator | Fragment shape |
|---|---|---|
| `half` (FP16) | `float` (FP32) | 16×16×16 |
| `__nv_bfloat16` (BF16) | `float` (FP32) | 16×16×16 |
| `double` (FP64) | `double` (FP64) | 8×4×8 |

### WMMA API Overview

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Declare fragments (held collectively by all 32 threads in the warp)
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float>         c_frag;

fill_fragment(c_frag, 0.0f);                         // zero accumulator
load_matrix_sync(a_frag, A_ptr, lda);                // load tile from memory
load_matrix_sync(b_frag, B_ptr, ldb);
mma_sync(c_frag, a_frag, b_frag, c_frag);            // Tensor Core MMA
store_matrix_sync(C_ptr, c_frag, ldc, mem_row_major); // store result
```

### Why it's slower than cuBLAS (without careful tuning)

- Without **double-buffering** (overlapping `cp.async` loads with compute),
  the warp stalls waiting for data — killing Tensor Core utilisation
- Without **warp-level tiling strategy**, shared memory bank conflicts
  degrade effective bandwidth

### Path to closing the gap with cuBLAS

1. Use `cp.async` for asynchronous tile loads
2. Pipeline with `cuda::pipeline` (CUDA 11+)
3. Use 64×64 or 128×128 global tiles per block (multiple WMMA calls per block)

---

## Build & Run

```bash
# From the repo root
make 03_wmma
./build/03_wmma
```

## Headers Required

```c
#include <mma.h>
#include <cuda_runtime.h>
#include "include/utils.cuh"
```

No extra link flags needed — WMMA is part of the CUDA runtime.
