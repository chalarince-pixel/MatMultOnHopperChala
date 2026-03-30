# 03 — WMMA Tensor Core Intrinsics

**Approach:** Warp Matrix Multiply-Accumulate (WMMA) API — manual Tensor Core access
**Performance on H100:** ~200–500 TFlop/s (FP16, depends on tuning)
**Rank:** 3 / 4 — below cuBLAS and TBC, above baseline

---

## What this does

Drives H100 Tensor Cores directly using the WMMA API (`nvcuda::wmma`).
Each **warp** (32 threads) operates on a fixed-size matrix fragment as a unit,
executing a Tensor Core `mma` instruction.
This gives access to Tensor Core throughput without the cuBLAS abstraction,
at the cost of significant implementation complexity.

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
