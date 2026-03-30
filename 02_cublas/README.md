# 02 — cuBLAS Matrix Multiplication

**Approach:** NVIDIA cuBLAS library (Tensor Core accelerated)
**Performance on H100:** ~800–900 TFlop/s (FP16, Tensor Cores)
**Rank:** 1 / 4 — highest performance

---

## What this does

Delegates `C = A × B` entirely to the cuBLAS library.
cuBLAS internally uses Tensor Cores, async memory pipelines, double-buffering,
and architecture-specific tuning accumulated over years — achieving 85–95% of
H100 theoretical peak with minimal user code.

---

## Key Concepts

### Why cuBLAS is fastest

cuBLAS is not just a wrapper — NVIDIA ships hand-optimised kernels for every
SM generation. On H100 (sm_90 / Hopper), it automatically activates:

- **Tensor Cores** for 4×4 matrix multiply-accumulate in one instruction
- **`cp.async`** for asynchronous global → shared memory copies
- **Double-buffering** so compute and memory load overlap
- **Thread Block Clusters** internally for large tiles

### API Overview

| Call | Purpose |
|---|---|
| `cublasCreate(&handle)` | Initialise a cuBLAS context |
| `cublasSgemm(...)` | FP32 GEMM (uses Tensor Cores in TF32 mode on Hopper) |
| `cublasGemmEx(...)` | Mixed-precision GEMM — use `CUDA_R_16F` for FP16 Tensor Core path |
| `cublasDestroy(handle)` | Release the context |

### TF32 vs FP16

- **TF32** (`cublasSgemm` default on Hopper): same interface as FP32, ~10× faster
  than FP32 CUDA cores with no code change
- **FP16** (`cublasGemmEx` with `CUDA_R_16F`): full Tensor Core throughput,
  ~989 TFlop/s theoretical

---

## Build & Run

```bash
# From the repo root
make 02_cublas
./build/02_cublas
```

> **Note:** Link with `-lcublas` — already set in the Makefile `LDFLAGS`.

---

## Headers Required

```c
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "include/utils.cuh"   // CUDA_CHECK, GpuTimer, printDeviceInfo
```

---

## Performance Note

On H100, expect cuBLAS to outperform the hand-written kernels in approaches
03 and 04 for all but the most carefully tuned custom implementations.
