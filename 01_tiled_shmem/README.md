# 01 — Tiled Shared-Memory Matrix Multiplication

**Approach:** Classic tiled GEMM using shared memory (CUDA Programming Guide §3.2.4)
**Performance on H100:** ~5–15 TFlop/s (FP32, CUDA cores only — no Tensor Cores)
**Rank:** 4 / 4 — pedagogical baseline

---

## What this does

Implements **C = A × B** for square N×N matrices using a tiled algorithm that
loads 16×16 tiles of A and B into on-chip shared memory before computing,
reducing expensive global-memory reads by a factor of `BLOCK_SIZE`.

---

## Code Walkthrough

### Data Structure — `Matrix`

```c
typedef struct { int width, height, stride; float *elements; } Matrix;
```

A flat 1D array (`elements`) represents a 2D matrix in **row-major order**.
`stride` is floats-per-row; it allows `GetSubMatrix` to return a window into a
larger matrix using only pointer arithmetic — no data is copied.

---

### Device Helper Functions

| Function | What it does |
|---|---|
| `GetElement(A, row, col)` | `A.elements[row * stride + col]` |
| `SetElement(A, row, col, v)` | Writes one element |
| `GetSubMatrix(A, row, col)` | Returns a `Matrix` view of the tile at block position (row, col) |

---

### The Kernel — `MatMulKernel`

**Thread organisation:**

| Level | Size | Responsibility |
|---|---|---|
| Thread | 1 | Computes 1 element of C |
| Thread Block | 16×16 = 256 | Computes one 16×16 tile of C |
| Grid | (N/16)×(N/16) | Covers all tiles of C |

**Tiling loop (per block):**

```
for m = 0 … (A.width / BLOCK_SIZE - 1):
  1. Load tile As[16][16] from A  (each thread loads 1 element)
  2. Load tile Bs[16][16] from B  (each thread loads 1 element)
  3. __syncthreads()              ← wait: all elements loaded
  4. Dot product: Cvalue += As[row][e] * Bs[e][col] for e = 0..15
  5. __syncthreads()              ← wait: done computing before next tile load
```

> **Why shared memory helps:** Global memory bandwidth ~900 GB/s;
> shared memory ~30 TB/s (>30× faster). Each tile is loaded once but
> reused 16 times across the dot product loop.

---

### Host Driver — `MatMul`

```
CPU memory ──[cudaMemcpy H→D]──▶ GPU global memory
                                        │
                           MatMulKernel<<<grid, block>>>
                                        │
CPU memory ◀──[cudaMemcpy D→H]── GPU global memory
```

Every CUDA call is wrapped in `CUDA_CHECK()` (from `include/utils.cuh`) —
failures print file + line and exit immediately.

---

### `main`

```
1. printDeviceInfo()       → confirm H100 / sm_90
2. initMatrix()            → 2048×2048 A and B, random floats [0–9]
3. GpuTimer (CUDA events)  → timed MatMul call (includes H↔D transfers)
4. GFlop/s = 2×N³ / time  → end-to-end throughput
5. CPU verify (N ≤ 512)    → element-wise max error check
```

---

### Expected Output on H100

```
=== GPU: NVIDIA H100 SXM5 80GB ===
  Compute Capability : 9.0
  Global Memory      : 80.0 GB
  SMs                : 132
  Max threads/block  : 1024
  Warp size          : 32
================================

Matrix size : 2048 x 2048
Block size  : 16 x 16
Grid size   : 128 x 128

GPU time    : ~2.5 ms
Performance : ~6800 GFlop/s

(Skipping CPU reference for N=2048 — would take too long)
```

---

## Build & Run

```bash
# From the repo root
make 01_tiled_shmem
./build/01_tiled_shmem

# Or in one step
make run_01_tiled_shmem
```

## Limitations

- Uses **FP32 CUDA cores only** — Tensor Cores are idle
- H100 FP32 CUDA-core peak is ~67 TFlop/s; this kernel reaches ~10%
- Next step: `02_cublas` for Tensor Core acceleration
