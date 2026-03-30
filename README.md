# CUDA Assignment

**GPU**: NVIDIA H100 80GB HBM3 | **CUDA**: 12.4 | **Compute Capability**: sm_90

## Project Structure
```
cuda-assignment/
├── src/
│   ├── 01_tiled_shmem.cu          # Tiled shared-memory matrix multiplication
│   ├── 02_cublas.cu               # cuBLAS approach (Tensor Cores via library)
│   ├── 03_wmma.cu                 # WMMA intrinsics approach (manual Tensor Cores)
│   └── 04_thread_block_clusters.cu # Thread Block Clusters + Distributed Shared Mem (sm_90)
├── include/utils.cuh       # Error checking, timer, device info helpers
├── results/                # Profiling outputs (nsys, ncu)
└── Makefile                # Builds each .cu → its own binary in build/
```

Each `.cu` file is compiled into a separate binary inside `build/`.
Adding a new approach is as simple as dropping a new `.cu` file in `src/`.

## Commands

| Command | Description |
|---------|-------------|
| `make` | Release build of **all** targets |
| `make 01_tiled_shmem` | Build only that target |
| `make run_01_tiled_shmem` | Build + run that target |
| `make debug` | Debug build of all targets (for cuda-gdb) |
| `make sanitize` | Run memory + race checker on all targets |
| `make profile` | Profile all targets with Nsight Systems |
| `make clean` | Remove build artifacts |

## Running
```bash
# Build and run all
make
./build/01_tiled_shmem

# Or in one step
make run_01_tiled_shmem
```

## Debugging with cuda-gdb
```bash
make debug
cuda-gdb ./build/01_tiled_shmem_debug
# Inside gdb:
(cuda-gdb) break MatMulKernel   # breakpoint in kernel
(cuda-gdb) run
(cuda-gdb) cuda thread (0,0,0)  # inspect thread
(cuda-gdb) print row
```

## Memory Checking
```bash
make sanitize
```

---

## Code Explanation — `src/main.cu`

Implements **matrix multiplication C = A × B** on the H100 GPU using a classic **tiled / shared-memory** algorithm, then measures performance in GFlop/s.

### 1. Data Structure — `Matrix`

```c
typedef struct { int width, height, stride; float *elements; } Matrix;
```

A flat 1D array (`elements`) represents a 2D matrix in **row-major order**. `stride` is the number of floats per row (equals `width` for a dense matrix), allowing it to also describe a sub-matrix window into a larger one without copying data.

### 2. Device Helper Functions (run on GPU)

| Function | What it does |
|---|---|
| `GetElement(A, row, col)` | Reads one element: `A.elements[row*stride + col]` |
| `SetElement(A, row, col, v)` | Writes one element |
| `GetSubMatrix(A, row, col)` | Returns a lightweight `Matrix` struct pointing into a 16×16 tile of `A` — no data copied, just pointer arithmetic |

### 3. The Kernel — `MatMulKernel`

**Thread organization:**
- The output matrix C is divided into **16×16 tiles**
- Each **thread block** (16×16 = 256 threads) computes exactly **one tile of C**
- Each **individual thread** computes exactly **one element** of that tile

**The tiling loop:**

```
For each column-tile of A / row-tile of B needed:
  1. All 256 threads cooperatively load a 16×16 tile of A → shared memory As[][]
  2. All 256 threads cooperatively load a 16×16 tile of B → shared memory Bs[][]
  3. __syncthreads() — wait for everyone to finish loading
  4. Each thread multiplies its row of As against its column of Bs (dot product, 16 elements)
  5. __syncthreads() — wait before overwriting shared memory in the next iteration
```

> **Why this is fast:** Global GPU memory is ~900 GB/s; shared memory is ~30 TB/s (>30× faster).
> By loading each tile once and reusing it 16 times, the kernel dramatically cuts expensive global memory reads.

### 4. Host Driver — `MatMul`

Orchestrates the full GPU lifecycle:

```
CPU memory → [cudaMemcpy H→D] → GPU memory
                                     ↓
                              MatMulKernel<<<grid, block>>>
                                     ↓
CPU memory ← [cudaMemcpy D→H] ← GPU memory
```

Every CUDA call is wrapped in `CUDA_CHECK(...)` — if anything fails (out of memory, kernel crash, etc.), it prints the exact file/line and exits instead of silently producing wrong results.

### 5. `main` — Puts It All Together

```
1. printDeviceInfo()           → confirms GPU is H100 (sm_90)
2. initMatrix()                → fill 2048×2048 A and B with random floats [0–9]
3. timerStart / MatMul / timerStop  → timed GPU run using CUDA events
4. Compute GFlop/s             → 2 × N³ FLOPs ÷ elapsed seconds
5. If N ≤ 512: cpuMatMul() + compare → correctness check (max element-wise error)
   If N > 512: skip CPU reference (2048³ operations would take minutes)
6. freeMatrix()                → clean up host memory
```

### Expected output on H100

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

> The H100's theoretical FP32 peak is ~67 TFlop/s. This basic tiled implementation achieves a fraction of that.
> A natural next step is upgrading to **Tensor Cores** via cuBLAS or WMMA intrinsics to approach peak throughput.
