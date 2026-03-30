# CUDA Assignment — Matrix Multiplication on H100

**GPU**: NVIDIA H100 80GB HBM3 | **CUDA**: 12.4 | **Compute Capability**: sm_90

Four implementations of `C = A × B`, progressing from a pedagogical baseline
to Hopper-native Tensor Core techniques.

---

## Approaches & Performance Ranking

| # | Folder | Technique | Est. Performance |
|---|--------|-----------|-----------------|
| 🥇 1 | [`02_cublas/`](02_cublas/README.md) | cuBLAS library | ~800–900 TFlop/s |
| 🥈 2 | [`04_thread_block_clusters/`](04_thread_block_clusters/README.md) | Thread Block Clusters + DSM | ~500–700 TFlop/s |
| 🥉 3 | [`03_wmma/`](03_wmma/README.md) | WMMA Tensor Core intrinsics | ~200–500 TFlop/s |
| 4 | [`01_tiled_shmem/`](01_tiled_shmem/README.md) | Tiled shared-memory (baseline) | ~5–15 TFlop/s |

Each folder contains the `.cu` source and its own `README.md` with a detailed explanation.

---

## Project Structure

```
cuda-assignment/
├── 01_tiled_shmem/
│   ├── 01_tiled_shmem.cu          # Tiled shared-memory baseline
│   └── README.md
├── 02_cublas/
│   ├── 02_cublas.cu               # cuBLAS (Tensor Cores via library)
│   └── README.md
├── 03_wmma/
│   ├── 03_wmma.cu                 # WMMA intrinsics (manual Tensor Cores)
│   └── README.md
├── 04_thread_block_clusters/
│   ├── 04_thread_block_clusters.cu # Thread Block Clusters + DSM (sm_90)
│   └── README.md
├── include/
│   └── utils.cuh                  # Shared: CUDA_CHECK, GpuTimer, printDeviceInfo
├── results/                       # Profiling outputs (nsys, ncu)
└── Makefile                       # Builds each .cu → its own binary in build/
```

Adding a new approach: drop a new folder + `.cu` file — the Makefile picks it up automatically.

---

## Build Commands

| Command | Description |
|---------|-------------|
| `make` | Release build of **all** targets |
| `make 01_tiled_shmem` | Build one specific target |
| `make run_01_tiled_shmem` | Build + run one target |
| `make debug` | Debug build of all targets (for cuda-gdb) |
| `make sanitize` | Memory + race checker on all targets |
| `make profile` | Nsight Systems profile of all targets |
| `make clean` | Remove `build/` artifacts |

```bash
# Build everything
make

# Build and run a specific approach
make run_01_tiled_shmem
make run_02_cublas
make run_03_wmma
make run_04_thread_block_clusters
```

---

## Debugging with cuda-gdb

```bash
make debug
cuda-gdb ./build/01_tiled_shmem_debug
(cuda-gdb) break MatMulKernel
(cuda-gdb) run
(cuda-gdb) cuda thread (0,0,0)
(cuda-gdb) print row
```

## Memory Checking

```bash
make sanitize
```
