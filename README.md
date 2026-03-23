# CUDA Assignment

**GPU**: NVIDIA H100 80GB HBM3 | **CUDA**: 12.4 | **Compute Capability**: sm_90

## Project Structure
```
cuda-assignment/
├── src/main.cu          # Main CUDA source (edit this)
├── include/utils.cuh    # Error checking, timer, device info helpers
├── tests/               # Put test inputs/scripts here
├── results/             # Profiling outputs
└── Makefile
```

## Commands

| Command | Description |
|---------|-------------|
| `make` | Release build (optimized) |
| `make debug` | Debug build (for cuda-gdb) |
| `make sanitize` | Run memory + race checker |
| `make profile` | Profile with Nsight Systems |
| `make clean` | Remove build artifacts |

## Running
```bash
make && ./build/cuda_assignment
```

## Debugging with cuda-gdb
```bash
make debug
cuda-gdb ./build/cuda_assignment_debug
# Inside gdb:
(cuda-gdb) break vectorAdd      # breakpoint in kernel
(cuda-gdb) run
(cuda-gdb) cuda thread (0,0,0)  # inspect thread
(cuda-gdb) print idx
```

## Memory Checking
```bash
make sanitize
```
