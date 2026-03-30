# 04 — Thread Block Clusters (Hopper / sm_90)

**Approach:** Thread Block Clusters + Distributed Shared Memory (DSM)
**Performance on H100:** ~500–700 TFlop/s (FP16, well-tuned)
**Rank:** 2 / 4 — second only to cuBLAS when well implemented

---

## What this does

Exploits **Thread Block Clusters**, a new hierarchy level introduced in the
Hopper architecture (sm_90). Multiple thread blocks within a cluster are
co-scheduled on the same GPC (Graphics Processing Cluster) and can directly
read each other's shared memory via **Distributed Shared Memory (DSM)** —
without a round-trip through L2 or global memory.

---

## CUDA Thread Hierarchy on H100

```
Thread          (1 lane)
  └─ Warp       (32 threads)
       └─ Thread Block   (up to 1024 threads, resident on 1 SM)
            └─ Thread Block Cluster  (up to 16 blocks, on 1 GPC)  ← NEW sm_90
                 └─ Grid
```

---

## Key Concepts

### Distributed Shared Memory (DSM)

Each block in a cluster retains its own shared memory, but any block can
read or write another block's shared memory via `cluster.map_shared_rank()`.
This is faster than global memory and enables **larger effective tile sizes**
without exceeding a single block's 228 KB shared memory limit.

### Why it can outperform WMMA

- Larger tiles → higher arithmetic intensity → Tensor Cores stay busier
- DSM communication avoids L2 bandwidth pressure between cooperating blocks
- Especially effective for problem sizes where a single block's shared memory
  would otherwise be the bottleneck

---

## Key API

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(...) {
    cg::cluster_group cluster = cg::this_cluster();

    cluster.sync();                         // cluster-wide barrier
    int rank = cluster.block_rank();        // this block's index in the cluster
    auto *remote = cluster.map_shared_rank(local_ptr, other_rank); // DSM access
}
```

### Kernel Launch with Cluster Dimensions (CUDA 12+)

```cpp
cudaLaunchConfig_t config    = {};
config.gridDim               = dimGrid;
config.blockDim              = dimBlock;

cudaLaunchAttribute clusterAttr;
clusterAttr.id               = cudaLaunchAttributeClusterDimension;
clusterAttr.val.clusterDim   = {CLUSTER_X, CLUSTER_Y, 1};  // e.g. {2, 2, 1}

config.attrs    = &clusterAttr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, MatMulClusterKernel, d_A, d_B, d_C);
```

### Cluster Size Constraints on H100

- Maximum cluster size: **8 blocks** (hardware), **16 blocks** (software model)
- Must be set at launch — cannot be changed dynamically
- Cluster dimensions must evenly divide the grid dimensions

---

## Build & Run

```bash
# From the repo root
make 04_thread_block_clusters
./build/04_thread_block_clusters
```

**Requires:** CUDA 12.0+, sm_90 (already set in Makefile)

## Headers Required

```c
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "include/utils.cuh"
```

---

## Reference

- [CUDA C++ Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
