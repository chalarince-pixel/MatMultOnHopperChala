// ============================================================
// Matrix multiplication using Thread Block Clusters (Hopper / sm_90)
// Target GPU : NVIDIA H100 (sm_90, Compute Capability 9.0)
// ============================================================
//
// Thread Block Clusters are a new level of the CUDA thread hierarchy
// introduced in the Hopper architecture (sm_90). A Cluster groups
// multiple Thread Blocks that are co-scheduled on the same GPC
// (Graphics Processing Cluster) and can share data through
// Distributed Shared Memory (DSM) without going to global memory.
//
// Hierarchy recap on H100:
//   Thread → Warp (32 threads)
//             → Thread Block (up to 1024 threads, 1 SM)
//               → Thread Block Cluster (up to 16 blocks, 1 GPC)  ← NEW in sm_90
//                 → Grid
//
// Key concepts:
//
//   Distributed Shared Memory (DSM):
//     Each block in a cluster can read/write the shared memory of
//     any other block in the same cluster via cluster.map_shared_rank().
//     This avoids a round-trip through L2 / global memory for inter-block
//     communication.
//
//   Cluster-wide synchronisation:
//     cluster.sync() — barrier across all blocks in the cluster
//                      (analogous to __syncthreads() within a block)
//
// Key headers:
//   #include <cooperative_groups.h>
//   namespace cg = cooperative_groups;
//
// Key API:
//   cg::cluster_group cluster = cg::this_cluster();
//   cluster.sync();                          // cluster-wide barrier
//   cluster.dim_blocks()                     // cluster dimensions
//   cluster.block_rank()                     // this block's index in the
//   cluster cluster.map_shared_rank(ptr, rank)       // pointer into another
//   block's shmem
//
// Kernel launch with cluster dimensions (requires CUDA 12+):
//   cudaLaunchConfig_t config = {};
//   cudaLaunchAttribute clusterAttr;
//   clusterAttr.id = cudaLaunchAttributeClusterDimension;
//   clusterAttr.val.clusterDim = {CLUSTER_X, CLUSTER_Y, 1};
//   config.attrs    = &clusterAttr;
//   config.numAttrs = 1;
//   config.gridDim  = dimGrid;
//   config.blockDim = dimBlock;
//   cudaLaunchKernelEx(&config, MatMulClusterKernel, d_A, d_B, d_C);
//
// Compile flag:
//   -arch=sm_90 (already set in Makefile)
//
// Expected benefit:
//   Tiles shared across a cluster of blocks reduce per-block shared
//   memory pressure and enable larger effective tile sizes for higher
//   arithmetic intensity, approaching cuBLAS-level throughput without
//   the library overhead.
//
// Reference:
//   CUDA C++ Programming Guide §K — Thread Block Clusters
//   https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters
// ============================================================
