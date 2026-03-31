# WGMMA Optimization Limitations

### Why FP32 / TF32 Cannot Simply Replace BF16 in WGMMA

You cannot use FP32/TF32 as a simple drop-in replacement for the `wgmma` optimization for several architectural and structural reasons:

#### 1. WGMMA Instruction Tile Shapes
The WGMMA instruction for Tensor Cores is rigidly defined by the precision of the operands. 
* The code uses `wgmma.mma_async...m64n64k16` for **BF16**. This means it computes a 64×64×16 tile per instruction.
* Because TF32 input elements are 32 bits (4 bytes) rather than 16 bits (2 bytes), the hardware cannot process as many elements at once in the `K` dimension. The equivalent TF32 instruction is `m64n64k8`.
* To use TF32, you would need to completely rewrite the `K` dimension loops, shared memory tile sizes, and descriptors to step by 8 instead of 16.

#### 2. Memory Size and Bandwidth
TF32 uses the mathematical precision of a 19-bit format, but **it is stored in memory as a standard 32-bit `float`** (4 bytes per element). BF16 and FP16 take up only 2 bytes per element.
* **Shared Memory Limit:** Upgrading the pipeline to 32-bit floats doubles the shared memory footprint. The current `STAGES=3` pipeline uses ~98 KB of shared memory. Switching to FP32/TF32 would consume ~196 KB. While the H100 can technically handle up to 228 KB of shared memory per block, using 196 KB severely limits the number of active thread blocks per Streaming Multiprocessor (SM), destroying occupancy.
* **Memory Bandwidth:** You would be moving twice as much data from Global Memory via `cp.async`. This halves your arithmetic intensity, causing the benchmark to become severely memory-bandwidth bound rather than compute-bound.

#### 3. No True FP32 Tensor Cores
NVIDIA Tensor Cores do not execute dense, full-precision FP32 matrix multiplications natively. 
If an application rigidly requires standard IEEE FP32 precision (where you cannot afford the precision loss of truncating the mantissa to 10 bits like TF32 does), you cannot use Tensor Cores at all. You would have to fall back to standard CUDA cores (FMA instructions), which strips away the massive throughput benefits of WGMMA completely.
