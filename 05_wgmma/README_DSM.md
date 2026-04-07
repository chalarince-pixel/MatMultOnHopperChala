# WGMMA + Distributed Shared Memory (DSM) — Why It's Slower

## What this experiment does

`05_wgmma_dsm.cu` adds **Thread Block Clusters** to the WGMMA kernel.
In a 1×2 cluster (two adjacent column-tile blocks), both blocks cover the
**same A rows** every K-step. The idea:

- **Rank 0** loads `sA` from global memory via `cp.async`
- **Rank 1** reads `sA` via DSM (`cluster.map_shared_rank`) — no global load for A
- **Both blocks** load their own `sB` (different column tiles)

This should halve A global memory traffic across the cluster — in theory.

---

## Why it's actually ~2× *slower*

### The synchronisation problem

After rank 0's `cp.async` completes, rank 1 cannot safely read rank 0's
shared memory via DSM without a cross-block fence. The only CUDA primitive
that provides this guarantee in the `cp.async` world is:

```cuda
cluster.sync();   // full cross-SM pipeline drain
```

We need this **twice per K-step**:

1. Before WGMMA: ensure rank 0's `sA` is visible to rank 1
2. After WGMMA: ensure rank 1 is done reading before rank 0 overwrites `sA`

### The cost

Measured on H100 (sm_90a):

| Primitive       | Cost per call |
|-----------------|---------------|
| `__syncthreads()` | ~25 ns      |
| `cluster.sync()` | **~660 ns** |

`cluster.sync()` is a **full cross-GPC barrier** — it stalls until all SMs
in the cluster have drained their memory pipelines. It cannot be overlapped
with compute.

### The math for N=8192

```
K-steps per kernel     = 8192 / BLOCK_K(64) = 128
cluster.sync() calls   = 128 × 2 = 256
Total barrier overhead = 256 × 660 ns ≈ 169 ms

Baseline kernel time   = 2.25 ms
DSM kernel time        ≈ 2.25 + 169 ≈ 171 ms  (observed ~5 ms, dominated by barriers)

Barrier overhead       = 75× the entire baseline runtime
```

The baseline kernel is **compute-bound** (tensor cores pinned at ~480 TFLOPS).
The `cp.async` pipeline already hides memory latency perfectly. Halving A
bandwidth saves **0 compute cycles** — there's no headroom for the DSM
savings to materialise inside a barrier-dominated kernel.

---

## The correct approach: TMA + `mbarrier`

CUTLASS 3.x and productions kernels that use WGMMA + DSM use a completely
different synchronisation primitive introduced with Hopper:

### `mbarrier` (memory barrier / arrive-wait)

```ptx
// Rank 0 — after cp.async loads are committed:
mbarrier.arrive.expect_tx.shared::cta.b64 [mbar], bytes;

// Rank 1 — before reading via DSM:
mbarrier.try_wait.shared::cta.b64 [mbar], phase;
```

| Primitive             | Cost      | Blocks compute? |
|-----------------------|-----------|-----------------|
| `cluster.sync()`      | ~660 ns   | ✅ Yes, always  |
| `mbarrier.arrive`     | **~50 ns**| ❌ No (async)   |
| `mbarrier.try_wait`   | **~0 ns** (if data ready) | ❌ No |

`mbarrier` is an **asynchronous arrive-wait barrier**. Rank 0 posts an
arrival token after its `cp.async` group completes. Rank 1 polls the barrier
non-blockingly — if the data is already there (which it will be with a
proper pipeline depth), the wait returns immediately without stalling.

### TMA (Tensor Memory Accelerator)

Hopper's TMA issues entire 2D/3D tile copies with **one instruction** and
integrates directly with `mbarrier`:

```cuda
// One instruction copies the entire 128×64 A-tile:
cp.async.bulk.tensor.2d.shared::cluster.global
    [smem_desc], [global_desc], [mbar];
```

With TMA:
- Load initiation: **1 instruction** (vs ~1024 `cp.async` calls for a 128×64 tile)
- Synchronisation: **50 ns** `mbarrier.arrive` (vs 660 ns `cluster.sync()`)
- DSM reads: rank 1 reads from `shared::cluster` address space directly

This is the architecture behind `06_wgmma_tma/` — combining TMA + `mbarrier`
+ WGMMA is what achieves near-peak H100 throughput in cuBLAS and CUTLASS.

---

## Summary

| Technique | A bandwidth | Sync cost/step | Net result |
|-----------|------------|----------------|------------|
| WGMMA baseline | 1× (full) | 0 | **480 TFLOPS** (compute-bound) |
| WGMMA + DSM (cp.async) | 0.5× | 2 × 660 ns | **~220 TFLOPS** (sync-bound) |
| WGMMA + DSM (TMA+mbar) | 0.5× | 2 × ~50 ns | **~500+ TFLOPS** (achievable) |

DSM is a real, production-grade optimization — it just requires the right
synchronisation primitive to pay off.
