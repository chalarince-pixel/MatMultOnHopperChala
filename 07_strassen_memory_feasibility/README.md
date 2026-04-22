# 07 — Two-Level Strassen Memory Feasibility (Hopper / H100)

**Approach:** Memory-only model for hybrid Strassen scheduling (Level-1 DFS/BFS + Level-2 BFS)
**Goal:** Check feasibility from HBM and DSMEM limits before implementing compute kernels

---

## What this example does

This sample does **not** run Strassen arithmetic kernels.

It evaluates whether your execution strategy is possible from a **memory standpoint only**:

1. Global memory (HBM) footprint for two top-level policies:
   - Level-1 DFS: one parent problem active at a time
   - Level-1 BFS: seven parent problems active at once
2. Level-2 BFS footprint inside each parent (seven child multiplies)
3. Hopper cluster DSMEM feasibility for storing full child matrices
4. Practical `cudaMalloc` probe to validate modeled totals

---

## Inputs

```bash
./build/07_strassen_memory_feasibility [N] [dtype] [cluster_blocks]
```

- `N`: square matrix size (must be divisible by 4)
- `dtype`: `fp16`, `bf16`, or `fp32`
- `cluster_blocks`: assumed thread blocks in one cluster for DSMEM budget (default `7`)

Examples:

```bash
./build/07_strassen_memory_feasibility 32768 fp32 7
./build/07_strassen_memory_feasibility 65536 fp32 7
```

---

## Memory model

Let $e$ be bytes per element and $N$ be the full matrix dimension.

Persistent full matrices:

$$
\text{base} = 3N^2e \quad (A, B, C)
$$

For one parent problem $M_i$ (size $N/2$), Level-2 BFS assumptions:

- Child output materialization: $7\cdot (N/4)^2 e$
- Parent output storage: $(N/2)^2 e$

Two workspace variants are reported:

1. Optimistic:
   - child operands are streamed/reused with only two child operand buffers
2. Conservative:
   - all child operands are materialized (14 operand buffers)

Derived total coefficients:

$$
\text{DFS+child BFS (optimistic)} = \frac{61}{16}N^2e
$$

$$
\text{DFS+child BFS (conservative)} = \frac{73}{16}N^2e
$$

$$
\text{BFS+child BFS (optimistic)} = \frac{139}{16}N^2e
$$

$$
\text{BFS+child BFS (conservative)} = \frac{223}{16}N^2e
$$

The program compares each total against **current free HBM** (`cudaMemGetInfo`) and computes max feasible $N$ for each case.

---

## DSMEM check (cluster side)

The sample queries:

- max shared memory per block (opt-in)
- cluster DSMEM pool = `cluster_blocks * maxShmemPerBlockOptin`

Then checks whether full Level-2 child matrices could fit in DSMEM:

$$
\text{one child full }(A+B+C) = 3\cdot (N/4)^2 e
$$

$$
\text{seven children full }(A+B+C) = 21\cdot (N/4)^2 e
$$

This is a strict upper-bound sanity check and is typically infeasible for large $N$.

---

## Verified results on this H100 session

### Run 1: `N=32768`, `FP32`, `cluster_blocks=7`

- Free HBM: 78.66 GiB
- DFS + child BFS conservative: 18.25 GiB (fits, allocation probe `YES`)
- BFS + child BFS conservative: 55.75 GiB (fits, allocation probe `YES`)
- DSMEM one-child full storage: `NO`
- DSMEM seven-children full storage: `NO`

### Run 2: `N=65536`, `FP32`, `cluster_blocks=7`

- Free HBM: 78.66 GiB
- DFS + child BFS conservative: 73.00 GiB (fits, allocation probe `YES`)
- BFS + child BFS conservative: 223.00 GiB (does not fit, allocation probe `NO`)
- DSMEM one-child full storage: `NO`
- DSMEM seven-children full storage: `NO`

---

## Interpretation

1. Level-1 DFS keeps top-level memory bounded and remains feasible much longer.
2. Level-1 BFS can become impossible at large $N$ due to HBM explosion.
3. Full child matrix storage in cluster DSMEM is unrealistic except at very small $N$.
4. Practical strategy: keep bulk matrices in HBM, use shared/DSMEM for tiles and coordination.

---

## Build and run

From repository root:

```bash
make 07_strassen_memory_feasibility
./build/07_strassen_memory_feasibility 32768 fp32 7
./build/07_strassen_memory_feasibility 65536 fp32 7
```
