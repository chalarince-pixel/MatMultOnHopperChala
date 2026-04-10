#!/usr/bin/env python3
"""
plot_results.py — Plot WMMA vs cuBLAS runtime benchmarks.

Usage (run on your local Mac after copying the CSVs):
    python3 plot_results.py

Expects:
    wmma_out.csv    — output of ./build/03_wmma
    cublas_out.csv  — output of ./build/02_cublas

Both CSVs share the same format:
    dtype,N,ms,tflops
    BF16,32,0.0312,0.21
    ...
"""

import csv, os, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#e0e0f0",
    "xtick.color":      "#a0a0c0",
    "ytick.color":      "#a0a0c0",
    "text.color":       "#e0e0f0",
    "grid.color":       "#2a2d3e",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#1a1d2e",
    "legend.edgecolor": "#444466",
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

WMMA_CSV   = os.path.join("results", "wmma_out.csv")
CUBLAS_CSV = os.path.join("results", "cublas_out.csv")

def load_csv(path):
    """Return dict: dtype -> {N: ms}"""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dtype = row["dtype"].strip()
            N     = int(row["N"])
            ms    = float(row["ms"])
            data.setdefault(dtype, {})[N] = ms
    return data

def sorted_xy(d):
    """Return sorted (Ns, ms_list) arrays from a {N: ms} dict."""
    pairs = sorted(d.items())
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

def pct_of_cublas(wmma_ms, cublas_ms, sizes):
    """Print efficiency table."""
    print(f"  {'N':>6}  {'WMMA ms':>10}  {'cuBLAS ms':>10}  {'WMMA/cBLAS %':>14}")
    for n, wm, cm in zip(sizes, wmma_ms, cublas_ms):
        print(f"  {n:>6}  {wm:>10.4f}  {cm:>10.4f}  {cm/wm*100:>13.1f}%")

def plot_dtype(ax, dtype, wmma, cublas, color_wmma, color_cbl):
    """Plot one dtype panel."""
    wNs,  wMs  = sorted_xy(wmma.get(dtype, {}))
    cNs,  cMs  = sorted_xy(cublas.get(dtype, {}))

    ax.loglog(wNs, wMs, "o-",  color=color_wmma, lw=2.2, ms=7,
              label="WMMA (ours)", zorder=3)
    ax.loglog(cNs, cMs, "s--", color=color_cbl,  lw=2.2, ms=7,
              label="cuBLAS",      zorder=3)

    ax.set_title(f"{dtype} — Runtime vs Matrix Dimension",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Matrix dimension N")
    ax.set_ylabel("Runtime (ms)")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.3g}"))
    ax.set_xticks(wNs if len(wNs) else cNs)
    ax.grid(True, which="both")
    ax.legend(fontsize=11)

    # Annotate efficiency at each N
    if len(wNs) and len(cNs):
        common = set(wNs) & set(cNs)
        for n in sorted(common):
            wm = wmma[dtype][n]
            cm = cublas[dtype][n]
            pct = cm / wm * 100
            ax.annotate(f"{pct:.0f}%",
                        xy=(n, wm), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8,
                        color="#aaaacc")

    return wNs, wMs, cNs, cMs


def main():
    for p in [WMMA_CSV, CUBLAS_CSV]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}", file=sys.stderr)
            print("Run the benchmarks first:\n"
                  "  ./build/03_wmma   > wmma_out.csv\n"
                  "  ./build/02_cublas > cublas_out.csv",
                  file=sys.stderr)
            sys.exit(1)

    wmma   = load_csv(WMMA_CSV)
    cublas = load_csv(CUBLAS_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("WMMA Tensor Core vs cuBLAS — H100 GPU\n"
                 "(lower is faster; annotations = WMMA as % of cuBLAS speed)",
                 fontsize=13, y=1.02)

    WMMA_COLOR   = "#7c83ff"
    CUBLAS_COLOR = "#ff7c7c"

    for dtype, ax in zip(["BF16", "FP32"], axes):
        wNs, wMs, cNs, cMs = plot_dtype(ax, dtype, wmma, cublas,
                                          WMMA_COLOR, CUBLAS_COLOR)
        print(f"\n── {dtype} ──")
        if len(wNs) and len(cNs):
            common = sorted(set(wNs) & set(cNs))
            wMs_c  = [wmma[dtype][n] for n in common]
            cMs_c  = [cublas[dtype][n] for n in common]
            pct_of_cublas(wMs_c, cMs_c, common)

    plt.tight_layout()
    out = os.path.join("results", "benchmark_plot.png")
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
