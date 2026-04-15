"""Micro-benchmark: sctransform-rs Pearson residuals vs a numpy reference.

This benchmark targets the residual-computation step in isolation, not the
full vst() pipeline. At v0.2.0 the GLM fitting and regularization steps are
not yet implemented, so we cannot benchmark against pySCTransform end-to-end.
We compare against the pure-numpy reference (broadcast computation), which is
already a well-optimized baseline because numpy dispatches to BLAS/SIMD for
the elementwise ops.

Run:
    pixi run -e dev python benchmarks/run_benchmarks.py

The benchmark writes results to benchmarks/results/residuals_benchmark.csv.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np

import sctransform_rs as st

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GRID = [
    # (n_genes, n_cells, label)
    (2000, 3000, "PBMC-3k-HVG"),
    (5000, 10000, "PBMC-10k-HVG"),
    (2000, 30000, "30k-wide"),
]

N_WARMUP = 1
N_REPEATS = 5


def numpy_reference(umi, theta, beta0, beta1, total_umi, clip=True):
    log10_total = np.log10(total_umi)[np.newaxis, :]
    lin = beta0[:, np.newaxis] + beta1[:, np.newaxis] * log10_total
    mu = np.exp(lin)
    var = mu + (mu ** 2) / theta[:, np.newaxis]
    z = (umi - mu) / np.sqrt(var)
    if clip:
        k = np.sqrt(umi.shape[1] / 30.0)
        np.clip(z, -k, k, out=z)
    return z


def synth(n_genes, n_cells, seed=0):
    rng = np.random.default_rng(seed)
    total = rng.integers(500, 50_000, size=n_cells).astype(np.float64)
    theta = rng.uniform(0.5, 50.0, size=n_genes)
    beta0 = rng.uniform(-12.0, -3.0, size=n_genes)
    beta1 = np.ones(n_genes)
    mu = np.exp(beta0[:, None] + beta1[:, None] * np.log10(total)[None, :])
    p = theta[:, None] / (theta[:, None] + mu)
    umi = rng.negative_binomial(theta[:, None], p).astype(np.float64)
    return umi, theta, beta0, beta1, total


def _time(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def bench_one(n_genes, n_cells, label):
    umi, theta, beta0, beta1, total = synth(n_genes, n_cells)

    # Warmup
    for _ in range(N_WARMUP):
        st.compute_residuals(umi, theta, beta0, beta1, total)
        numpy_reference(umi, theta, beta0, beta1, total)

    rust_times = []
    for _ in range(N_REPEATS):
        dt, _ = _time(st.compute_residuals, umi, theta, beta0, beta1, total)
        rust_times.append(dt)

    numpy_times = []
    for _ in range(N_REPEATS):
        dt, _ = _time(numpy_reference, umi, theta, beta0, beta1, total)
        numpy_times.append(dt)

    rust_med = statistics.median(rust_times)
    numpy_med = statistics.median(numpy_times)
    speedup = numpy_med / rust_med

    return {
        "label": label,
        "n_genes": n_genes,
        "n_cells": n_cells,
        "rust_sec_median": rust_med,
        "rust_sec_min": min(rust_times),
        "numpy_sec_median": numpy_med,
        "numpy_sec_min": min(numpy_times),
        "speedup_median": speedup,
    }


def main():
    rows = []
    print(f"{'label':<16} {'n_genes':>8} {'n_cells':>8} "
          f"{'rust (s)':>10} {'numpy (s)':>10} {'speedup':>8}")
    print("-" * 66)
    for n_genes, n_cells, label in GRID:
        r = bench_one(n_genes, n_cells, label)
        rows.append(r)
        print(f"{r['label']:<16} {r['n_genes']:>8} {r['n_cells']:>8} "
              f"{r['rust_sec_median']:>10.4f} {r['numpy_sec_median']:>10.4f} "
              f"{r['speedup_median']:>7.2f}x")

    out_path = RESULTS_DIR / "residuals_benchmark.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
