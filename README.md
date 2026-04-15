# sctransform-rs

A fast Rust implementation of [SCTransform v2](https://doi.org/10.1186/s13059-021-02584-9) normalization for single-cell RNA-seq data, exposed as a Python package via [PyO3](https://pyo3.rs/).

**Goal:** drop-in replacement for [`pySCTransform`](https://github.com/saketkc/pySCTransform) with measurably better performance on datasets ≥10k cells, while producing outputs numerically equivalent to R `sctransform` v2.

## Status

**v0.1.0 — environment + skeleton.** Build system works end-to-end; algorithm not yet implemented.

### Roadmap

| Version | Milestone |
|---|---|
| v0.1.0 | Build skeleton, CI, smoke-test function |
| v0.2.0 | Pearson residual computation (pure arithmetic) |
| v0.3.0 | Per-gene Poisson/NB GLM fitting (IRLS, offset model) |
| v0.4.0 | Kernel regularization + end-to-end `vst()` |
| v0.5.0 | Scanpy/AnnData integration |
| v1.0.0 | Full v1 model, PyPI release |

## Installation

### For users (recommended)

Pre-built wheels are published to PyPI and conda-forge. Pick one:

```bash
pip install sctransform-rs
```

```bash
conda install -c conda-forge sctransform-rs
```

No Rust toolchain, no repository clone, and no `pixi` are required — the wheel
ships a compiled ABI3 extension module that works on CPython ≥ 3.10 across
Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64).

<details>
<summary><strong>Building from source (for contributors)</strong></summary>

Only needed if you are modifying the Rust backend. Requires
[pixi](https://pixi.sh) and [rustup](https://rustup.rs).

```bash
git clone https://github.com/edumenotti/sctransform-rs
cd sctransform-rs
pixi install -e dev
pixi run -e dev build
pixi run -e dev test
```

The `pixi run -e dev build` task invokes `maturin develop --release`, which
compiles the Rust crate and installs an editable Python package pointing at
`python/sctransform_rs/`. Pure-Python changes under that directory are picked
up without rebuilding; Rust changes require re-running the build task.

</details>

## Project layout

```
sctransform-rs/
├── pixi.toml             # environment manifest (pixi)
├── pyproject.toml        # Python package + maturin build config
├── Cargo.toml            # Rust crate config
├── src/lib.rs            # PyO3 module entrypoint
├── python/sctransform_rs/ # Python package (re-exports Rust)
├── tests/                # pytest suite
├── benchmarks/           # performance comparisons
└── notebooks/            # validation notebooks
```

## License

MIT
