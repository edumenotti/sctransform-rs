#!/usr/bin/env bash
set -euxo pipefail

# Build the abi3 wheel with maturin, then install it into the host prefix.
maturin build --release --interpreter "$PYTHON" --out dist
"$PYTHON" -m pip install --no-deps --no-build-isolation --find-links dist sctransform-rs
