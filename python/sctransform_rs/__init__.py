"""sctransform-rs: Rust implementation of SCTransform for scRNA-seq."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sctransform_rs._core import __version__, add, compute_residuals as _compute_residuals_rs

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["__version__", "add", "compute_residuals"]


def compute_residuals(
    umi: "npt.ArrayLike",
    theta: "npt.ArrayLike",
    beta0: "npt.ArrayLike",
    beta1: "npt.ArrayLike",
    total_umi: "npt.ArrayLike | None" = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Compute Pearson residuals for a dense genes × cells UMI matrix.

    Parameters
    ----------
    umi
        Dense UMI matrix, shape ``(n_genes, n_cells)``. Will be cast to
        ``float64`` contiguous.
    theta, beta0, beta1
        Per-gene NB GLM parameters, each shape ``(n_genes,)``.
    total_umi
        Per-cell total UMI, shape ``(n_cells,)``. If ``None``, computed as
        ``umi.sum(axis=0)``.
    clip
        If True (default), clip residuals to ``±sqrt(n_cells / 30)``.

    Returns
    -------
    np.ndarray
        Dense Pearson residuals, shape ``(n_genes, n_cells)``, ``float64``.
    """
    umi_arr = np.ascontiguousarray(umi, dtype=np.float64)
    if umi_arr.ndim != 2:
        raise ValueError(f"umi must be 2-D, got shape {umi_arr.shape}")

    theta_arr = np.ascontiguousarray(theta, dtype=np.float64).ravel()
    beta0_arr = np.ascontiguousarray(beta0, dtype=np.float64).ravel()
    beta1_arr = np.ascontiguousarray(beta1, dtype=np.float64).ravel()

    if total_umi is None:
        total_arr = umi_arr.sum(axis=0)
    else:
        total_arr = np.ascontiguousarray(total_umi, dtype=np.float64).ravel()

    return _compute_residuals_rs(
        umi_arr, theta_arr, beta0_arr, beta1_arr, total_arr, clip
    )
