"""sctransform-rs: Rust implementation of SCTransform for scRNA-seq."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sctransform_rs._core import (
    __version__,
    add,
    compute_residuals as _compute_residuals_rs,
    fit_glm_offset as _fit_glm_offset_rs,
)

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["__version__", "add", "compute_residuals", "fit_glm_offset", "vst"]


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


def fit_glm_offset(
    umi: "npt.ArrayLike",
    total_umi: "npt.ArrayLike | None" = None,
    *,
    max_iter: int = 50,
) -> dict:
    """Fit the SCTransform v2 offset-model NB GLM for every gene.

    The offset model fixes the log-total-UMI slope: beta1 = ln(10) on a
    log10(total_umi) covariate. Only beta0 and theta are estimated per gene.

    Parameters
    ----------
    umi
        Dense UMI matrix, shape ``(n_genes, n_cells)``.
    total_umi
        Per-cell total UMI, shape ``(n_cells,)``. If ``None``, computed as
        ``umi.sum(axis=0)``.
    max_iter
        Newton-Raphson iteration cap for theta fitting.

    Returns
    -------
    dict with keys ``theta``, ``beta0``, ``beta1`` — each a ``(n_genes,)``
    float64 array. ``beta1`` is a constant vector of ``ln(10)``.
    """
    umi_arr = np.ascontiguousarray(umi, dtype=np.float64)
    if umi_arr.ndim != 2:
        raise ValueError(f"umi must be 2-D, got shape {umi_arr.shape}")

    if total_umi is None:
        total_arr = umi_arr.sum(axis=0)
    else:
        total_arr = np.ascontiguousarray(total_umi, dtype=np.float64).ravel()

    theta, beta0 = _fit_glm_offset_rs(umi_arr, total_arr, max_iter)
    beta1 = np.full(umi_arr.shape[0], np.log(10.0), dtype=np.float64)
    return {"theta": theta, "beta0": beta0, "beta1": beta1}


def vst(
    umi: "npt.ArrayLike",
    total_umi: "npt.ArrayLike | None" = None,
    *,
    clip: bool = True,
    max_iter: int = 50,
) -> dict:
    """Run the full SCTransform v2 variance-stabilising transform.

    This is the main entry point: fit the NB GLM offset model per gene,
    then compute clipped Pearson residuals.

    Parameters
    ----------
    umi
        Dense UMI count matrix, shape ``(n_genes, n_cells)``.
        Integer counts should be passed as float64 (or will be cast).
    total_umi
        Per-cell total UMI, shape ``(n_cells,)``. If ``None``, computed
        as ``umi.sum(axis=0)``.
    clip
        If True (default), clip residuals to ``+-sqrt(n_cells / 30)``.
    max_iter
        Newton-Raphson iteration cap for theta fitting.

    Returns
    -------
    dict with keys:

    - ``residuals`` : np.ndarray, shape ``(n_genes, n_cells)``
    - ``theta``     : np.ndarray, shape ``(n_genes,)``
    - ``beta0``     : np.ndarray, shape ``(n_genes,)``
    - ``beta1``     : np.ndarray, shape ``(n_genes,)``
    """
    umi_arr = np.ascontiguousarray(umi, dtype=np.float64)
    if umi_arr.ndim != 2:
        raise ValueError(f"umi must be 2-D, got shape {umi_arr.shape}")

    if total_umi is None:
        total_arr = umi_arr.sum(axis=0)
    else:
        total_arr = np.ascontiguousarray(total_umi, dtype=np.float64).ravel()

    theta, beta0 = _fit_glm_offset_rs(umi_arr, total_arr, max_iter)
    beta1 = np.full(umi_arr.shape[0], np.log(10.0), dtype=np.float64)

    residuals = _compute_residuals_rs(
        umi_arr, theta, beta0, beta1, total_arr, clip
    )

    return {
        "residuals": residuals,
        "theta": theta,
        "beta0": beta0,
        "beta1": beta1,
    }
