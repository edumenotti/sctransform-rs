"""AnnData integration for sctransform-rs.

Provides :func:`vst_anndata` which accepts an AnnData object with raw UMI
counts and writes Pearson residuals back into the object, following the
same convention as scanpy / scvi-tools workflows.

Requires the ``anndata`` extra: ``pip install sctransform-rs[anndata]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from sctransform_rs import vst

if TYPE_CHECKING:
    import anndata as ad


def vst_anndata(
    adata: "ad.AnnData",
    *,
    layer: str | None = None,
    clip: bool = True,
    max_iter: int = 50,
    inplace: bool = True,
) -> "ad.AnnData | None":
    """Run SCTransform v2 on an AnnData object.

    Parameters
    ----------
    adata
        AnnData with raw integer UMI counts. Genes in rows (``adata.X``
        is cells x genes by AnnData convention; this function transposes
        internally).
    layer
        Layer to read counts from. If ``None``, uses ``adata.X``.
    clip
        Clip residuals to ``+-sqrt(n_cells / 30)``.
    max_iter
        Newton-Raphson iteration cap for theta fitting.
    inplace
        If True, writes residuals into ``adata.layers["sct_residuals"]``
        and model parameters into ``adata.var``. Returns ``None``.
        If False, returns a new AnnData with residuals as ``.X``.

    Returns
    -------
    None if ``inplace=True``, otherwise a new AnnData.
    """
    import anndata as ad

    X = adata.layers[layer] if layer is not None else adata.X

    # AnnData is cells × genes; our Rust kernel expects genes × cells.
    if sp.issparse(X):
        umi = np.ascontiguousarray(X.toarray().T, dtype=np.float64)
    else:
        umi = np.ascontiguousarray(np.asarray(X).T, dtype=np.float64)

    result = vst(umi, clip=clip, max_iter=max_iter)

    # Transpose residuals back to cells × genes.
    residuals_cg = result["residuals"].T

    if inplace:
        adata.layers["sct_residuals"] = residuals_cg
        adata.var["sct_theta"] = result["theta"]
        adata.var["sct_beta0"] = result["beta0"]
        adata.var["sct_beta1"] = result["beta1"]
        return None

    adata_out = ad.AnnData(
        X=residuals_cg,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata_out.var["sct_theta"] = result["theta"]
    adata_out.var["sct_beta0"] = result["beta0"]
    adata_out.var["sct_beta1"] = result["beta1"]
    return adata_out
