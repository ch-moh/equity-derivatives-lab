"""Correlation matrices: admissibility check and Higham's nearest correlation matrix.

A correlation matrix is admissible if it is symmetric, has a unit diagonal and is
positive semi-definite (PSD). Matrices estimated pairwise or stressed by hand often
fail the PSD condition, and Cholesky-based simulation then breaks.
"""
from __future__ import annotations

import numpy as np


def is_admissible(C, tol=1e-10) -> bool:
    C = np.asarray(C, dtype=float)
    return (np.allclose(C, C.T, atol=tol) and np.allclose(np.diag(C), 1.0, atol=tol)
            and np.linalg.eigvalsh(C).min() >= -tol)


def _proj_psd(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    return (V * np.clip(w, 0.0, None)) @ V.T


def nearest_correlation(A, tol=1e-10, max_iter=1000):
    """Higham (2002) alternating projections with Dykstra's correction.
    Returns the closest admissible correlation matrix in Frobenius norm, and the number of iterations."""
    Y = np.asarray(A, dtype=float).copy()
    dS = np.zeros_like(Y)
    for it in range(1, max_iter + 1):
        R = Y - dS
        X = _proj_psd(R)
        dS = X - R
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)
        if np.linalg.norm(Y_new - Y, "fro") / max(np.linalg.norm(Y, "fro"), 1e-16) < tol:
            Y = Y_new
            break
        Y = Y_new
    Y = (Y + Y.T) / 2
    np.fill_diagonal(Y, 1.0)
    return Y, it
