"""Basket call on several correlated GBM underlyings, priced by Monte Carlo."""
from __future__ import annotations

import numpy as np

from .correlation import is_admissible


def basket_call_mc(S0, weights, K, T, r, sigmas, corr, q=None, n_paths=200_000, seed=0):
    S0, w, sig = map(lambda x: np.asarray(x, dtype=float), (S0, weights, sigmas))
    q = np.zeros_like(S0) if q is None else np.asarray(q, dtype=float)
    if not is_admissible(corr):
        raise ValueError("correlation matrix is not admissible (not PSD / unit diagonal)")
    # Eigen-decomposition instead of Cholesky: works for singular PSD matrices too,
    # which is exactly what the nearest-correlation repair returns (smallest eigenvalue = 0).
    w_eig, V = np.linalg.eigh(np.asarray(corr, dtype=float))
    L = V * np.sqrt(np.clip(w_eig, 0.0, None))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, len(S0))) @ L.T
    s_T = S0 * np.exp((r - q - 0.5 * sig**2) * T + sig * np.sqrt(T) * z)
    pv = np.exp(-r * T) * np.maximum(s_T @ w - K, 0.0)
    return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(n_paths))
