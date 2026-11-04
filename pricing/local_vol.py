"""Dupire local volatility from an implied volatility surface, and Monte Carlo under local vol.

Dupire in total implied variance w(y, T) = sigma_imp(K, T)^2 * T, with log-moneyness
y = ln(K / F_T) (Gatheral, The Volatility Surface, eq. 1.10):

    sigma_loc^2(K, T) = (dw/dT) / [ 1 - (y / w) dw/dy
                                    + 1/4 (-1/4 - 1/w + y^2 / w^2) (dw/dy)^2
                                    + 1/2 d2w/dy2 ]

Derivatives are taken by central finite differences on a smooth parametric surface.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def toy_smile_surface(atm_vol=0.20, skew=-0.10, curvature=0.05) -> Callable:
    """A smooth equity-style smile used to test the pipeline (not calibrated to market data):
    sigma(y, T) = atm_vol + skew * y / sqrt(T) + curvature * y^2 / T,  y = ln(K / F_T).
    The skew flattens with maturity, as on equity index surfaces."""
    def sigma(y, T):
        T = np.maximum(T, 1e-4)
        return np.maximum(atm_vol + skew * y / np.sqrt(T) + curvature * y**2 / T, 0.05)
    return sigma


def dupire_local_vol(sigma_imp: Callable, y, T, dy=1e-3, dT=1e-3, floor=None, cap=None):
    """Local volatility at log-moneyness y and maturity T.

    With floor=None (default) any non-positive local variance raises: the surface is
    arbitrageable there. With a floor, invalid points are replaced by the floor and the
    returned mask says where this happened, so the repair is visible and can be reported."""
    w = lambda yy, tt: sigma_imp(yy, tt) ** 2 * tt
    w0 = w(y, T)
    w_T = (w(y, T + dT) - w(y, T - dT)) / (2 * dT)
    w_y = (w(y + dy, T) - w(y - dy, T)) / (2 * dy)
    w_yy = (w(y + dy, T) - 2 * w0 + w(y - dy, T)) / dy**2
    denom = (1 - y / w0 * w_y + 0.25 * (-0.25 - 1 / w0 + y**2 / w0**2) * w_y**2 + 0.5 * w_yy)
    var = w_T / denom
    bad = ~(var > 0) | ~(denom > 0)
    if floor is None:
        if np.any(bad):
            raise ValueError("negative local variance: the surface has calendar or butterfly arbitrage")
        return np.sqrt(var)
    vol = np.where(bad, floor, np.sqrt(np.where(bad, 1.0, var)))
    vol = np.maximum(vol, floor)
    if cap is not None:
        vol = np.minimum(vol, cap)
    return vol, bad


def mc_local_vol(S, r, q, sigma_imp: Callable, T, strikes, n_paths=100_000, n_steps=200, seed=0,
                 floor=0.05, cap=1.5):
    """Euler scheme on log S under local vol. Returns discounted call prices, standard errors,
    and the share of path-steps where the local variance had to be repaired (floor)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.full(n_paths, np.log(S))
    n_repaired = 0
    for i in range(n_steps):
        t = (i + 0.5) * dt  # mid-step time avoids evaluating the surface at T = 0
        F_t = S * np.exp((r - q) * t)
        y = x - np.log(F_t)
        sig, bad = dupire_local_vol(sigma_imp, y, t, floor=floor, cap=cap)
        n_repaired += int(bad.sum())
        z = rng.standard_normal(n_paths)
        x += (r - q - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * z
    s_T = np.exp(x)
    disc = np.exp(-r * T)
    prices, errors = [], []
    for K in strikes:
        pv = disc * np.maximum(s_T - K, 0.0)
        prices.append(pv.mean())
        errors.append(pv.std(ddof=1) / np.sqrt(n_paths))
    return np.array(prices), np.array(errors), n_repaired / (n_paths * n_steps)
