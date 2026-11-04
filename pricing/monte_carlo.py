"""Monte Carlo pricing under geometric Brownian motion, with confidence intervals
and bump-and-revalue Greeks using common random numbers."""
from __future__ import annotations

import numpy as np


def terminal_gbm(S, T, r, sigma, q, z):
    """Exact simulation of S_T given standard normal draws z."""
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def mc_price(S, K, T, r, sigma, q=0.0, call=True, n_paths=200_000, seed=0, antithetic=True):
    """Returns (price, standard error). Antithetic pairs are averaged before the error estimate."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2 if antithetic else n_paths)
    def payoff(zz):
        s_T = terminal_gbm(S, T, r, sigma, q, zz)
        return np.maximum(s_T - K, 0.0) if call else np.maximum(K - s_T, 0.0)
    pv = np.exp(-r * T) * (0.5 * (payoff(z) + payoff(-z)) if antithetic else payoff(z))
    return float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv)))


def mc_greeks_fd(S, K, T, r, sigma, q=0.0, call=True, n_paths=200_000, seed=0,
                 h_spot=0.01, h_vol=0.01):
    """Central finite differences with the same random numbers for every bump.
    h_spot is relative to S; h_vol is absolute (0.01 = 1 vol point)."""
    p = lambda s, v: mc_price(s, K, T, r, v, q, call, n_paths, seed)[0]
    dS = h_spot * S
    up, mid, down = p(S + dS, sigma), p(S, sigma), p(S - dS, sigma)
    return {
        "delta": (up - down) / (2 * dS),
        "gamma": (up - 2 * mid + down) / dS**2,
        "vega": (p(S, sigma + h_vol) - p(S, sigma - h_vol)) / (2 * h_vol),
    }
