"""Black-Scholes pricing, analytic Greeks and implied volatility.

Conventions: continuous risk-free rate r, continuous dividend yield q,
maturity T in years, volatility sigma (annualised). Prices are per unit of underlying.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, sigma):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def price(S, K, T, r, sigma, q=0.0, call=True):
    """European option price under Black-Scholes."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if call:
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def greeks(S, K, T, r, sigma, q=0.0, call=True) -> dict:
    """Analytic delta, gamma, vega (per 1.00 of vol), theta (per year), rho."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf = norm.pdf(d1)
    gamma = df_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf * np.sqrt(T)
    if call:
        delta = df_q * norm.cdf(d1)
        theta = (-S * df_q * pdf * sigma / (2 * np.sqrt(T))
                 - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1))
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = (-S * df_q * pdf * sigma / (2 * np.sqrt(T))
                 + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1))
        rho = -K * T * df_r * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(target_price, S, K, T, r, q=0.0, call=True, lo=1e-6, hi=5.0):
    """Implied volatility by Brent's method. Raises ValueError outside no-arbitrage bounds."""
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    lower = max(S * df_q - K * df_r, 0.0) if call else max(K * df_r - S * df_q, 0.0)
    upper = S * df_q if call else K * df_r
    if not (lower - 1e-12 <= target_price <= upper + 1e-12):
        raise ValueError(f"price {target_price:.6f} outside no-arbitrage bounds [{lower:.6f}, {upper:.6f}]")
    f = lambda s: price(S, K, T, r, s, q, call) - target_price
    return brentq(f, lo, hi, xtol=1e-12, maxiter=200)
