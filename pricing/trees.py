"""Cox-Ross-Rubinstein binomial tree for European and American options."""
from __future__ import annotations

import numpy as np


def crr_price(S, K, T, r, sigma, q=0.0, n_steps=500, call=True, american=False):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0, 1): increase n_steps")
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    s_T = S * u**j * d**(n_steps - j)
    values = np.maximum(s_T - K, 0.0) if call else np.maximum(K - s_T, 0.0)
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            s = S * u**j * d**(step - j)
            exercise = np.maximum(s - K, 0.0) if call else np.maximum(K - s, 0.0)
            values = np.maximum(values, exercise)
    return float(values[0])
