"""Reproduce every figure and number of the README.

    python -m examples.report
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pricing import basket, bs, correlation, local_vol, monte_carlo as mc, trees

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
S, K, T, r, q, SIG = 100.0, 100.0, 1.0, 0.03, 0.01, 0.20
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
results = {}

# 1. Binomial tree convergence
exact = bs.price(S, K, T, r, SIG, q)
ns = np.arange(10, 401, 5)
err = [trees.crr_price(S, K, T, r, SIG, q, n_steps=n) - exact for n in ns]
plt.figure(figsize=(6, 3.4))
plt.plot(ns, err, lw=1)
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("number of steps"); plt.ylabel("CRR price - Black-Scholes")
plt.title("CRR tree: oscillating convergence to Black-Scholes")
plt.tight_layout(); plt.savefig(OUT / "crr_convergence.png"); plt.close()
results["bs_call"] = exact
results["crr_error_n400"] = err[-1]

# 2. Monte Carlo convergence, plain vs antithetic
paths = np.unique(np.logspace(3, 6, 13).astype(int))
se_plain = [mc.mc_price(S, K, T, r, SIG, q, n_paths=n, seed=0, antithetic=False)[1] for n in paths]
se_anti = [mc.mc_price(S, K, T, r, SIG, q, n_paths=n, seed=0, antithetic=True)[1] for n in paths]
plt.figure(figsize=(6, 3.4))
plt.loglog(paths, se_plain, "o-", ms=3, label="plain")
plt.loglog(paths, se_anti, "s-", ms=3, label="antithetic")
plt.loglog(paths, se_plain[0] * np.sqrt(paths[0] / paths), "k--", lw=0.8, label=r"$1/\sqrt{N}$")
plt.xlabel("number of paths"); plt.ylabel("standard error"); plt.legend()
plt.title("Monte Carlo: error decreases as 1/sqrt(N)")
plt.tight_layout(); plt.savefig(OUT / "mc_convergence.png"); plt.close()
results["mc_se_ratio_antithetic"] = se_plain[-1] / se_anti[-1]

# 3. Greeks: analytic vs finite differences across spot
spots = np.linspace(70, 130, 25)
a_delta = [bs.greeks(s, K, T, r, SIG, q)["delta"] for s in spots]
a_gamma = [bs.greeks(s, K, T, r, SIG, q)["gamma"] for s in spots]
fd = [mc.mc_greeks_fd(s, K, T, r, SIG, q, n_paths=200_000, seed=1) for s in spots]
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].plot(spots, a_delta, label="analytic"); ax[0].plot(spots, [f["delta"] for f in fd], ".", label="MC finite diff.")
ax[0].set_title("Delta"); ax[0].set_xlabel("spot"); ax[0].legend()
ax[1].plot(spots, a_gamma, label="analytic"); ax[1].plot(spots, [f["gamma"] for f in fd], ".", label="MC finite diff.")
ax[1].set_title("Gamma"); ax[1].set_xlabel("spot")
plt.tight_layout(); plt.savefig(OUT / "greeks.png"); plt.close()
results["max_delta_err"] = float(np.max(np.abs(np.array(a_delta) - [f["delta"] for f in fd])))

# 4. Nearest correlation matrix and basket option
A = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]])
X, iters = correlation.nearest_correlation(A)
results["corr_min_eig_before"] = float(np.linalg.eigvalsh(A).min())
results["corr_min_eig_after"] = float(np.linalg.eigvalsh(X).min())
results["corr_frobenius_change"] = float(np.linalg.norm(X - A))
results["corr_iterations"] = iters
results["nearest_corr"] = X
results["basket_price"] = basket.basket_call_mc([100, 100, 100], [1 / 3] * 3, 100, T, r, [0.2, 0.25, 0.3], X)

# 5. Local volatility: surface and repricing of vanillas
sig = local_vol.toy_smile_surface()
F = S * np.exp((r - q) * T)
ys = np.linspace(-0.4, 0.3, 50)
fig, ax = plt.subplots(figsize=(6, 3.4))
for tt in (0.25, 1.0, 2.0):
    Ft = S * np.exp((r - q) * tt)
    ax.plot(ys, sig(ys, tt), "--", label=f"implied, T={tt}")
    ax.plot(ys, local_vol.dupire_local_vol(sig, ys, tt), label=f"local, T={tt}")
ax.set_xlabel("log-moneyness ln(K/F)"); ax.set_ylabel("volatility"); ax.legend(fontsize=7)
ax.set_title("Dupire local vol: a steeper skew than implied vol")
plt.tight_layout(); plt.savefig(OUT / "local_vol.png"); plt.close()

strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
prices, errors, repaired = local_vol.mc_local_vol(S, r, q, sig, T, strikes, n_paths=200_000, n_steps=250, seed=5)
iv_target = sig(np.log(strikes / F), T)
iv_mc = np.array([bs.implied_vol(p, S, k, T, r, q) for p, k in zip(prices, strikes)])
results["lv_iv_error_bp"] = dict(zip(strikes.astype(int), np.round((iv_mc - iv_target) * 1e4, 1)))
results["lv_repaired_share"] = repaired

vegas = np.array([bs.greeks(S, k, T, r, v, q)["vega"] for k, v in zip(strikes, iv_target)])
results["lv_mc_se_bp"] = dict(zip(strikes.astype(int), np.round(errors / vegas * 1e4, 1)))

for k, v in results.items():
    print(f"{k:28s} {v}")
