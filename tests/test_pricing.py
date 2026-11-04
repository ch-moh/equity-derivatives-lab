import numpy as np

from pricing import basket, bs, correlation, local_vol, monte_carlo as mc, trees

S, K, T, r, q, SIG = 100.0, 100.0, 1.0, 0.03, 0.01, 0.20


def test_put_call_parity():
    c = bs.price(S, K, T, r, SIG, q, call=True)
    p = bs.price(S, K, T, r, SIG, q, call=False)
    assert abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-12


def test_no_arbitrage_bounds():
    for k in (50, 100, 150):
        c = bs.price(S, k, T, r, SIG, q)
        assert max(S * np.exp(-q * T) - k * np.exp(-r * T), 0) <= c <= S * np.exp(-q * T)


def test_crr_converges_to_bs():
    exact = bs.price(S, K, T, r, SIG, q)
    assert abs(trees.crr_price(S, K, T, r, SIG, q, n_steps=2000) - exact) < 2e-3


def test_american_put_worth_at_least_european():
    eu = trees.crr_price(S, K, T, r, SIG, q, n_steps=500, call=False)
    am = trees.crr_price(S, K, T, r, SIG, q, n_steps=500, call=False, american=True)
    assert am >= eu


def test_mc_confidence_interval_contains_bs():
    est, se = mc.mc_price(S, K, T, r, SIG, q, n_paths=400_000, seed=1)
    assert abs(est - bs.price(S, K, T, r, SIG, q)) < 3 * se


def test_fd_greeks_match_analytic():
    a = bs.greeks(S, K, T, r, SIG, q)
    fd = mc.mc_greeks_fd(S, K, T, r, SIG, q, n_paths=400_000, seed=2)
    assert abs(fd["delta"] - a["delta"]) < 5e-3
    assert abs(fd["vega"] - a["vega"]) / a["vega"] < 1e-2


def test_implied_vol_round_trip():
    for k in (80, 100, 120):
        p = bs.price(S, k, T, r, 0.27, q)
        assert abs(bs.implied_vol(p, S, k, T, r, q) - 0.27) < 1e-8


def test_nearest_correlation_is_admissible_and_minimal_change():
    A = np.array([[1, 0.9, 0.7], [0.9, 1, -0.4], [0.7, -0.4, 1]])
    assert not correlation.is_admissible(A)
    X, _ = correlation.nearest_correlation(A)
    assert correlation.is_admissible(X, tol=1e-8)
    # an already valid matrix must be left (almost) unchanged
    B = np.array([[1, 0.3], [0.3, 1]])
    Y, _ = correlation.nearest_correlation(B)
    assert np.allclose(Y, B, atol=1e-10)


def test_basket_with_one_asset_equals_bs():
    est, se = basket.basket_call_mc([S], [1.0], K, T, r, [SIG], np.eye(1), q=[q], n_paths=400_000, seed=3)
    assert abs(est - bs.price(S, K, T, r, SIG, q)) < 3 * se


def test_flat_surface_gives_flat_local_vol():
    flat = lambda y, t: np.full_like(np.asarray(y, dtype=float), 0.25)
    lv = local_vol.dupire_local_vol(flat, np.linspace(-0.5, 0.5, 11), 1.0)
    assert np.allclose(lv, 0.25, atol=1e-6)


def test_local_vol_mc_reprices_vanillas():
    sig = local_vol.toy_smile_surface()
    F = S * np.exp((r - q) * T)
    strikes = np.array([80.0, 100.0, 120.0])
    prices, errors, _ = local_vol.mc_local_vol(S, r, q, sig, T, strikes, n_paths=100_000, n_steps=200, seed=4)
    target = np.array([bs.price(S, k, T, r, sig(np.log(k / F), T), q) for k in strikes])
    assert np.all(np.abs(prices - target) < 4 * errors + 0.02)
