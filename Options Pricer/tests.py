"""
Unit tests for the Black-Scholes engine.
Run with:  python -m pytest tests.py -v
"""

import numpy as np
import pytest
from black_scholes import black_scholes, implied_volatility


# ── Known benchmark values (cross-checked against Bloomberg/Hull) ─────────
# ATM call:  S=100, K=100, T=1yr, r=5%, σ=20%, q=0
ATM_CALL = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
ATM_PUT  = dict(**ATM_CALL, option_type="put")


def test_atm_call_price():
    res = black_scholes(**ATM_CALL)
    assert abs(res.price - 10.4506) < 0.001, f"ATM call price: {res.price}"


def test_atm_put_price():
    res = black_scholes(100, 100, 1.0, 0.05, 0.20, 0.0, "put")
    assert abs(res.price - 5.5735) < 0.001, f"ATM put price: {res.price}"


def test_put_call_parity():
    """C - P = S*e^(-qT) - K*e^(-rT)"""
    S, K, T, r, sigma, q = 105, 100, 0.5, 0.05, 0.25, 0.02
    call = black_scholes(S, K, T, r, sigma, q, "call")
    put  = black_scholes(S, K, T, r, sigma, q, "put")
    lhs  = call.price - put.price
    rhs  = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8, f"PCP violated: {lhs} ≠ {rhs}"


def test_call_delta_bounds():
    res = black_scholes(100, 100, 1.0, 0.05, 0.20, 0.0, "call")
    assert 0 < res.delta < 1, f"Call delta out of bounds: {res.delta}"


def test_put_delta_bounds():
    res = black_scholes(100, 100, 1.0, 0.05, 0.20, 0.0, "put")
    assert -1 < res.delta < 0, f"Put delta out of bounds: {res.delta}"


def test_call_put_delta_relation():
    """Call delta - Put delta = e^(-qT)"""
    S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
    call = black_scholes(S, K, T, r, sigma, q, "call")
    put  = black_scholes(S, K, T, r, sigma, q, "put")
    assert abs(call.delta - put.delta - np.exp(-q * T)) < 1e-8


def test_gamma_positive():
    """Gamma is always positive for long options."""
    for otype in ["call", "put"]:
        res = black_scholes(100, 100, 0.5, 0.05, 0.20, 0.0, otype)
        assert res.gamma > 0


def test_theta_negative():
    """Theta is always negative for long options (time decay)."""
    for otype in ["call", "put"]:
        res = black_scholes(100, 100, 0.5, 0.05, 0.20, 0.0, otype)
        assert res.theta < 0


def test_vega_positive():
    """Vega is always positive for long options."""
    for otype in ["call", "put"]:
        res = black_scholes(100, 100, 0.5, 0.05, 0.20, 0.0, otype)
        assert res.vega > 0


def test_deep_itm_call_delta():
    """Deep ITM call delta ≈ 1."""
    res = black_scholes(200, 100, 1.0, 0.05, 0.20, 0.0, "call")
    assert abs(res.delta - 1.0) < 0.01


def test_deep_otm_call_delta():
    """Deep OTM call delta ≈ 0."""
    res = black_scholes(50, 100, 1.0, 0.05, 0.20, 0.0, "call")
    assert abs(res.delta) < 0.01


def test_implied_volatility_roundtrip():
    """IV(BS_price) should return original sigma."""
    for otype in ["call", "put"]:
        S, K, T, r, sigma, q = 100, 100, 0.5, 0.05, 0.22, 0.0
        market_price = black_scholes(S, K, T, r, sigma, q, otype).price
        iv = implied_volatility(market_price, S, K, T, r, q, otype)
        assert abs(iv - sigma) < 1e-4, f"IV roundtrip failed ({otype}): {iv} ≠ {sigma}"


def test_at_expiry_intrinsic():
    """At expiry (T=0), price = intrinsic value."""
    res = black_scholes(110, 100, 0.0, 0.05, 0.20, 0.0, "call")
    assert abs(res.price - 10.0) < 1e-8

    res = black_scholes(90, 100, 0.0, 0.05, 0.20, 0.0, "put")
    assert abs(res.price - 10.0) < 1e-8


def test_time_value_nonnegative():
    """Time value should always be ≥ 0."""
    for S in [80, 100, 120]:
        for otype in ["call", "put"]:
            res = black_scholes(S, 100, 0.5, 0.05, 0.20, 0.0, otype)
            assert res.time_value >= -1e-8, f"Negative time value: {res.time_value}"


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "--tb=short"])