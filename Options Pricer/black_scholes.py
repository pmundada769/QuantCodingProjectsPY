"""
Black-Scholes Options Pricing Engine
Computes option prices and all first/second-order Greeks.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass
class OptionResult:
    """Container for BS price + all Greeks."""
    option_type: str
    price: float
    # First-order Greeks
    delta: float
    theta: float   # per calendar day
    rho: float     # per 1% change in r
    vega: float    # per 1% change in vol
    # Second-order Greeks
    gamma: float
    vanna: float   # dDelta/dVol
    charm: float   # dDelta/dTime (per day)
    volga: float   # dVega/dVol (aka vomma)
    # Auxiliary
    d1: float
    d2: float
    intrinsic: float
    time_value: float
    nd1: float     # N(d1) for calls / N(-d1) for puts
    nd2: float


def black_scholes(
    S: float,          # Spot price
    K: float,          # Strike price
    T: float,          # Time to expiry in years
    r: float,          # Risk-free rate (decimal)
    sigma: float,      # Volatility (decimal)
    q: float = 0.0,    # Continuous dividend yield
    option_type: Literal["call", "put"] = "call",
) -> OptionResult:
    """
    Full Black-Scholes pricer with Greeks.

    Args:
        S: Current underlying price
        K: Option strike price
        T: Time to expiration in years (e.g. 30/365)
        r: Annualised risk-free rate as decimal (e.g. 0.05 = 5%)
        sigma: Annualised implied volatility as decimal (e.g. 0.20 = 20%)
        q: Continuous dividend yield as decimal
        option_type: 'call' or 'put'

    Returns:
        OptionResult with price and all Greeks
    """
    if T <= 0:
        # At expiry – return intrinsic value, zero Greeks
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return OptionResult(
            option_type=option_type, price=intrinsic,
            delta=1.0 if (option_type == "call" and S > K) else (-1.0 if (option_type == "put" and S < K) else 0.0),
            theta=0, rho=0, vega=0, gamma=0, vanna=0, charm=0, volga=0,
            d1=0, d2=0, intrinsic=intrinsic, time_value=0,
            nd1=float(S > K), nd2=float(S > K),
        )

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    Nd1  = norm.cdf(d1)
    Nd2  = norm.cdf(d2)
    Nnd1 = norm.cdf(-d1)
    Nnd2 = norm.cdf(-d2)
    nd1  = norm.pdf(d1)   # standard normal PDF at d1

    disc = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option_type == "call":
        price = S * disc_q * Nd1 - K * disc * Nd2
        delta = disc_q * Nd1
        rho   = K * T * disc * Nd2 / 100          # per 1% move in r
        nd_signed = Nd1
    else:
        price = K * disc * Nnd2 - S * disc_q * Nnd1
        delta = -disc_q * Nnd1
        rho   = -K * T * disc * Nnd2 / 100
        nd_signed = Nnd1

    # Gamma (same for call and put)
    gamma = disc_q * nd1 / (S * sigma * sqrt_T)

    # Vega (same for call and put) – per 1% move in vol
    vega = S * disc_q * nd1 * sqrt_T / 100

    # Theta – per calendar day
    theta_base = (
        -(S * disc_q * nd1 * sigma) / (2 * sqrt_T)
        + q * S * disc_q * nd_signed
    )
    if option_type == "call":
        theta = (theta_base - r * K * disc * Nd2) / 365
    else:
        theta = (theta_base + r * K * disc * Nnd2) / 365

    # Vanna = dDelta/dSigma = dVega/dS
    vanna = -disc_q * nd1 * d2 / sigma

    # Charm = dDelta/dT (per calendar day)
    if option_type == "call":
        charm = (disc_q * (nd1 * (r - q - (d2 * sigma) / (2 * T)) / (sigma * sqrt_T) - q * Nd1)) / 365
    else:
        charm = (disc_q * (nd1 * (r - q - (d2 * sigma) / (2 * T)) / (sigma * sqrt_T) + q * Nnd1)) / 365

    # Volga / Vomma = dVega/dSigma
    volga = vega * d1 * d2 / sigma  # still per 1% vol (vega already scaled)

    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    time_value = price - intrinsic

    return OptionResult(
        option_type=option_type,
        price=price,
        delta=delta,
        theta=theta,
        rho=rho,
        vega=vega,
        gamma=gamma,
        vanna=vanna,
        charm=charm,
        volga=volga,
        d1=d1,
        d2=d2,
        intrinsic=intrinsic,
        time_value=time_value,
        nd1=Nd1 if option_type == "call" else Nnd1,
        nd2=Nd2 if option_type == "call" else Nnd2,
    )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Newton-Raphson implied volatility solver.

    Returns IV as a decimal (e.g. 0.25 = 25%), or np.nan if it fails to converge.
    """
    # Bounds check
    if T <= 0:
        return np.nan

    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        res = black_scholes(S, K, T, r, sigma, q, option_type)
        diff = res.price - market_price
        # vega in $ per 1% vol → convert back to per unit vol
        vega_unit = res.vega * 100
        if abs(vega_unit) < 1e-10:
            break
        sigma_new = sigma - diff / vega_unit
        sigma_new = max(1e-4, min(sigma_new, 10.0))   # clamp to sane range
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new
    return np.nan