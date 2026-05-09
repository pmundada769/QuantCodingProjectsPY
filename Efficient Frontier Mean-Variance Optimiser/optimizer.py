#optimizer.py

# Core mean-variance optimization engine.
# Implements Markowitz (1952) portfolio theory from scratch using scipy.
# Also wraps pyportfolioopt for convenience — falls back to pure scipy if not installed.

import numpy as np
import pandas as pd
from scipy.optimize import minimize # type: ignore
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Portfolio:
    # a single point on the efficient frontier
    weights:        dict            # {ticker: weight}
    expected_return: float          # annualised expected return
    volatility:     float           # annualised standard deviation
    sharpe:         float           # Sharpe ratio (risk-free adjusted)
    label:          str = ""        # e.g. "Max Sharpe", "Min Vol"


@dataclass
class FrontierResult:
    # full output from the optimizer
    frontier_portfolios: list       # list of Portfolio objects along the frontier
    max_sharpe:         Portfolio   # tangency portfolio (highest Sharpe)
    min_vol:            Portfolio   # global minimum variance portfolio
    tickers:            list        # asset names
    mean_returns:       np.ndarray  # annualised expected returns vector
    cov_matrix:         np.ndarray  # annualised covariance matrix
    risk_free_rate:     float       # risk-free rate used


def compute_frontier(
    prices:          pd.DataFrame,  # DataFrame of adjusted close prices, one column per ticker
    risk_free_rate:  float = 0.04,  # annualised risk-free rate (e.g. 0.04 = 4%)
    n_frontier:      int   = 120,   # number of portfolios to plot along the frontier
    n_random:        int   = 3000,  # random portfolios to scatter behind the frontier
) -> FrontierResult:

    # Step 1: compute daily returns and annualise
    daily_returns = prices.pct_change().dropna()

    # mean_returns: annualised expected return per asset (252 trading days per year)
    mean_returns = daily_returns.mean() * 252

    # cov_matrix: annualised covariance matrix
    # daily cov * 252 gives annual cov because variance scales linearly with time
    cov_matrix   = daily_returns.cov() * 252

    tickers = list(prices.columns)
    n       = len(tickers)

    # --- helper functions ---

    def portfolio_return(w):
        # expected return of a portfolio with weights w
        return float(np.dot(w, mean_returns))

    def portfolio_vol(w):
        # annualised standard deviation: sqrt(w.T @ Σ @ w)
        return float(np.sqrt(w @ cov_matrix.values @ w))

    def neg_sharpe(w):
        # negative Sharpe for minimisation (scipy minimises, so negate to maximise)
        ret = portfolio_return(w)
        vol = portfolio_vol(w)
        return -(ret - risk_free_rate) / vol if vol > 0 else 0

    # constraints and bounds shared across optimisations
    # "eq" constraint: weights sum to 1 (fully invested portfolio)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # bounds: each weight between 0 and 1 (long-only, no short selling or leverage)
    bounds = [(0.0, 1.0)] * n

    # initial guess: equal weight across all assets
    w0 = np.ones(n) / n

    # --- Max Sharpe portfolio (tangency portfolio) ---
    # This is the portfolio on the Capital Market Line tangent to the frontier
    res_sharpe = minimize(
        neg_sharpe, w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"ftol": 1e-12, "maxiter": 1000},
    )
    w_sharpe   = res_sharpe.x
    max_sharpe = Portfolio(
        weights         = dict(zip(tickers, w_sharpe)),
        expected_return = portfolio_return(w_sharpe),
        volatility      = portfolio_vol(w_sharpe),
        sharpe          = (portfolio_return(w_sharpe) - risk_free_rate) / portfolio_vol(w_sharpe),
        label           = "Max Sharpe",
    )

    # --- Min Volatility portfolio (global minimum variance) ---
    # Minimise portfolio variance regardless of return
    res_minvol = minimize(
        portfolio_vol, w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"ftol": 1e-12, "maxiter": 1000},
    )
    w_minvol = res_minvol.x
    min_vol  = Portfolio(
        weights         = dict(zip(tickers, w_minvol)),
        expected_return = portfolio_return(w_minvol),
        volatility      = portfolio_vol(w_minvol),
        sharpe          = (portfolio_return(w_minvol) - risk_free_rate) / portfolio_vol(w_minvol),
        label           = "Min Vol",
    )

    # --- Efficient Frontier: sweep target returns from min-vol to max possible ---
    # The frontier is the set of portfolios with minimum volatility for each return level
    ret_min = portfolio_return(w_minvol)
    ret_max = float(mean_returns.max())
    target_returns = np.linspace(ret_min, ret_max * 0.99, n_frontier)

    frontier_portfolios = []
    w_prev = w0.copy()

    for target in target_returns:
        # add a second constraint: portfolio must hit exactly this target return
        cons = constraints + [
            {"type": "eq", "fun": lambda w, t=target: portfolio_return(w) - t}
        ]
        res = minimize(
            portfolio_vol, w_prev,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = cons,
            options     = {"ftol": 1e-12, "maxiter": 500},
        )
        if res.success:
            w = res.x
            w_prev = w   # warm start: use solution as next initial guess (faster convergence)
            frontier_portfolios.append(Portfolio(
                weights         = dict(zip(tickers, w)),
                expected_return = portfolio_return(w),
                volatility      = portfolio_vol(w),
                sharpe          = (portfolio_return(w) - risk_free_rate) / portfolio_vol(w),
            ))

    return FrontierResult(
        frontier_portfolios = frontier_portfolios,
        max_sharpe          = max_sharpe,
        min_vol             = min_vol,
        tickers             = tickers,
        mean_returns        = mean_returns.values,
        cov_matrix          = cov_matrix.values,
        risk_free_rate      = risk_free_rate,
    )


def random_portfolios(
    mean_returns: np.ndarray,
    cov_matrix:   np.ndarray,
    tickers:      list,
    n:            int   = 3000,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    # Generate n random weight vectors to scatter behind the frontier
    # Visually shows the "feasible set" — all possible portfolios
    # The frontier is the left/upper edge of this cloud

    n_assets = len(tickers)
    results  = []

    for _ in range(n):
        # draw from Dirichlet distribution — guarantees weights sum to 1 and are all >= 0
        w = np.random.dirichlet(np.ones(n_assets))
        ret = float(np.dot(w, mean_returns))
        vol = float(np.sqrt(w @ cov_matrix @ w))
        sr  = (ret - risk_free_rate) / vol if vol > 0 else 0
        results.append({"Return": ret, "Volatility": vol, "Sharpe": sr})

    return pd.DataFrame(results)


def get_price_data(tickers: list, start: str = "2018-01-01") -> pd.DataFrame:
    # Download adjusted close prices from Yahoo Finance
    import yfinance as yf
    data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data.dropna()


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    # Pearson correlation of daily returns — used for the heatmap tab
    return prices.pct_change().dropna().corr()