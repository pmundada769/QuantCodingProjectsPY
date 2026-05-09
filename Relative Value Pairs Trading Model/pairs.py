#pairs.py

# Statistical Arbitrage — Pairs Trading via Cointegration
#
# Two approaches implemented:
#   1. Engle-Granger (1987): two-step OLS test for cointegration between two series
#   2. Johansen (1991): multivariate test, handles more than 2 assets
#
# Trading rule:
#   - Spread = price_A - hedge_ratio × price_B
#   - z-score = (spread - mean) / std  (rolling)
#   - Enter long when z < -2  (spread too low → expect mean reversion upward)
#   - Enter short when z > +2 (spread too high → expect mean reversion downward)
#   - Exit at z = 0
#
# Use statsmodels for cointegration tests.

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Optional, Tuple
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from itertools import combinations


@dataclass
class PairResult:
    ticker_a:        str
    ticker_b:        str
    hedge_ratio:     float         # OLS slope: how many shares of B per share of A
    eg_pvalue:       float         # Engle-Granger cointegration p-value
    cointegrated:    bool          # True if p < 0.05
    spread:          pd.Series     # price_A - hedge_ratio * price_B
    zscore:          pd.Series     # rolling z-score of spread
    signals:         pd.Series     # +1 long spread, -1 short spread, 0 flat
    strategy_returns: pd.Series    # daily P&L of the pairs trade
    sharpe:          float
    max_drawdown:    float
    n_trades:        int
    half_life:       float         # mean reversion speed in days


@dataclass
class ScanResult:
    all_pairs:         list         # all PairResult objects tested
    cointegrated_pairs: list        # only those that passed cointegration test
    best_sharpe:       Optional[object]  # PairResult with highest Sharpe


def fetch_prices(tickers: list, start: str = "2015-01-01") -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.ffill().dropna()


def hedge_ratio_ols(price_a: pd.Series, price_b: pd.Series) -> float:
    # OLS regression of price_A on price_B to find the hedge ratio
    # hedge_ratio = how many shares of B to hold per share of A
    # We regress in log space to avoid spurious relationships from price trends
    X = sm.add_constant(price_b.values)
    model = sm.OLS(price_a.values, X).fit()
    return float(model.params[1])


def compute_spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: float) -> pd.Series:
    spread = price_a - hedge_ratio * price_b
    spread.name = f"{price_a.name}–{price_b.name}_spread"
    return spread


def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    # z-score of the spread relative to its rolling mean and std
    # tells you how many standard deviations away from "normal" the spread is
    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    z = (spread - roll_mean) / roll_std.replace(0, np.nan)
    z.name = "z-score"
    return z


def half_life(spread: pd.Series) -> float:
    # Ornstein-Uhlenbeck half-life of mean reversion
    # Measures: how many days does it take for the spread to revert halfway?
    # Shorter = faster mean reversion = better for pairs trading
    # Computed via OLS: Δspread_t = λ × spread_{t-1} + ε
    # half_life = -log(2) / λ

    spread_lag   = spread.shift(1).dropna()
    spread_diff  = spread.diff().dropna()
    aligned      = pd.concat([spread_diff, spread_lag], axis=1).dropna()

    X = sm.add_constant(aligned.iloc[:, 1].values)
    y = aligned.iloc[:, 0].values
    res = sm.OLS(y, X).fit()

    lam = res.params[1]
    if lam >= 0:
        return np.nan   # no mean reversion
    return float(-np.log(2) / lam)


def generate_signals(
    zscore:           pd.Series,
    entry_threshold:  float = 2.0,   # enter when |z| > this
    exit_threshold:   float = 0.5,   # exit when |z| < this
) -> pd.Series:
    # +1 = long spread (spread is too low, expect it to rise)
    # -1 = short spread (spread is too high, expect it to fall)
    #  0 = flat

    signals = pd.Series(0, index=zscore.index, dtype=float)
    position = 0

    for i in range(1, len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if z < -entry_threshold:
                position = 1     # spread too low → go long
            elif z > entry_threshold:
                position = -1    # spread too high → go short
        elif position == 1:
            if z > -exit_threshold:
                position = 0     # mean-reverted, exit
        elif position == -1:
            if z < exit_threshold:
                position = 0     # mean-reverted, exit

        signals.iloc[i] = position

    return signals


def backtest_pair(
    price_a:    pd.Series,
    price_b:    pd.Series,
    signals:    pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    # P&L of the pairs trade:
    # Long spread = long A, short (hedge_ratio × B)
    # Short spread = short A, long (hedge_ratio × B)

    ret_a = price_a.pct_change()
    ret_b = price_b.pct_change()

    # portfolio return = signal × (ret_a - hedge_ratio × ret_b) — lagged signal
    port_ret = signals.shift(1) * (ret_a - hedge_ratio * ret_b)
    port_ret = port_ret.dropna()
    port_ret.name = f"{price_a.name}_{price_b.name}_pnl"
    return port_ret


def analyse_pair(
    ticker_a: str,
    ticker_b: str,
    prices:   pd.DataFrame,
    window:   int   = 60,
    entry:    float = 2.0,
    exit:     float = 0.5,
) -> PairResult:

    pa = prices[ticker_a]
    pb = prices[ticker_b]

    # Engle-Granger cointegration test
    # H0: no cointegration. Low p-value → reject H0 → cointegrated
    score, pval, _ = coint(pa, pb)

    hr    = hedge_ratio_ols(pa, pb)
    spr   = compute_spread(pa, pb, hr)
    z     = rolling_zscore(spr, window=window)
    hl    = half_life(spr)
    sigs  = generate_signals(z, entry, exit)
    pnl   = backtest_pair(pa, pb, sigs, hr)

    cum      = (1 + pnl).cumprod()
    ann_ret  = pnl.mean() * 252
    ann_vol  = pnl.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0

    peak     = cum.cummax()
    max_dd   = ((cum - peak) / peak).min()

    n_trades = int((sigs.diff().abs() > 0).sum() // 2)

    return PairResult(
        ticker_a          = ticker_a,
        ticker_b          = ticker_b,
        hedge_ratio       = hr,
        eg_pvalue         = pval,
        cointegrated      = pval < 0.05,
        spread            = spr,
        zscore            = z,
        signals           = sigs,
        strategy_returns  = pnl,
        sharpe            = sharpe,
        max_drawdown      = max_dd,
        n_trades          = n_trades,
        half_life         = hl,
    )


def scan_all_pairs(
    prices: pd.DataFrame,
    window: int   = 60,
    entry:  float = 2.0,
    exit:   float = 0.5,
) -> ScanResult:
    # test all possible pairs in the universe for cointegration
    tickers = prices.columns.tolist()
    all_results = []

    for a, b in combinations(tickers, 2):
        try:
            result = analyse_pair(a, b, prices, window, entry, exit)
            all_results.append(result)
        except Exception:
            pass

    cointegrated = [r for r in all_results if r.cointegrated]
    best = max(cointegrated, key=lambda r: r.sharpe) if cointegrated else None

    return ScanResult(
        all_pairs          = all_results,
        cointegrated_pairs = cointegrated,
        best_sharpe        = best,
    )


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    # Johansen cointegration test for the full universe
    # Returns the number of cointegrating relationships at 95% confidence
    # More powerful than Engle-Granger for multiple assets

    result = coint_johansen(prices.dropna(), det_order=det_order, k_ar_diff=k_ar_diff)

    # trace statistic vs critical values at 95%
    n_cointegrating = 0
    for i in range(len(result.lr1)):
        if result.lr1[i] > result.cvt[i, 1]:   # cvt[:,1] = 95% critical value
            n_cointegrating += 1
        else:
            break

    return {
        "n_cointegrating_vectors": n_cointegrating,
        "trace_stats":   result.lr1.tolist(),
        "critical_95":   result.cvt[:, 1].tolist(),
        "eigenvalues":   result.lr2.tolist(),
        "tickers":       prices.columns.tolist(),
    }