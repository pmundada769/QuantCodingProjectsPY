#alpha_analysis.py

# alphalens-style Factor Analysis
#
# alphalens (Quantopian) is the industry standard for evaluating alpha factors.
# It computes:
#   - IC (Information Coefficient): correlation of factor with forward returns
#   - IC IR (Information Ratio): mean IC / std IC — consistency of the signal
#   - Factor returns by quantile: do top-ranked stocks actually outperform?
#   - Turnover: how much does the ranking change month-to-month?
#   - Factor decay: does the signal fade over longer horizons?
#
# This project replicates the key alphalens outputs using pandas/scipy
# so it works without the alphalens install, with an upgrade path to
# alphalens-reloaded for the full output.

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr # type: ignore
from dataclasses import dataclass

try:
    import alphalens # type: ignore
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
    print("[alphalens] alphalens not installed — using manual implementation.")
    print("[alphalens] Install: pip install alphalens-reloaded")


@dataclass
class FactorAnalysis:
    # full IC time series
    ic_series:      pd.Series     # daily Spearman IC vs 1-day forward return
    ic_5d:          pd.Series     # IC vs 5-day forward return
    ic_21d:         pd.Series     # IC vs 21-day forward return
    # summary
    mean_ic:        float
    ic_ir:          float         # mean IC / std IC
    mean_ic_5d:     float
    mean_ic_21d:    float
    # quantile returns
    quantile_returns: pd.DataFrame    # mean return per quantile per period
    spread_return:    pd.Series       # top quantile minus bottom quantile
    # turnover
    monthly_turnover: pd.Series
    avg_turnover:     float
    # decay
    ic_by_horizon:    pd.DataFrame   # IC at 1,2,3,5,10,21 day horizons
    factor_name:      str


def fetch_prices(tickers: list, start: str = "2015-01-01") -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.ffill().dropna()


def momentum_factor(prices: pd.DataFrame, lookback: int = 252,
                    short: int = 21) -> pd.DataFrame:
    return prices.pct_change(lookback) - prices.pct_change(short)


def compute_ic_series(factor: pd.DataFrame, returns: pd.DataFrame,
                      horizon: int = 1) -> pd.Series:
    # Spearman IC between today's factor ranks and horizon-day forward returns
    fwd_returns = returns.shift(-horizon).rolling(horizon).sum()
    ics = []
    dates = []

    for date in factor.index:
        if date not in fwd_returns.index:
            continue
        f_vals = factor.loc[date].dropna()
        r_vals = fwd_returns.loc[date].reindex(f_vals.index).dropna()
        common = f_vals.index.intersection(r_vals.index)
        if len(common) < 5:
            continue
        ic, _ = spearmanr(f_vals[common], r_vals[common])
        ics.append(ic)
        dates.append(date)

    return pd.Series(ics, index=dates, name=f"IC_{horizon}d")


def compute_quantile_returns(factor: pd.DataFrame, returns: pd.DataFrame,
                              n_quantiles: int = 5,
                              horizon: int = 5) -> pd.DataFrame:
    # compute mean return for each factor quantile
    fwd_returns = returns.shift(-horizon).rolling(horizon).sum()
    result_rows = []

    for date in factor.index:
        if date not in fwd_returns.index:
            continue
        f_vals = factor.loc[date].dropna()
        r_vals = fwd_returns.loc[date].reindex(f_vals.index).dropna()
        common = f_vals.index.intersection(r_vals.index)
        if len(common) < n_quantiles:
            continue

        try:
            q = pd.qcut(f_vals[common].rank(), n_quantiles, labels=False, duplicates="drop")
        except Exception:
            continue

        for qnum in range(n_quantiles):
            mask = q == qnum
            if mask.sum() > 0:
                mean_ret = r_vals[common][mask].mean()
                result_rows.append({"date": date, "quantile": qnum + 1, "return": mean_ret})

    if not result_rows:
        return pd.DataFrame()

    df = pd.DataFrame(result_rows)
    pivot = df.groupby(["date", "quantile"])["return"].mean().unstack("quantile")
    pivot.columns = [f"Q{int(c)}" for c in pivot.columns]
    return pivot


def compute_turnover(factor: pd.DataFrame, n_quantiles: int = 5) -> pd.Series:
    monthly = factor.resample("ME").last().dropna(how="all")
    turnovers = []
    dates     = []

    for i in range(1, len(monthly)):
        curr = monthly.iloc[i].dropna()
        prev = monthly.iloc[i-1].dropna()
        common = curr.index.intersection(prev.index)
        if len(common) < n_quantiles:
            continue

        try:
            q_curr = pd.qcut(curr[common].rank(), n_quantiles, labels=False, duplicates="drop")
            q_prev = pd.qcut(prev[common].rank(), n_quantiles, labels=False, duplicates="drop")
        except Exception:
            continue

        # fraction of stocks that changed quantile
        changed = (q_curr != q_prev).mean()
        turnovers.append(changed)
        dates.append(monthly.index[i])

    return pd.Series(turnovers, index=dates, name="Turnover")


def ic_by_horizon(factor: pd.DataFrame, returns: pd.DataFrame,
                  horizons: list = [1, 2, 3, 5, 10, 21]) -> pd.DataFrame:
    mean_ics = {}
    for h in horizons:
        ic = compute_ic_series(factor, returns, horizon=h)
        mean_ics[h] = ic.mean() if len(ic) > 0 else 0
    return pd.DataFrame({"Horizon (days)": list(mean_ics.keys()),
                          "Mean IC": list(mean_ics.values())})


def run_factor_analysis(
    tickers:     list,
    start:       str = "2015-01-01",
    lookback:    int = 252,
    n_quantiles: int = 5,
    factor_name: str = "Momentum (12-1)",
) -> FactorAnalysis:

    prices  = fetch_prices(tickers, start=start)
    returns = prices.pct_change().dropna()
    factor  = momentum_factor(prices, lookback=lookback)
    factor  = factor.reindex(returns.index)

    ic_1d  = compute_ic_series(factor, returns, horizon=1)
    ic_5d  = compute_ic_series(factor, returns, horizon=5)
    ic_21d = compute_ic_series(factor, returns, horizon=21)

    q_rets  = compute_quantile_returns(factor, returns, n_quantiles=n_quantiles)
    spread  = (q_rets[f"Q{n_quantiles}"] - q_rets["Q1"]).dropna() if len(q_rets) > 0 else pd.Series(dtype=float)
    turnover = compute_turnover(factor, n_quantiles=n_quantiles)
    decay    = ic_by_horizon(factor, returns)

    ic_ir = float(ic_1d.mean() / ic_1d.std()) if ic_1d.std() > 0 else 0

    return FactorAnalysis(
        ic_series        = ic_1d,
        ic_5d            = ic_5d,
        ic_21d           = ic_21d,
        mean_ic          = float(ic_1d.mean()),
        ic_ir            = ic_ir,
        mean_ic_5d       = float(ic_5d.mean()),
        mean_ic_21d      = float(ic_21d.mean()),
        quantile_returns = q_rets,
        spread_return    = spread,
        monthly_turnover = turnover,
        avg_turnover     = float(turnover.mean()) if len(turnover) > 0 else 0,
        ic_by_horizon    = decay,
        factor_name      = factor_name,
    )