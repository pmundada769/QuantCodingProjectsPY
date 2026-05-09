#correlation.py

# Rolling correlation engine.
# Computes rolling pairwise correlations, detects regime shifts,
# and flags when correlations spike (crisis) or fall (normal/low-vol).

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass


@dataclass
class RegimeShift:
    date:       str
    pair:       str      # e.g. "AAPL–SPY"
    old_corr:   float    # correlation before the shift
    new_corr:   float    # correlation after the shift
    delta:      float    # magnitude of change
    direction:  str      # "spike" or "collapse"


def fetch_prices(tickers: list, start: str = "2018-01-01") -> pd.DataFrame:
    data = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data.ffill().dropna()


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def rolling_correlation(
    returns:  pd.DataFrame,
    window:   int = 60,       # rolling window in trading days
) -> dict:
    # returns a dict of {(ticker_a, ticker_b): pd.Series of rolling correlation}
    cols   = returns.columns.tolist()
    result = {}

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            roll = returns[a].rolling(window).corr(returns[b])
            result[(a, b)] = roll.dropna()

    return result


def correlation_at_date(returns: pd.DataFrame, date=None) -> pd.DataFrame:
    # full correlation matrix at a specific date using the trailing window
    # defaults to most recent date
    if date is None:
        subset = returns
    else:
        subset = returns.loc[:date]

    return subset.corr()


def average_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    # average pairwise correlation across all pairs at each point in time
    # a high average correlation = assets moving together = low diversification benefit
    # spikes during market stress (crisis regime)

    cols    = returns.columns.tolist()
    pairs   = [(a, b) for i, a in enumerate(cols) for b in cols[i+1:]]
    roll_corrs = pd.DataFrame({
        f"{a}_{b}": returns[a].rolling(window).corr(returns[b])
        for a, b in pairs
    })

    avg = roll_corrs.mean(axis=1).dropna()
    avg.name = "Avg Pairwise Correlation"
    return avg


def detect_regime_shifts(
    rolling_corr: dict,
    threshold:    float = 0.30,    # minimum change to count as a regime shift
    window:       int   = 20,      # look-back for "old" vs "new" correlation
) -> list:
    # scan each pair's rolling correlation for sudden large jumps or drops
    # a jump of >threshold in a short window is flagged as a regime shift

    shifts = []
    for (a, b), series in rolling_corr.items():
        diff = series.diff(window).dropna()
        for date, delta in diff.items():
            if abs(delta) >= threshold:
                old_c = series.loc[date] - delta
                new_c = series.loc[date]
                shifts.append(RegimeShift(
                    date      = str(date.date()),
                    pair      = f"{a}–{b}",
                    old_corr  = round(old_c, 3),
                    new_corr  = round(new_c, 3),
                    delta     = round(delta, 3),
                    direction = "spike" if delta > 0 else "collapse",
                ))

    return sorted(shifts, key=lambda x: abs(x.delta), reverse=True)


def correlation_percentile(
    rolling_corr: dict,
    pair:         tuple,
) -> pd.Series:
    # for a given pair, how does the current correlation compare historically?
    # returns expanding percentile rank: 100 = highest ever, 0 = lowest ever

    series = rolling_corr.get(pair)
    if series is None:
        return pd.Series(dtype=float)

    return series.expanding().rank(pct=True) * 100


def dispersion(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    # cross-sectional dispersion of returns — how different are assets from each other?
    # high dispersion = assets diverging = potentially more alpha opportunities
    # low dispersion = assets moving together = crisis or macro-driven market

    rolling_std = returns.rolling(window).std().mean(axis=1)
    rolling_std.name = "Cross-Sectional Dispersion"
    return rolling_std.dropna()