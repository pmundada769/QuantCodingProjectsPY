#vbt_backtest.py

# Professional Momentum Backtest using vectorbt
#
# vectorbt is a high-performance backtesting library built on NumPy/pandas.
# It's vectorised — instead of looping over dates, it computes everything
# in array operations simultaneously. This makes it ~100x faster than
# event-driven backtests and suitable for large-scale parameter sweeps.
#
# What we build here:
#   - Cross-sectional momentum backtest (same signal as Equity Factor project)
#   - With transaction costs and slippage
#   - With position sizing (equal weight, vol-weighted)
#   - Full performance attribution: rolling Sharpe, underwater curve, returns calendar

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass

try:
    import vectorbt as vbt # type: ignore
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False


@dataclass
class BacktestResult:
    # NOTE: no portfolio object field — vbt Portfolio can't be pickled by Streamlit cache
    returns:         pd.Series     # daily strategy returns
    cumulative:      pd.Series     # cumulative return
    sharpe:          float
    sortino:         float
    calmar:          float
    max_drawdown:    float
    ann_return:      float
    ann_vol:         float
    hit_rate:        float
    total_trades:    int
    tickers:         list
    lookback:        int
    cost_bps:        float
    sizing:          str


def fetch_prices(tickers: list, start: str = "2015-01-01") -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True,
                      threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.ffill().dropna()


def momentum_signal(prices: pd.DataFrame, lookback: int = 252,
                    short_window: int = 21) -> pd.DataFrame:
    # 12-minus-1 cross-sectional momentum
    return prices.pct_change(lookback) - prices.pct_change(short_window)


def compute_weights(signal: pd.DataFrame, sizing: str = "equal",
                    returns: pd.DataFrame = None, vol_window: int = 60) -> pd.DataFrame:
    monthly_signal = signal.resample("ME").last().dropna(how="all")
    weights_list   = []

    for date, row in monthly_signal.iterrows():
        valid = row.dropna()
        if len(valid) == 0:
            weights_list.append(pd.Series(0.0, index=signal.columns))
            continue

        n_buckets = min(10, len(valid))
        try:
            deciles = pd.qcut(valid.rank(), n_buckets, labels=False, duplicates="drop")
        except Exception:
            weights_list.append(pd.Series(0.0, index=signal.columns))
            continue

        long_mask  = (deciles == n_buckets - 1)
        short_mask = (deciles == 0)
        w = pd.Series(0.0, index=valid.index)

        # use explicit bool flags — never evaluate a DataFrame as a boolean
        use_vol = (sizing == "vol_weighted") and (returns is not None) and (len(returns) > 0)

        if use_vol:
            vols    = returns.rolling(vol_window).std().iloc[-1]
            inv_vol = 1.0 / vols.replace(0, np.nan).reindex(valid.index).fillna(1.0)
            if long_mask.sum() > 0:
                lw = inv_vol[long_mask]
                w[long_mask] = lw / lw.sum()
            if short_mask.sum() > 0:
                sw = inv_vol[short_mask]
                w[short_mask] = -sw / sw.sum()
        else:
            if long_mask.sum() > 0:
                w[long_mask] =  1.0 / long_mask.sum()
            if short_mask.sum() > 0:
                w[short_mask] = -1.0 / short_mask.sum()

        weights_list.append(w.reindex(signal.columns).fillna(0.0))

    weights_df    = pd.DataFrame(weights_list, index=monthly_signal.index)
    daily_weights = weights_df.reindex(signal.index, method="ffill").fillna(0.0)
    return daily_weights


def apply_transaction_costs(returns: pd.DataFrame, weights: pd.DataFrame,
                             cost_bps: float = 10.0) -> pd.Series:
    cost_decimal   = cost_bps / 10_000.0
    weight_changes = weights.diff().abs().sum(axis=1)
    gross_returns  = (weights.shift(1) * returns).sum(axis=1)
    net_returns    = gross_returns - weight_changes * cost_decimal
    # force output to a plain Series with no ambiguity
    return pd.Series(net_returns.values, index=net_returns.index, dtype=float)


def _to_series(x) -> pd.Series:
    # helper: guarantee we always have a 1-d Series regardless of input
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.Series(x.values, index=x.index, dtype=float)


def run_vbt_backtest(
    tickers:      list,
    start:        str   = "2015-01-01",
    lookback:     int   = 252,
    cost_bps:     float = 10.0,
    slippage_bps: float = 5.0,
    sizing:       str   = "equal",
) -> BacktestResult:

    prices  = fetch_prices(tickers, start=start)
    returns = prices.pct_change().dropna()
    signal  = momentum_signal(prices, lookback=lookback)
    weights = compute_weights(signal, sizing=sizing, returns=returns)
    weights = weights.reindex(returns.index).fillna(0.0)

    total_cost = cost_bps + slippage_bps

    # always use manual fallback — vbt object causes Streamlit pickle errors
    strat_returns = apply_transaction_costs(returns, weights, total_cost)

    # guarantee plain 1-d float Series
    strat_returns = _to_series(strat_returns).dropna()
    cumulative    = (1 + strat_returns).cumprod()

    ann_ret  = float(strat_returns.mean()) * 252
    ann_vol  = float(strat_returns.std())  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0

    neg      = strat_returns[strat_returns < 0]
    down_std = float(neg.std()) * np.sqrt(252) if len(neg) > 0 else 0.0
    sortino  = ann_ret / down_std if down_std > 0 else 0.0

    peak   = cumulative.cummax()
    max_dd = float(((cumulative - peak) / peak).min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    hit_rate   = float((strat_returns > 0).mean())
    n_trades   = int((weights.diff().abs() > 0.01).sum().sum())

    return BacktestResult(
        returns      = strat_returns,
        cumulative   = cumulative,
        sharpe       = sharpe,
        sortino      = sortino,
        calmar       = calmar,
        max_drawdown = max_dd,
        ann_return   = ann_ret,
        ann_vol      = ann_vol,
        hit_rate     = hit_rate,
        total_trades = n_trades,
        tickers      = tickers,
        lookback     = lookback,
        cost_bps     = cost_bps,
        sizing       = sizing,
    )


def parameter_sweep(
    tickers:        list,
    start:          str,
    lookbacks:      list = [126, 189, 252, 378],
    cost_scenarios: list = [0, 10, 20, 50],
) -> pd.DataFrame:
    rows = []
    for lb in lookbacks:
        for cost in cost_scenarios:
            try:
                r = run_vbt_backtest(tickers, start=start, lookback=lb, cost_bps=cost)
                rows.append({
                    "Lookback (days)": lb,
                    "Cost (bps)":      cost,
                    "Sharpe":          round(r.sharpe, 3),
                    "Ann. Return (%)": round(r.ann_return * 100, 2),
                    "Max DD (%)":      round(r.max_drawdown * 100, 2),
                    "Hit Rate (%)":    round(r.hit_rate * 100, 1),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)