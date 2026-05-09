#tracker.py

# Portfolio tracker engine.
# Reads a CSV of holdings (ticker, shares, avg_cost, sector),
# fetches live prices via yfinance, and computes P&L, sector exposure,
# daily returns, and performance metrics.

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Optional


@dataclass
class Holding:
    ticker:    str
    shares:    float
    avg_cost:  float       # cost per share
    sector:    str
    cur_price: float = 0   # filled by fetch_prices
    cur_value: float = 0   # shares * cur_price
    cost_basis:float = 0   # shares * avg_cost
    pnl:       float = 0   # cur_value - cost_basis
    pnl_pct:   float = 0   # pnl / cost_basis


def load_holdings(csv_path: str = "holdings.csv") -> list:
    # read the CSV and return a list of Holding objects
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"ticker", "shares", "avg_cost"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Need: ticker, shares, avg_cost, (sector optional)")

    if "sector" not in df.columns:
        df["sector"] = "Unknown"

    holdings = []
    for _, row in df.iterrows():
        h = Holding(
            ticker   = str(row["ticker"]).strip().upper(),
            shares   = float(row["shares"]),
            avg_cost = float(row["avg_cost"]),
            sector   = str(row.get("sector", "Unknown")).strip(),
        )
        h.cost_basis = h.shares * h.avg_cost
        holdings.append(h)

    return holdings


def fetch_prices(holdings: list) -> list:
    # fetch current prices and 1-year history for all tickers in one batch download
    tickers = [h.ticker for h in holdings]

    # current price — use period="5d" to get last available close even over weekends
    raw = yf.download(tickers, period="5d", auto_adjust=True, threads=False, progress=False)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])

    last_prices = raw.ffill().iloc[-1]   # most recent available price per ticker

    for h in holdings:
        price = float(last_prices.get(h.ticker, 0))
        if price > 0:
            h.cur_price  = price
            h.cur_value  = h.shares * price
            h.pnl        = h.cur_value - h.cost_basis
            h.pnl_pct    = (h.pnl / h.cost_basis) * 100 if h.cost_basis > 0 else 0

    return holdings


def fetch_history(holdings: list, period: str = "1y") -> pd.DataFrame:
    # fetch historical daily close prices for the full portfolio
    tickers = [h.ticker for h in holdings]
    raw     = yf.download(tickers, period=period, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.ffill().dropna()


def portfolio_daily_returns(holdings: list, prices: pd.DataFrame) -> pd.Series:
    # compute value-weighted portfolio daily returns based on current weights
    shares_dict = {h.ticker: h.shares for h in holdings}
    tickers     = [t for t in shares_dict if t in prices.columns]

    # portfolio value each day = sum(shares * price)
    portfolio_value = sum(
        prices[t] * shares_dict[t] for t in tickers
    )

    daily_ret = portfolio_value.pct_change().dropna()
    daily_ret.name = "Portfolio"
    return daily_ret


def holdings_to_df(holdings: list) -> pd.DataFrame:
    # convert list of Holding objects to a clean DataFrame for display
    rows = []
    for h in holdings:
        rows.append({
            "Ticker":      h.ticker,
            "Sector":      h.sector,
            "Shares":      h.shares,
            "Avg Cost":    h.avg_cost,
            "Cur Price":   round(h.cur_price, 2),
            "Cur Value":   round(h.cur_value, 2),
            "Cost Basis":  round(h.cost_basis, 2),
            "P&L ($)":     round(h.pnl, 2),
            "P&L (%)":     round(h.pnl_pct, 2),
        })
    return pd.DataFrame(rows).sort_values("Cur Value", ascending=False)


def sector_summary(holdings: list) -> pd.DataFrame:
    # aggregate by sector: total value and total P&L
    df = holdings_to_df(holdings)
    total_value = df["Cur Value"].sum()

    summary = df.groupby("Sector").agg(
        Value   = ("Cur Value",  "sum"),
        PnL     = ("P&L ($)",    "sum"),
        CostBasis = ("Cost Basis","sum"),
    ).reset_index()

    summary["Weight (%)"] = (summary["Value"] / total_value * 100).round(2)
    summary["Return (%)"] = (summary["PnL"] / summary["CostBasis"] * 100).round(2)
    return summary.sort_values("Value", ascending=False)


def portfolio_metrics(daily_returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
    # compute key performance metrics for the portfolio
    r    = daily_returns.dropna()
    ann  = np.sqrt(252)

    total_return  = (1 + r).prod() - 1
    ann_return    = (1 + total_return) ** (252 / len(r)) - 1
    ann_vol       = r.std() * ann
    sharpe        = (r.mean() * 252) / (r.std() * ann) if r.std() > 0 else 0

    downside      = r[r < 0].std() * ann
    sortino       = (r.mean() * 252) / downside if downside > 0 else 0

    cum           = (1 + r).cumprod()
    peak          = cum.cummax()
    max_dd        = ((cum - peak) / peak).min()

    hit_rate      = (r > 0).mean()

    metrics = {
        "Total Return":     f"{total_return*100:.2f}%",
        "Ann. Return":      f"{ann_return*100:.2f}%",
        "Ann. Volatility":  f"{ann_vol*100:.2f}%",
        "Sharpe Ratio":     f"{sharpe:.3f}",
        "Sortino Ratio":    f"{sortino:.3f}",
        "Max Drawdown":     f"{max_dd*100:.2f}%",
        "Hit Rate":         f"{hit_rate*100:.1f}%",
        "Trading Days":     str(len(r)),
    }

    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).dropna()
        if len(b) > 10:
            from scipy import stats
            slope, intercept, rv, _, _ = stats.linregress(b.values, r.reindex(b.index).values)
            metrics["Beta (vs SPY)"]  = f"{slope:.3f}"
            metrics["Alpha (ann %)"]  = f"{intercept*252*100:.2f}%"

    return metrics