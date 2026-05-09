#macro_regime.py

# Macro Regime Allocation Model
#
# Classifies the economy into 4 quadrants based on growth and inflation direction,
# inspired by Bridgewater's All Weather framework and Ray Dalio's work on economic machines.
#
# Quadrants:
#   1. Growth↑  Inflation↑  → Commodities, TIPS, EM equities
#   2. Growth↑  Inflation↓  → Equities, Credit, Real Estate
#   3. Growth↓  Inflation↓  → Long Bonds, Gold, Defensive equities
#   4. Growth↓  Inflation↑  → Gold, Commodities, Short bonds (stagflation)
#
# Data sources (all free via FRED API + yfinance):
#   Growth proxy:    ISM Manufacturing PMI (MANEMP or NAPM)
#   Inflation proxy: CPI YoY % change (CPIAUCSL)
#   Yield curve:     10Y - 2Y spread (T10Y2Y) — bonus recession indicator

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from dataclasses import dataclass
from typing import Optional

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# Asset ETFs per regime quadrant
REGIME_ALLOCATIONS = {
    "Growth↑ Inflation↑": {
        "DBC":  0.30,   # commodities
        "TIP":  0.25,   # TIPS (inflation-protected bonds)
        "EEM":  0.25,   # EM equities
        "GLD":  0.20,   # gold
    },
    "Growth↑ Inflation↓": {
        "SPY":  0.40,   # US equities
        "QQQ":  0.25,   # tech / growth
        "HYG":  0.20,   # high yield credit
        "VNQ":  0.15,   # real estate
    },
    "Growth↓ Inflation↓": {
        "TLT":  0.40,   # long-duration bonds
        "GLD":  0.30,   # gold
        "IEF":  0.20,   # intermediate bonds
        "VDC":  0.10,   # consumer staples (defensive)
    },
    "Growth↓ Inflation↑": {
        "GLD":  0.35,   # gold
        "DBC":  0.30,   # commodities
        "TIP":  0.25,   # TIPS
        "SHY":  0.10,   # short bonds (less rate sensitivity)
    },
}


@dataclass
class RegimeResult:
    dates:          pd.DatetimeIndex
    regime:         pd.Series          # regime label per month
    growth_signal:  pd.Series          # PMI smoothed (>50 = expansion)
    inflation_signal: pd.Series        # CPI YoY %
    yield_curve:    pd.Series          # 10Y-2Y spread
    current_regime: str
    regime_history: pd.DataFrame       # full history with signals
    portfolio_returns: pd.Series
    allocation:     dict               # current target allocation


def fetch_fred(series_id: str) -> Optional[pd.Series]:
    try:
        url  = FRED_BASE + series_id
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        rows  = [l.split(",") for l in lines[1:] if "." in l]
        df    = pd.DataFrame(rows, columns=["date", "value"])
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()
        return s
    except Exception as e:
        print(f"[macro] FRED fetch failed for {series_id}: {e}")
        return None


def fetch_macro_data(start: str = "2000-01-01") -> dict:
    print("[macro] Fetching FRED data...")

    # ISM Manufacturing PMI: MANEMP is employment, better PMI proxy is NAPM
    # Use UMCSENT (consumer sentiment) as growth fallback if PMI unavailable
    pmi    = fetch_fred("NAPM")         # ISM Manufacturing PMI
    if pmi is None:
        pmi = fetch_fred("UMCSENT")     # consumer sentiment fallback

    cpi    = fetch_fred("CPIAUCSL")     # CPI all items
    yc     = fetch_fred("T10Y2Y")       # 10Y-2Y yield spread
    unrate = fetch_fred("UNRATE")       # unemployment rate (bonus)

    # compute CPI YoY
    cpi_yoy = None
    if cpi is not None:
        cpi_monthly = cpi.resample("ME").last()
        cpi_yoy     = cpi_monthly.pct_change(12) * 100

    # resample PMI to monthly
    pmi_monthly = pmi.resample("ME").last() if pmi is not None else None
    yc_monthly  = yc.resample("ME").last()  if yc  is not None else None
    ur_monthly  = unrate.resample("ME").last() if unrate is not None else None

    return {
        "pmi":      pmi_monthly,
        "cpi_yoy":  cpi_yoy,
        "yc":       yc_monthly,
        "unrate":   ur_monthly,
    }


def classify_regime(growth_val: float, inflation_val: float,
                    growth_threshold: float = 50.0,
                    inflation_threshold: float = 2.5) -> str:
    g_up = growth_val    > growth_threshold
    i_up = inflation_val > inflation_threshold

    if g_up  and i_up:   return "Growth↑ Inflation↑"
    if g_up  and not i_up: return "Growth↑ Inflation↓"
    if not g_up and not i_up: return "Growth↓ Inflation↓"
    return "Growth↓ Inflation↑"


def build_regime_series(macro: dict, start: str = "2000-01-01") -> RegimeResult:
    pmi     = macro["pmi"]
    cpi_yoy = macro["cpi_yoy"]
    yc      = macro["yc"]

    # if PMI missing, use rolling z-score of another signal
    if pmi is None:
        print("[macro] PMI unavailable — using synthetic growth proxy")
        idx  = cpi_yoy.index if cpi_yoy is not None else pd.date_range(start, periods=100, freq="ME")
        pmi  = pd.Series(50.0, index=idx)   # neutral fallback

    if cpi_yoy is None:
        print("[macro] CPI unavailable — using 2% flat inflation")
        cpi_yoy = pd.Series(2.0, index=pmi.index)

    # align
    df = pd.DataFrame({"pmi": pmi, "cpi": cpi_yoy}).dropna()
    df = df[df.index >= pd.Timestamp(start)]

    # smooth PMI with 3-month rolling mean to reduce noise
    df["pmi_smooth"] = df["pmi"].rolling(3, min_periods=1).mean()

    # classify each month
    df["regime"] = df.apply(
        lambda r: classify_regime(r["pmi_smooth"], r["cpi"]), axis=1
    )

    if yc is not None:
        df["yc"] = yc.reindex(df.index).ffill()
    else:
        df["yc"] = 0.0

    current_regime = df["regime"].iloc[-1] if len(df) > 0 else "Growth↑ Inflation↓"

    # fetch asset prices for backtesting the allocation
    all_tickers = list(set(t for alloc in REGIME_ALLOCATIONS.values() for t in alloc))
    raw = yf.download(all_tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw = raw.ffill().dropna()

    monthly_prices  = raw.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    # backtest: each month use the regime from last month's data to set weights
    port_returns = []
    for i in range(1, len(df)):
        date   = df.index[i]
        regime = df["regime"].iloc[i-1]    # use prior month's regime (no look-ahead)
        alloc  = REGIME_ALLOCATIONS.get(regime, REGIME_ALLOCATIONS["Growth↑ Inflation↓"])

        if date not in monthly_returns.index:
            continue

        month_ret = 0.0
        total_w   = 0.0
        for ticker, weight in alloc.items():
            if ticker in monthly_returns.columns:
                r = monthly_returns.loc[date, ticker]
                if not np.isnan(r):
                    month_ret += weight * r
                    total_w   += weight

        if total_w > 0:
            month_ret /= total_w   # normalise if some assets missing

        port_returns.append({"date": date, "return": month_ret})

    port_series = pd.DataFrame(port_returns).set_index("date")["return"] if port_returns else pd.Series(dtype=float)

    return RegimeResult(
        dates            = df.index,
        regime           = df["regime"],
        growth_signal    = df["pmi_smooth"],
        inflation_signal = df["cpi"],
        yield_curve      = df["yc"],
        current_regime   = current_regime,
        regime_history   = df,
        portfolio_returns = port_series,
        allocation       = REGIME_ALLOCATIONS[current_regime],
    )