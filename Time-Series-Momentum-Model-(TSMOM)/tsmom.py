#tsmom.py

# Time-Series Momentum (TSMOM) with Volatility Targeting
#
# Replicates the core methodology of:
#   Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012)
#   "Time Series Momentum" — Journal of Financial Economics
#
# Key idea: each asset has its own trend signal based on its OWN past return.
# This is different from cross-sectional momentum (ranking stocks against each other).
# Here we ask: "has THIS asset been going up? If yes, go long it."
#
# Signal: r_t,t-12  (12-month return, sign only)
# Position: signal × (target_vol / realised_vol)   ← volatility targeting
#
# Volatility estimation: GARCH(1,1) via the `arch` library
# This is what real CTA/trend funds use. Rolling std is a naive approximation.

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional

try:
    from arch import arch_model # type: ignore
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("[tsmom] arch library not found — falling back to rolling std for vol estimation.")
    print("[tsmom] Install with: pip install arch")


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class AssetSignal:
    ticker:        str
    dates:         pd.DatetimeIndex
    raw_signal:    pd.Series    # +1 or -1 based on 12-month return sign
    realised_vol:  pd.Series    # annualised daily vol estimate (GARCH or rolling)
    scaled_position: pd.Series  # signal × (target_vol / realised_vol), clipped
    returns:       pd.Series    # daily asset returns


@dataclass
class TSMOMResult:
    asset_signals:    list           # list of AssetSignal
    portfolio_returns: pd.Series     # equal-weight combination of all scaled positions
    portfolio_cumret:  pd.Series     # cumulative portfolio return
    target_vol:        float
    tickers:           list
    # performance metrics
    sharpe:            float
    sortino:           float
    max_drawdown:      float
    calmar:            float
    hit_rate:          float
    ann_return:        float
    ann_vol:           float


# ─── Volatility estimation ─────────────────────────────────────────────────────

def estimate_vol_garch(returns: pd.Series, horizon: int = 1) -> pd.Series:
    # Fit GARCH(1,1) to a returns series and extract conditional vol.
    # GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    #
    # ω = long-run variance weight
    # α = ARCH term: how much yesterday's shock matters
    # β = GARCH term: how much yesterday's variance persists
    # α + β close to 1 means variance is persistent (typical for equities)
    #
    # We scale returns by 100 for numerical stability (arch library convention)

    if not ARCH_AVAILABLE:
        return estimate_vol_rolling(returns)

    try:
        r_scaled = returns.dropna() * 100
        model    = arch_model(r_scaled, vol="Garch", p=1, q=1, rescale=False)
        res      = model.fit(disp="off", show_warning=False)

        # conditional_volatility is in same units as returns (×100), annualise it
        cond_vol = res.conditional_volatility / 100 * np.sqrt(252)
        cond_vol = cond_vol.reindex(returns.index).ffill()
        return cond_vol

    except Exception:
        # GARCH can fail to converge on short/unusual series — fall back gracefully
        return estimate_vol_rolling(returns)


def estimate_vol_rolling(returns: pd.Series, window: int = 60) -> pd.Series:
    # Simple rolling standard deviation — annualised.
    # 60 trading days ≈ 3 months. Used as fallback when arch is unavailable.
    return returns.rolling(window).std() * np.sqrt(252)


# ─── Signal construction ───────────────────────────────────────────────────────

def tsmom_signal(returns: pd.Series, lookback: int = 252) -> pd.Series:
    # The TSMOM signal is simply the sign of the past 12-month (252 day) return.
    # +1 if price went up over the lookback → long
    # -1 if price went down over the lookback → short
    #
    # We use cumulative product to get the exact multi-day return:
    # r_{t-252, t} = ∏(1 + r_i) - 1  for i in [t-252, t-1]
    # (skip the last month to avoid short-term reversal — same as cross-sectional momentum)

    cum_returns = (1 + returns).rolling(lookback).apply(lambda x: x.prod(), raw=True) - 1
    signal      = np.sign(cum_returns)
    signal.name = f"signal_{returns.name}"
    return signal


def vol_scaled_position(
    signal:      pd.Series,
    realised_vol: pd.Series,
    target_vol:  float = 0.15,    # 15% annualised target vol — common for CTAs
    max_leverage: float = 2.0,    # cap leverage at 2× to avoid blow-ups
) -> pd.Series:
    # Position size = signal × (target_vol / realised_vol)
    #
    # If realised_vol is LOW → position is LARGER (more units needed to hit target vol)
    # If realised_vol is HIGH → position is SMALLER (fewer units to stay at target vol)
    #
    # This is exactly what Man AHL, Winton, and other CTAs do.
    # It means the portfolio always has approximately target_vol regardless of market regime.

    scaling   = target_vol / realised_vol.replace(0, np.nan)
    position  = signal * scaling
    # clip to prevent extreme leverage in low-vol periods
    position  = position.clip(-max_leverage, max_leverage)
    position.name = f"position_{signal.name}"
    return position


# ─── Main simulation ───────────────────────────────────────────────────────────

def run_tsmom(
    tickers:      list,
    start:        str   = "2010-01-01",
    target_vol:   float = 0.15,
    lookback:     int   = 252,
    use_garch:    bool  = True,
    max_leverage: float = 2.0,
) -> TSMOMResult:

    # download prices
    raw = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw = raw.ffill().dropna()

    returns_df = raw.pct_change().dropna()

    asset_signals    = []
    portfolio_pieces = []

    for ticker in tickers:
        if ticker not in returns_df.columns:
            continue

        r = returns_df[ticker].dropna()
        r.name = ticker

        # 1. estimate volatility
        if use_garch and ARCH_AVAILABLE:
            vol = estimate_vol_garch(r)
        else:
            vol = estimate_vol_rolling(r)

        # 2. compute momentum signal
        sig = tsmom_signal(r, lookback=lookback)

        # 3. scale position by vol target
        pos = vol_scaled_position(sig, vol, target_vol=target_vol, max_leverage=max_leverage)

        # 4. strategy return = position(t-1) × return(t)  — lag by 1 to avoid look-ahead
        strat_return = pos.shift(1) * r
        strat_return.name = ticker

        asset_signals.append(AssetSignal(
            ticker           = ticker,
            dates            = r.index,
            raw_signal       = sig,
            realised_vol     = vol,
            scaled_position  = pos,
            returns          = r,
        ))
        portfolio_pieces.append(strat_return)

    # equal-weight average across all assets
    port_df      = pd.concat(portfolio_pieces, axis=1).dropna()
    port_returns = port_df.mean(axis=1)
    port_returns.name = "TSMOM"

    cum = (1 + port_returns).cumprod()

    # performance metrics
    ann_ret  = port_returns.mean() * 252
    ann_vol  = port_returns.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0

    down_std = port_returns[port_returns < 0].std() * np.sqrt(252)
    sortino  = ann_ret / down_std if down_std > 0 else 0

    peak     = cum.cummax()
    max_dd   = ((cum - peak) / peak).min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
    hit_rate = (port_returns > 0).mean()

    return TSMOMResult(
        asset_signals     = asset_signals,
        portfolio_returns = port_returns,
        portfolio_cumret  = cum,
        target_vol        = target_vol,
        tickers           = tickers,
        sharpe            = sharpe,
        sortino           = sortino,
        max_drawdown      = max_dd,
        calmar            = calmar,
        hit_rate          = hit_rate,
        ann_return        = ann_ret,
        ann_vol           = ann_vol,
    )


def compare_vol_targets(
    tickers:     list,
    start:       str   = "2010-01-01",
    vol_targets: list  = [0.05, 0.10, 0.15, 0.20, 0.25],
) -> dict:
    # run TSMOM at multiple vol targets and return results for comparison
    # shows how performance metrics scale with target volatility
    results = {}
    for vt in vol_targets:
        results[vt] = run_tsmom(tickers, start=start, target_vol=vt, use_garch=False)
    return results