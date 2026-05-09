#signal_bot.py

# Unified Trading Signal Bot — Capstone Project
#
# Architecture:
#   Each signal model outputs a score in [-1, +1]
#   → Weighted ensemble → Vol-targeted position sizing
#   → Drawdown stop overlay → Order generation
#
# Signal models (all implemented here, no external imports needed):
#   1. TSMOM:           sign of 12-month return (Moskowitz et al. 2012)
#   2. Cross-sect mom:  rank within universe, rescaled to [-1, +1]
#   3. Vol regime:      reduce when market vol is elevated vs average
#   4. Trend (SMA):     50-day vs 200-day moving average crossover
#   5. Sentiment:       RSS headline sentiment z-score (optional, soft dependency)
#
# Risk layer:
#   - Vol targeting: position = signal × (target_vol / realised_vol)
#   - Max drawdown stop: zero all positions when portfolio DD > threshold
#   - Per-asset leverage cap
#
# Execution layer:
#   - Alpaca paper trading API (free, no real money)
#   - Dry run mode: shows orders without submitting
#
# This file is self-contained — it does NOT import from your other project files.
# That means it works even if other projects are in different folders.

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from alpaca.trading.client import TradingClient # type: ignore
    from alpaca.trading.requests import MarketOrderRequest # type: ignore
    from alpaca.trading.enums import OrderSide, TimeInForce # type: ignore
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


# ─── signal weights ────────────────────────────────────────────────────────────
# adjust these to change how much each model contributes
DEFAULT_WEIGHTS = {
    "tsmom":      0.30,   # time-series momentum
    "xsmom":      0.25,   # cross-sectional momentum
    "vol_regime": 0.20,   # volatility regime filter
    "trend":      0.15,   # moving average trend
    "sentiment":  0.10,   # news sentiment (0 if not available)
}


# ─── data classes ─────────────────────────────────────────────────────────────

@dataclass
class AssetSignal:
    ticker:           str
    date:             str
    # raw signal scores — each in [-1, +1]
    tsmom:            float = 0.0
    xsmom:            float = 0.0
    vol_regime:       float = 0.0
    trend:            float = 0.0
    sentiment:        float = 0.0
    # derived
    composite:        float = 0.0   # weighted average of above
    vol_scaled:       float = 0.0   # composite × (target_vol / realised_vol)
    final_position:   float = 0.0   # after drawdown stop and leverage cap
    # context
    realised_vol:     float = 0.0
    signal_agreement: float = 0.0   # fraction of signals pointing same direction
    dd_stop_active:   bool  = False


@dataclass
class BotResult:
    tickers:           list
    signals:           dict            # {ticker: AssetSignal} — current signals
    portfolio_returns: pd.Series       # daily P&L of the backtest
    portfolio_cumret:  pd.Series
    weights_history:   pd.DataFrame    # daily position per asset
    composite_history: pd.DataFrame    # daily composite signal per asset
    sharpe:            float
    sortino:           float
    max_drawdown:      float
    ann_return:        float
    ann_vol:           float
    hit_rate:          float
    calmar:            float
    target_vol:        float
    weights_used:      dict


# ─── individual signal functions ──────────────────────────────────────────────

def _tsmom(returns: pd.Series, lookback: int = 252) -> pd.Series:
    # sign of trailing 12-month return — +1 long, -1 short
    cum = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    return np.sign(cum).fillna(0.0)


def _xsmom_universe(returns_df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    # rank each asset within universe, rescale ranks to [-1, +1]
    cum = (1 + returns_df).rolling(lookback).apply(np.prod, raw=True) - 1
    ranks = cum.rank(axis=1, pct=True)     # 0 to 1
    return (ranks * 2 - 1).fillna(0.0)    # -1 to +1


def _vol_regime(market_returns: pd.Series, window: int = 20) -> pd.Series:
    # compare recent vol to long-run vol
    # below average vol → +1 (normal, trade freely)
    # double average vol → 0 (elevated, reduce)
    # triple average vol → -1 (extreme, go defensive)
    rv   = market_returns.rolling(window).std() * np.sqrt(252)
    lrv  = rv.rolling(252).mean()
    ratio = (rv / lrv.replace(0, np.nan)).fillna(1.0)
    return (2.0 - ratio).clip(-1.0, 1.0)


def _trend(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    # simple moving average crossover: fast > slow = +1, else -1
    ma_f = prices.rolling(fast).mean()
    ma_s = prices.rolling(slow).mean()
    return np.sign(ma_f - ma_s).fillna(0.0)


def _vol_scale(signal: pd.Series, returns: pd.Series,
               target_vol: float = 0.15, window: int = 60,
               max_lev: float = 2.0) -> pd.Series:
    rv = returns.rolling(window).std() * np.sqrt(252)
    rv = rv.replace(0, np.nan)
    pos = signal * (target_vol / rv)
    return pos.clip(-max_lev, max_lev).fillna(0.0)


def _drawdown_stop(cum_returns: pd.Series, threshold: float = 0.20) -> pd.Series:
    # 1 = trading normally, 0 = stopped out
    peak = cum_returns.cummax()
    dd   = (cum_returns - peak) / peak
    return (dd > -threshold).astype(float)


def _fetch_sentiment_signal(ticker: str) -> float:
    # lightweight RSS sentiment — returns a single current z-score
    # returns 0 if RSS unavailable (graceful fallback)
    try:
        import xml.etree.ElementTree as ET, re
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        POS  = {"beat","beats","surge","rally","profit","growth","upgrade","record","strong"}
        NEG  = {"miss","misses","fall","plunge","loss","decline","warning","downgrade","hike"}
        scores = []
        for item in root.findall(".//item")[:20]:
            title = (item.findtext("title") or "").lower()
            words = set(re.findall(r'\b\w+\b', title))
            n_pos = len(words & POS)
            n_neg = len(words & NEG)
            total = max(n_pos + n_neg, 1)
            scores.append((n_pos - n_neg) / total)
        if not scores:
            return 0.0
        arr  = np.array(scores)
        mean = arr.mean()
        std  = arr.std()
        return float(mean / std) if std > 0 else float(mean * 5)
    except Exception:
        return 0.0


# ─── main simulation ───────────────────────────────────────────────────────────

def run_bot(
    tickers:       list,
    start:         str   = "2015-01-01",
    target_vol:    float = 0.15,
    max_leverage:  float = 2.0,
    dd_threshold:  float = 0.20,
    weights:       dict  = None,
    include_sentiment: bool = False,
) -> BotResult:

    w = weights or DEFAULT_WEIGHTS.copy()

    # normalise weights to sum to 1
    if not include_sentiment:
        w = {k: v for k, v in w.items() if k != "sentiment"}
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    # download prices — include SPY as market proxy
    all_tickers = list(dict.fromkeys(tickers + ["SPY"]))
    raw = yf.download(all_tickers, start=start, auto_adjust=True,
                      threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(all_tickers[0])
    raw = raw.ffill().dropna()

    spy_ret    = raw["SPY"].pct_change().fillna(0.0) if "SPY" in raw.columns else raw.iloc[:,0].pct_change().fillna(0.0)
    asset_cols = [t for t in tickers if t in raw.columns]
    prices_df  = raw[asset_cols]
    returns_df = prices_df.pct_change().fillna(0.0)

    # compute all signals
    tsmom_df  = pd.DataFrame({t: _tsmom(returns_df[t])       for t in asset_cols})
    xsmom_df  = _xsmom_universe(returns_df)
    vol_reg   = _vol_regime(spy_ret)
    trend_df  = pd.DataFrame({t: _trend(prices_df[t])        for t in asset_cols})

    # build composite signal per asset per day
    composite_df = pd.DataFrame(index=returns_df.index, columns=asset_cols, dtype=float)
    for t in asset_cols:
        ts = tsmom_df[t].reindex(returns_df.index).fillna(0.0)
        xs = xsmom_df[t].reindex(returns_df.index).fillna(0.0) if t in xsmom_df.columns else pd.Series(0.0, index=returns_df.index)
        vr = vol_reg.reindex(returns_df.index).fillna(1.0)
        tr = trend_df[t].reindex(returns_df.index).fillna(0.0)

        composite = (w.get("tsmom", 0)      * ts +
                     w.get("xsmom", 0)      * xs +
                     w.get("vol_regime", 0) * vr +
                     w.get("trend", 0)      * tr)
        composite_df[t] = composite.values

    # vol-scale each position
    scaled_df = pd.DataFrame(index=returns_df.index, columns=asset_cols, dtype=float)
    for t in asset_cols:
        scaled_df[t] = _vol_scale(
            composite_df[t], returns_df[t],
            target_vol=target_vol, max_lev=max_leverage,
        ).values

    # portfolio returns = lagged weights × next-day returns
    port_gross = (scaled_df.shift(1) * returns_df).mean(axis=1)

    # drawdown stop overlay
    cum_gross = (1 + port_gross.fillna(0)).cumprod()
    dd_stop   = _drawdown_stop(cum_gross, threshold=dd_threshold)
    port_ret  = pd.Series(
        port_gross.values * dd_stop.reindex(port_gross.index).fillna(1.0).values,
        index=port_gross.index, dtype=float,
    ).dropna()

    cum_ret  = (1 + port_ret).cumprod()

    # performance
    ann_ret  = float(port_ret.mean()) * 252
    ann_vol  = float(port_ret.std())  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
    neg      = port_ret[port_ret < 0]
    down_std = float(neg.std()) * np.sqrt(252) if len(neg) > 0 else 0.0
    sortino  = ann_ret / down_std if down_std > 0 else 0.0
    peak     = cum_ret.cummax()
    max_dd   = float(((cum_ret - peak) / peak).min())
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    hit_rate = float((port_ret > 0).mean())

    # build current signals
    current_signals = {}
    for t in asset_cols:
        ts_val  = float(tsmom_df[t].dropna().iloc[-1])  if len(tsmom_df[t].dropna())  > 0 else 0.0
        xs_val  = float(xsmom_df[t].dropna().iloc[-1])  if t in xsmom_df.columns and len(xsmom_df[t].dropna()) > 0 else 0.0
        vr_val  = float(vol_reg.dropna().iloc[-1])       if len(vol_reg.dropna())      > 0 else 0.0
        tr_val  = float(trend_df[t].dropna().iloc[-1])  if len(trend_df[t].dropna())  > 0 else 0.0

        sent_val = 0.0
        if include_sentiment:
            try:
                raw_s = _fetch_sentiment_signal(t)
                sent_val = float(np.clip(raw_s / 3.0, -1.0, 1.0))
            except Exception:
                sent_val = 0.0

        comp = (w.get("tsmom",0)*ts_val + w.get("xsmom",0)*xs_val +
                w.get("vol_regime",0)*vr_val + w.get("trend",0)*tr_val +
                w.get("sentiment",0)*sent_val)

        rv_val  = float(returns_df[t].rolling(60).std().dropna().iloc[-1]) * np.sqrt(252) if len(returns_df[t].dropna()) > 60 else target_vol
        scaled  = float(np.clip(comp * (target_vol / rv_val) if rv_val > 0 else 0, -max_leverage, max_leverage))

        cur_dd  = float((cum_ret.iloc[-1] - cum_ret.cummax().iloc[-1]) / cum_ret.cummax().iloc[-1]) if len(cum_ret) > 0 else 0.0
        dd_active = cur_dd < -dd_threshold

        active_signals = [ts_val, xs_val, vr_val, tr_val]
        if include_sentiment:
            active_signals.append(sent_val)
        nonzero = [s for s in active_signals if s != 0.0]
        agreement = abs(sum(np.sign(s) for s in nonzero)) / len(nonzero) if nonzero else 0.0

        current_signals[t] = AssetSignal(
            ticker=t, date=str(datetime.now().date()),
            tsmom=ts_val, xsmom=xs_val, vol_regime=vr_val,
            trend=tr_val, sentiment=sent_val,
            composite=comp, vol_scaled=scaled,
            final_position=0.0 if dd_active else scaled,
            realised_vol=rv_val, signal_agreement=agreement,
            dd_stop_active=dd_active,
        )

    return BotResult(
        tickers           = asset_cols,
        signals           = current_signals,
        portfolio_returns = port_ret,
        portfolio_cumret  = cum_ret,
        weights_history   = scaled_df,
        composite_history = composite_df,
        sharpe            = sharpe,
        sortino           = sortino,
        max_drawdown      = max_dd,
        ann_return        = ann_ret,
        ann_vol           = ann_vol,
        hit_rate          = hit_rate,
        calmar            = calmar,
        target_vol        = target_vol,
        weights_used      = w,
    )


# ─── Alpaca order execution ────────────────────────────────────────────────────

def generate_orders(
    signals:         dict,
    portfolio_value: float = 100_000,
    api_key:         str   = "",
    secret_key:      str   = "",
    dry_run:         bool  = True,
) -> list:
    n      = max(len(signals), 1)
    orders = []

    for ticker, sig in signals.items():
        alloc_value = sig.final_position * (portfolio_value / n)
        side        = "buy" if alloc_value > 0 else ("sell" if alloc_value < 0 else "flat")
        if side == "flat":
            orders.append({"ticker": ticker, "side": "flat", "shares": 0,
                            "position": 0.0, "alloc": 0.0, "status": "no trade"})
            continue

        try:
            info  = yf.Ticker(ticker).fast_info
            price = float(getattr(info, "last_price", None) or
                          getattr(info, "previous_close", 100))
        except Exception:
            price = 100.0

        shares = max(int(abs(alloc_value) / price), 0)

        order = {
            "ticker":   ticker,
            "side":     side,
            "shares":   shares,
            "price_est":f"${price:.2f}",
            "position": round(sig.final_position, 4),
            "alloc":    round(alloc_value, 2),
            "agreement":f"{sig.signal_agreement*100:.0f}%",
            "status":   "dry_run",
        }

        if not dry_run and api_key and secret_key and shares > 0 and ALPACA_AVAILABLE:
            try:
                client = TradingClient(api_key, secret_key, paper=True)
                req    = MarketOrderRequest(
                    symbol        = ticker,
                    qty           = shares,
                    side          = OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    time_in_force = TimeInForce.DAY,
                )
                resp             = client.submit_order(req)
                order["status"]  = f"submitted — {resp.id}"
            except Exception as e:
                order["status"]  = f"error: {e}"

        orders.append(order)

    return orders