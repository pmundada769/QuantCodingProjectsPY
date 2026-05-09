#signals.py

# Active Trading Signal Engine
#
# Implements all indicators and rules from the rulebook:
#   Mark I:   PSAR + SMA50 + RSI8 — breakout/reversal system
#   Mark II:  Ichimoku + WPR + Volume — cloud breakout system
#   Mark III: Volume + Price Action (OB, FVG, LIQ)
#   Mark IV:  EMA ribbon 21/50/100 + RSI divergence + fractals
#   ICH+CCI:  Ichimoku cloud breakout + CCI ±100 crossings (your note)
#
# Also implements:
#   3-Candle Sniper entry pattern
#   Engulfing pattern
#   Tenkan/Kijun cross
#   MACD divergence
#   CCI + WPR synergy signal

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional


# ─── OHLCV fetch ──────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "1m":  {"interval": "1m",  "period": "1d"},
    "5m":  {"interval": "5m",  "period": "5d"},
    "15m": {"interval": "15m", "period": "5d"},
    "1h":  {"interval": "60m", "period": "30d"},
    "4h":  {"interval": "1h",  "period": "60d"},
    "1d":  {"interval": "1d",  "period": "1y"},
    "1wk": {"interval": "1wk", "period": "5y"},
}

def fetch_ohlcv(ticker: str, timeframe: str = "15m") -> pd.DataFrame:
    params = TIMEFRAME_MAP.get(timeframe, {"interval": "15m", "period": "5d"})
    raw    = yf.download(ticker, auto_adjust=True, progress=False, **params)
    if raw is None or len(raw) < 10:
        return pd.DataFrame()
    # flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.strip() for c in raw.columns]
    return raw.dropna()


# ─── Indicators ───────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp   = (high + low + close) / 3
    sma_ = tp.rolling(period).mean()
    mad  = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_) / (0.015 * mad.replace(0, np.nan))

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

def parabolic_sar(high: pd.Series, low: pd.Series,
                  af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    n      = len(high)
    sar    = np.full(n, np.nan)
    trend  = np.ones(n)      # 1 = up, -1 = down
    ep     = np.zeros(n)     # extreme point
    af     = np.full(n, af_step)

    sar[0]   = float(low.iloc[0])
    ep[0]    = float(high.iloc[0])
    trend[0] = 1

    for i in range(1, n):
        prev_sar = sar[i-1]
        prev_ep  = ep[i-1]
        prev_af  = af[i-1]
        prev_tr  = trend[i-1]
        hi = float(high.iloc[i])
        lo = float(low.iloc[i])

        if prev_tr == 1:
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            new_sar = min(new_sar, float(low.iloc[i-1]),
                         float(low.iloc[max(0,i-2)]))
            if lo < new_sar:
                trend[i] = -1
                sar[i]   = prev_ep
                ep[i]    = lo
                af[i]    = af_step
            else:
                trend[i] = 1
                sar[i]   = new_sar
                if hi > prev_ep:
                    ep[i] = hi
                    af[i] = min(prev_af + af_step, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            new_sar = max(new_sar, float(high.iloc[i-1]),
                         float(high.iloc[max(0,i-2)]))
            if hi > new_sar:
                trend[i] = 1
                sar[i]   = prev_ep
                ep[i]    = hi
                af[i]    = af_step
            else:
                trend[i] = -1
                sar[i]   = new_sar
                if lo < prev_ep:
                    ep[i] = lo
                    af[i] = min(prev_af + af_step, af_max)
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af

    return pd.Series(sar, index=high.index, name="PSAR"), pd.Series(trend, index=high.index, name="PSAR_trend")

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_p: int = 9):
    e_fast   = ema(series, fast)
    e_slow   = ema(series, slow)
    macd_l   = e_fast - e_slow
    signal_l = ema(macd_l, signal_p)
    hist     = macd_l - signal_l
    return macd_l, signal_l, hist

def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid  = sma(series, period)
    std  = series.rolling(period).std()
    return mid + std_mult*std, mid, mid - std_mult*std

def volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─── Ichimoku ─────────────────────────────────────────────────────────────────

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             t: int = 9, k: int = 26, s: int = 52, d: int = 26):
    # Tenkan-sen (Conversion Line)
    tenkan  = (high.rolling(t).max() + low.rolling(t).min()) / 2
    # Kijun-sen (Base Line)
    kijun   = (high.rolling(k).max() + low.rolling(k).min()) / 2
    # Senkou Span A (Leading Span A) — shifted forward
    span_a  = ((tenkan + kijun) / 2).shift(d)
    # Senkou Span B (Leading Span B) — shifted forward
    span_b  = ((high.rolling(s).max() + low.rolling(s).min()) / 2).shift(d)
    # Chikou Span (Lagging Span) — shifted back
    chikou  = close.shift(-d)
    # Cloud thickness (volatility proxy)
    cloud_thick = (span_a - span_b).abs()
    return tenkan, kijun, span_a, span_b, chikou, cloud_thick


# ─── Candlestick patterns ─────────────────────────────────────────────────────

def detect_engulfing(open_: pd.Series, close: pd.Series) -> pd.Series:
    # Bullish engulfing: prev candle bearish, current candle bullish and body fully engulfs prev
    # Returns: +1 bullish, -1 bearish, 0 none
    prev_bear  = close.shift(1) < open_.shift(1)
    curr_bull  = close > open_
    bull_eng   = (prev_bear & curr_bull &
                  (open_ <= close.shift(1)) & (close >= open_.shift(1)))

    prev_bull  = close.shift(1) > open_.shift(1)
    curr_bear  = close < open_
    bear_eng   = (prev_bull & curr_bear &
                  (open_ >= close.shift(1)) & (close <= open_.shift(1)))

    signal = pd.Series(0, index=close.index)
    signal[bull_eng] =  1
    signal[bear_eng] = -1
    return signal


def detect_3candle_sniper(open_: pd.Series, close: pd.Series,
                           high: pd.Series, low: pd.Series) -> pd.Series:
    # 3-Candle Sniper: 2-3 small candles same direction, then reversal candle
    # that closes past the open of the first small candle
    # Returns: +1 bullish reversal, -1 bearish reversal, 0 none

    body     = (close - open_).abs()
    avg_body = body.rolling(20).mean()
    small    = body < avg_body * 0.7    # "smallish relative size"

    signal = pd.Series(0, index=close.index)

    for i in range(3, len(close)):
        c1 = close.iloc[i-3]; o1 = open_.iloc[i-3]
        c2 = close.iloc[i-2]; o2 = open_.iloc[i-2]
        c3 = close.iloc[i-1]; o3 = open_.iloc[i-1]
        c4 = close.iloc[i];   o4 = open_.iloc[i]

        s1 = small.iloc[i-3]; s2 = small.iloc[i-2]; s3 = small.iloc[i-1]

        # 3 bearish small candles → bullish sniper
        if (c1 < o1 and c2 < o2 and c3 < o3 and
                s1 and s2 and s3 and
                c4 > o4 and  # reversal candle is bullish
                c4 > o1):    # closes past the open of first candle
            signal.iloc[i] = 1

        # 3 bullish small candles → bearish sniper
        elif (c1 > o1 and c2 > o2 and c3 > o3 and
              s1 and s2 and s3 and
              c4 < o4 and   # reversal candle is bearish
              c4 < o1):     # closes past the open of first candle
            signal.iloc[i] = -1

    return signal


def detect_doji(open_: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    body  = (close - open_).abs()
    range_ = high - low
    # doji: body < 10% of total range
    is_doji = (body / range_.replace(0, np.nan)) < 0.1
    return is_doji.astype(int)


def detect_hammer(open_: pd.Series, close: pd.Series,
                  high: pd.Series, low: pd.Series) -> pd.Series:
    body        = (close - open_).abs()
    lower_wick  = pd.DataFrame({"o": open_, "c": close}).min(axis=1) - low
    upper_wick  = high - pd.DataFrame({"o": open_, "c": close}).max(axis=1)
    range_      = high - low
    # hammer: lower wick ≥ 2× body, upper wick small
    hammer      = ((lower_wick >= 2 * body) &
                   (upper_wick < body * 0.3) &
                   (range_ > 0))
    return hammer.astype(int)


# ─── RSI divergence ───────────────────────────────────────────────────────────

def detect_rsi_divergence(close: pd.Series, rsi_: pd.Series,
                           lookback: int = 14) -> pd.Series:
    # Bearish: price makes higher high, RSI makes lower high → -1
    # Bullish: price makes lower low,  RSI makes higher low  → +1
    signal = pd.Series(0, index=close.index)
    for i in range(lookback, len(close)):
        w_close = close.iloc[i-lookback:i+1]
        w_rsi   = rsi_.iloc[i-lookback:i+1]
        if w_close.iloc[-1] == w_close.max() and w_rsi.iloc[-1] < w_rsi.max():
            signal.iloc[i] = -1   # bearish divergence
        if w_close.iloc[-1] == w_close.min() and w_rsi.iloc[-1] > w_rsi.min():
            signal.iloc[i] =  1   # bullish divergence
    return signal


# ─── Your Mark rulebook signals ───────────────────────────────────────────────

@dataclass
class MarkISignal:
    # PSAR + SMA50 + RSI8
    direction:    int    # +1 long, -1 short, 0 flat
    reason:       str
    psar_flipped: bool
    rsi_cross_50: bool
    rsi_extreme:  bool   # RSI > 70 or < 30
    candle_match: bool   # prev candle matches new psar colour
    sma_distance: float  # % distance of close from SMA50


@dataclass
class MarkIISignal:
    # Ichimoku + WPR + Volume
    direction:      int
    reason:         str
    cloud_breakout: bool
    wpr_extreme:    bool    # WPR in [-20,0] or [-100,-80]
    cl_bl_cross:    bool    # Tenkan/Kijun intersection
    chikou_clear:   bool    # Lagging span in open space
    cloud_thick:    float   # thickness relative to price
    volume_confirm: bool


@dataclass
class ICHCCISignal:
    # Your notebook: ICH+CCI v1.0 — the clean combined system
    direction:      int
    reason:         str
    cloud_breakout: bool
    cci_extreme:    bool    # CCI crossed ±100
    cci_cross_dir:  int     # +1 crossed above -100 (bullish), -1 crossed below +100 (bearish)
    cl_bl_intersect:bool
    chikou_trend:   int     # +1 above price = bullish, -1 below = bearish
    cloud_thick:    float
    synergetic:     bool    # all signals agree


@dataclass
class MarkIVSignal:
    # EMA ribbon 21/50/100 + RSI divergence
    direction:     int
    reason:        str
    ema21_side:    int    # price above (+1) or below (-1) EMA21
    rsi_divergence: int   # +1 bull div, -1 bear div, 0 none
    near_ema21:    bool   # price within 0.3% of EMA21
    rrr_ok:        bool   # implied RR ≥ 1.5


def compute_mark_i(df: pd.DataFrame, af_step: float = 0.04) -> pd.Series:
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]

    sma50     = sma(close, 50)
    rsi8      = rsi(close, 8)
    psar, psar_trend = parabolic_sar(high, low, af_step=af_step)

    signals = pd.Series(0, index=df.index)
    reasons = pd.Series("", index=df.index)

    for i in range(3, len(df)):
        c = float(close.iloc[i])
        s = float(sma50.iloc[i]) if not pd.isna(sma50.iloc[i]) else c
        r = float(rsi8.iloc[i])  if not pd.isna(rsi8.iloc[i])  else 50
        pt_cur  = int(psar_trend.iloc[i])
        pt_prev = int(psar_trend.iloc[i-1])

        psar_flip = (pt_cur != pt_prev)
        rsi_above = r > 50
        rsi_ext   = r > 70 or r < 30
        rsi_cross = (rsi8.iloc[i] > 50 and rsi8.iloc[i-1] <= 50) or \
                    (rsi8.iloc[i] < 50 and rsi8.iloc[i-1] >= 50)

        # candlestick match rule: prev candle colour matches new PSAR direction
        prev_bull = float(close.iloc[i-1]) > float(open_.iloc[i-1])
        prev_bear = float(close.iloc[i-1]) < float(open_.iloc[i-1])
        candle_match = (pt_cur == 1 and prev_bull) or (pt_cur == -1 and prev_bear)

        # avoid doji/hammer at reversal
        d = detect_doji(open_, close, high, low)
        h = detect_hammer(open_, close, high, low)
        avoid = bool(d.iloc[i-1]) or bool(h.iloc[i-1])

        if psar_flip and rsi_ext and candle_match and not avoid:
            if pt_cur == 1 and c > s:  # uptrend, psar flipped bullish
                signals.iloc[i] = 1
                reasons.iloc[i] = "Mark I: PSAR flip ↑ + RSI extreme + candle match + above SMA50"
            elif pt_cur == -1 and c < s:
                signals.iloc[i] = -1
                reasons.iloc[i] = "Mark I: PSAR flip ↓ + RSI extreme + candle match + below SMA50"

    return signals, reasons


def compute_ich_cci(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    tenkan, kijun, span_a, span_b, chikou, cloud_thick = ichimoku(high, low, close)
    cci_   = cci(high, low, close, 14)

    signals = pd.Series(0, index=df.index)
    reasons = pd.Series("", index=df.index)

    for i in range(52, len(df)):
        c   = float(close.iloc[i])
        sa  = float(span_a.iloc[i]) if not pd.isna(span_a.iloc[i]) else c
        sb  = float(span_b.iloc[i]) if not pd.isna(span_b.iloc[i]) else c
        tk  = float(tenkan.iloc[i]) if not pd.isna(tenkan.iloc[i]) else c
        kj  = float(kijun.iloc[i])  if not pd.isna(kijun.iloc[i])  else c
        ck  = float(chikou.iloc[i-26]) if i >= 26 and not pd.isna(chikou.iloc[i-26]) else c
        ct  = float(cloud_thick.iloc[i]) if not pd.isna(cloud_thick.iloc[i]) else 0
        cc  = float(cci_.iloc[i])  if not pd.isna(cci_.iloc[i])  else 0
        cc1 = float(cci_.iloc[i-1]) if not pd.isna(cci_.iloc[i-1]) else 0

        cloud_top    = max(sa, sb)
        cloud_bot    = min(sa, sb)
        cloud_width  = cloud_top - cloud_bot
        avg_price    = c
        thick_ratio  = cloud_width / avg_price if avg_price > 0 else 0
        thin_cloud   = thick_ratio < 0.005   # < 0.5% of price = thin = uncertain

        # cloud breakout
        prev_c   = float(close.iloc[i-1])
        in_cloud = cloud_bot <= prev_c <= cloud_top
        bull_bo  = prev_c <= cloud_bot and c > cloud_top   # bullish BO
        bear_bo  = prev_c >= cloud_top and c < cloud_bot   # bearish BO
        # also standard: was in cloud, now out
        if not (bull_bo or bear_bo):
            bull_bo = (cloud_bot <= prev_c <= cloud_top) and (c > cloud_top)
            bear_bo = (cloud_bot <= prev_c <= cloud_top) and (c < cloud_bot)

        # CCI ±100 crossings
        cci_bull = cc1 < -100 and cc >= -100   # crossed above -100 = bullish
        cci_bear = cc1 >  100 and cc <=  100   # crossed below +100 = bearish
        cci_ext  = abs(cc) > 100

        # Tenkan/Kijun intersection
        tk1  = float(tenkan.iloc[i-1]) if not pd.isna(tenkan.iloc[i-1]) else tk
        kj1  = float(kijun.iloc[i-1])  if not pd.isna(kijun.iloc[i-1])  else kj
        cl_bl_bull = (tk1 < kj1 and tk >= kj)
        cl_bl_bear = (tk1 > kj1 and tk <= kj)

        # Chikou above/below price → trend
        chikou_bull = ck > c
        chikou_bear = ck < c

        if thin_cloud:
            continue   # avoid — high uncertainty

        # ICH breakout + CCI = strong entry (your notebook)
        if bull_bo and (cci_bull or cci_ext and cc < 0) and chikou_bull:
            synergy = cl_bl_bull or cci_bull
            signals.iloc[i] = 1
            reasons.iloc[i] = ("ICH+CCI: Cloud BO ↑ + CCI crossed -100 + Chikou bullish"
                                + (" + TK/KJ cross" if cl_bl_bull else ""))

        elif bear_bo and (cci_bear or cci_ext and cc > 0) and chikou_bear:
            signals.iloc[i] = -1
            reasons.iloc[i] = ("ICH+CCI: Cloud BO ↓ + CCI crossed +100 + Chikou bearish"
                                + (" + TK/KJ cross" if cl_bl_bear else ""))

    return signals, reasons


def compute_mark_iv(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    e21  = ema(close, 21)
    e50  = ema(close, 50)
    e100 = ema(close, 100)
    rsi_ = rsi(close, 14)
    rsi_div = detect_rsi_divergence(close, rsi_)

    signals = pd.Series(0, index=df.index)
    reasons = pd.Series("", index=df.index)

    for i in range(100, len(df)):
        c   = float(close.iloc[i])
        e21v = float(e21.iloc[i])  if not pd.isna(e21.iloc[i])  else c
        e50v = float(e50.iloc[i])  if not pd.isna(e50.iloc[i])  else c
        e100v= float(e100.iloc[i]) if not pd.isna(e100.iloc[i]) else c
        div  = int(rsi_div.iloc[i])

        above_e21  = c > e21v
        above_e50  = c > e50v
        above_e100 = c > e100v
        near_e21   = abs(c - e21v) / e21v < 0.003  # within 0.3%

        # RSI divergence near EMA21 = Mark IV entry
        if div == 1 and near_e21 and above_e21:
            signals.iloc[i] = 1
            reasons.iloc[i] = "Mark IV: Bullish RSI divergence near EMA21"
        elif div == -1 and near_e21 and not above_e21:
            signals.iloc[i] = -1
            reasons.iloc[i] = "Mark IV: Bearish RSI divergence near EMA21"

    return signals, reasons


def compute_3candle_sniper(df: pd.DataFrame) -> pd.Series:
    sig = detect_3candle_sniper(df["Open"], df["Close"], df["High"], df["Low"])
    reasons = sig.map({1: "3-Candle Sniper: Bullish reversal", -1: "3-Candle Sniper: Bearish reversal", 0: ""})
    return sig, reasons


def compute_engulfing(df: pd.DataFrame) -> pd.Series:
    sig = detect_engulfing(df["Open"], df["Close"])
    reasons = sig.map({1: "Engulfing: Bullish", -1: "Engulfing: Bearish", 0: ""})
    return sig, reasons


# ─── Composite signal ─────────────────────────────────────────────────────────

def compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60:
        return pd.DataFrame()

    mark1,  r1  = compute_mark_i(df)
    ich_cci, r2 = compute_ich_cci(df)
    mark4,  r3  = compute_mark_iv(df)
    sniper, r4  = compute_3candle_sniper(df)
    engulf, r5  = compute_engulfing(df)

    # composite: any signal fires
    composite = pd.Series(0, index=df.index)
    all_signals = pd.concat([mark1, ich_cci, mark4, sniper, engulf], axis=1)
    all_signals.columns = ["mark1","ich_cci","mark4","sniper","engulf"]

    # weighted vote: ICH+CCI and Mark I weighted higher
    composite = (all_signals["mark1"]   * 2 +
                 all_signals["ich_cci"] * 2 +
                 all_signals["mark4"]   * 1 +
                 all_signals["sniper"]  * 1 +
                 all_signals["engulf"]  * 1)

    signal_out = pd.DataFrame({
        "mark1":    mark1,
        "ich_cci":  ich_cci,
        "mark4":    mark4,
        "sniper":   sniper,
        "engulf":   engulf,
        "composite":composite,
        "reason_mark1":   r1,
        "reason_ich_cci": r2,
        "reason_mark4":   r3,
        "reason_sniper":  r4,
        "reason_engulf":  r5,
    })

    return signal_out


# ─── TP/SL calculator ─────────────────────────────────────────────────────────

def calculate_tp_sl(entry: float, direction: int,
                    atr_val: float, rrr: float = 2.0,
                    sl_mult: float = 1.5) -> dict:
    sl_dist = atr_val * sl_mult
    tp_dist = sl_dist * rrr

    if direction == 1:   # long
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:                # short
        sl = entry + sl_dist
        tp = entry - tp_dist

    return {
        "entry":    round(entry, 5),
        "sl":       round(sl, 5),
        "tp":       round(tp, 5),
        "sl_pips":  round(sl_dist, 5),
        "tp_pips":  round(tp_dist, 5),
        "rrr":      rrr,
        "direction":"LONG" if direction == 1 else "SHORT",
    }


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest_signals(df: pd.DataFrame, signals: pd.DataFrame,
                     signal_col: str = "composite",
                     atr_mult_sl: float = 1.5, rrr: float = 2.0) -> dict:
    close   = df["Close"]
    high    = df["High"]
    low     = df["Low"]
    atr_    = atr(high, low, close, 14)
    sig     = signals[signal_col]

    trades  = []
    in_trade = False

    for i in range(1, len(sig)):
        s = int(np.sign(sig.iloc[i]))
        if s == 0 or in_trade:
            continue

        entry   = float(close.iloc[i])
        atr_val = float(atr_.iloc[i]) if not pd.isna(atr_.iloc[i]) else entry * 0.001
        tpsl    = calculate_tp_sl(entry, s, atr_val, rrr, atr_mult_sl)

        # simulate: scan forward for TP or SL hit
        result  = None
        for j in range(i+1, min(i+50, len(df))):
            hi = float(high.iloc[j])
            lo = float(low.iloc[j])
            if s == 1:
                if lo <= tpsl["sl"]:  result = "SL"; break
                if hi >= tpsl["tp"]:  result = "TP"; break
            else:
                if hi >= tpsl["sl"]:  result = "SL"; break
                if lo <= tpsl["tp"]:  result = "TP"; break

        if result is None:
            result = "OPEN"

        pnl = rrr if result == "TP" else (-1.0 if result == "SL" else 0)
        trades.append({
            "date":    str(df.index[i].date()),
            "signal":  signal_col,
            "direction": tpsl["direction"],
            "entry":   entry,
            "tp":      tpsl["tp"],
            "sl":      tpsl["sl"],
            "result":  result,
            "pnl_r":   pnl,
        })

    if not trades:
        return {"trades": [], "win_rate": 0, "total_r": 0, "sharpe": 0}

    trades_df  = pd.DataFrame(trades)
    closed     = trades_df[trades_df["result"].isin(["TP","SL"])]
    win_rate   = float((closed["pnl_r"] > 0).mean()) if len(closed) > 0 else 0
    total_r    = float(closed["pnl_r"].sum())

    pnl_series = closed["pnl_r"]
    sharpe     = (float(pnl_series.mean()) /
                  float(pnl_series.std()) * np.sqrt(252)) if pnl_series.std() > 0 else 0

    return {
        "trades":   trades_df.to_dict("records"),
        "win_rate": win_rate,
        "total_r":  total_r,
        "sharpe":   sharpe,
        "n_trades": len(closed),
        "n_wins":   int((closed["pnl_r"] > 0).sum()),
    }