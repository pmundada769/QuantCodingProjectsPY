#signals.py — indicator library for Active Trading Dashboard

import numpy as np
import pandas as pd
import yfinance as yf

TIMEFRAME_MAP = {
    "1m":  {"interval": "1m",  "period": "1d"},
    "5m":  {"interval": "5m",  "period": "5d"},
    "15m": {"interval": "15m", "period": "5d"},
    "30m": {"interval": "30m", "period": "10d"},
    "1h":  {"interval": "60m", "period": "60d"},
    "4h":  {"interval": "1h",  "period": "120d"},
    "1d":  {"interval": "1d",  "period": "2y"},
    "1wk": {"interval": "1wk", "period": "5y"},
}

# common FX pairs — yfinance needs =X suffix
FX_ALIASES = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X", "XAUUSD": "GC=F",     "GOLD":   "GC=F",
    "OIL":    "CL=F",     "SILVER": "SI=F",
}


def fetch_ohlcv(ticker: str, timeframe: str = "15m") -> pd.DataFrame:
    # normalise ticker
    t = ticker.upper().strip()
    t = FX_ALIASES.get(t, t)
    if t.endswith("USD") and "=" not in t and len(t) == 6:
        t = t + "=X"

    params = dict(TIMEFRAME_MAP.get(timeframe, {"interval": "15m", "period": "5d"}))

    try:
        raw = yf.download(t, auto_adjust=True, progress=False,
                          threads=False, **params)
    except Exception as e:
        print(f"[signals] Download failed for {t}: {e}")
        return pd.DataFrame()

    if raw is None or len(raw) < 3:
        return pd.DataFrame()

    # flatten MultiIndex columns (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [str(c).strip() for c in raw.columns]

    # normalise column names
    rename = {}
    for c in raw.columns:
        cl = c.lower()
        if   "open"  in cl: rename[c] = "Open"
        elif "high"  in cl: rename[c] = "High"
        elif "low"   in cl: rename[c] = "Low"
        elif "close" in cl: rename[c] = "Close"
        elif "vol"   in cl: rename[c] = "Volume"
    raw = raw.rename(columns=rename)

    needed = ["Open", "High", "Low", "Close"]
    if not all(c in raw.columns for c in needed):
        return pd.DataFrame()

    if "Volume" not in raw.columns:
        raw["Volume"] = 0

    # drop NaN rows
    raw = raw.dropna(subset=needed)

    # drop zero-price rows (gaps yfinance sometimes inserts)
    raw = raw[(raw["Open"] > 0) & (raw["Close"] > 0)]

    return raw


# ── core indicators ────────────────────────────────────────────────────────────

def sma(s, n):   return s.rolling(n).mean()
def ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def wma(s, n):
    w = np.arange(1, n+1)
    return s.rolling(n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))

def cci(h, l, c, n=20):
    tp  = (h+l+c)/3
    m   = tp.rolling(n).mean()
    md  = tp.rolling(n).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True)
    return (tp-m)/(0.015*md.replace(0, np.nan))

def williams_r(h, l, c, n=14):
    hh = h.rolling(n).max()
    ll = l.rolling(n).min()
    return -100*(hh-c)/(hh-ll).replace(0, np.nan)

def stochastic(h, l, c, k=14, d=3):
    ll = l.rolling(k).min()
    hh = h.rolling(k).max()
    k_ = 100*(c-ll)/(hh-ll).replace(0, np.nan)
    d_ = k_.rolling(d).mean()
    return k_, d_

def macd(s, fast=12, slow=26, sig=9):
    m = ema(s, fast) - ema(s, slow)
    signal = ema(m, sig)
    return m, signal, m-signal

def bollinger(s, n=20, std=2.0):
    m = sma(s, n)
    sd = s.rolling(n).std()
    return m+std*sd, m, m-std*sd

def atr(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def vwap(h, l, c, vol):
    tp = (h+l+c)/3
    return (tp*vol).cumsum() / vol.cumsum().replace(0, np.nan)

def supertrend(h, l, c, n=10, mult=3.0):
    a = atr(h, l, c, n)
    hl2 = (h+l)/2
    upper = hl2 + mult*a
    lower = hl2 - mult*a
    st  = pd.Series(np.nan, index=c.index)
    dir_ = pd.Series(1, index=c.index)
    for i in range(1, len(c)):
        prev_upper = upper.iloc[i-1]
        prev_lower = lower.iloc[i-1]
        upper.iloc[i] = upper.iloc[i] if upper.iloc[i] < prev_upper or c.iloc[i-1] > prev_upper else prev_upper
        lower.iloc[i] = lower.iloc[i] if lower.iloc[i] > prev_lower or c.iloc[i-1] < prev_lower else prev_lower
        if dir_.iloc[i-1] == 1:
            dir_.iloc[i] = -1 if c.iloc[i] < lower.iloc[i] else 1
        else:
            dir_.iloc[i] = 1 if c.iloc[i] > upper.iloc[i] else -1
        st.iloc[i] = lower.iloc[i] if dir_.iloc[i] == 1 else upper.iloc[i]
    return st, dir_

def parabolic_sar(h, l, af_step=0.02, af_max=0.2):
    n = len(h)
    sar   = np.full(n, np.nan)
    trend = np.ones(n)
    ep    = np.zeros(n)
    af    = np.full(n, af_step)
    sar[0] = float(l.iloc[0]); ep[0] = float(h.iloc[0])
    for i in range(1, n):
        hi = float(h.iloc[i]); lo = float(l.iloc[i])
        prev_s = sar[i-1]; prev_e = ep[i-1]; prev_a = af[i-1]; prev_t = trend[i-1]
        if prev_t == 1:
            new_s = prev_s + prev_a*(prev_e-prev_s)
            new_s = min(new_s, float(l.iloc[i-1]), float(l.iloc[max(0,i-2)]))
            if lo < new_s:
                trend[i]=-1; sar[i]=prev_e; ep[i]=lo; af[i]=af_step
            else:
                trend[i]=1; sar[i]=new_s
                ep[i]=max(prev_e,hi); af[i]=min(prev_a+(af_step if hi>prev_e else 0), af_max)
        else:
            new_s = prev_s + prev_a*(prev_e-prev_s)
            new_s = max(new_s, float(h.iloc[i-1]), float(h.iloc[max(0,i-2)]))
            if hi > new_s:
                trend[i]=1; sar[i]=prev_e; ep[i]=hi; af[i]=af_step
            else:
                trend[i]=-1; sar[i]=new_s
                ep[i]=min(prev_e,lo); af[i]=min(prev_a+(af_step if lo<prev_e else 0), af_max)
    return (pd.Series(sar, index=h.index),
            pd.Series(trend, index=h.index))

def ichimoku(h, l, c, t=9, k=26, s=52, d=26):
    tenkan = (h.rolling(t).max() + l.rolling(t).min())/2
    kijun  = (h.rolling(k).max() + l.rolling(k).min())/2
    span_a = ((tenkan+kijun)/2).shift(d)
    span_b = ((h.rolling(s).max()+l.rolling(s).min())/2).shift(d)
    chikou = c.shift(-d)
    return tenkan, kijun, span_a, span_b, chikou

def mfi(h, l, c, vol, n=14):
    tp  = (h+l+c)/3
    rmf = tp*vol
    pos = rmf.where(tp > tp.shift(1), 0)
    neg = rmf.where(tp < tp.shift(1), 0)
    mfr = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
    return 100 - 100/(1+mfr)

def adx(h, l, c, n=14):
    up   = h.diff()
    down = -l.diff()
    pdm  = up.where((up > down) & (up > 0), 0)
    ndm  = down.where((down > up) & (down > 0), 0)
    a    = atr(h, l, c, n)
    pdi  = 100*pdm.ewm(alpha=1/n, adjust=False).mean()/a.replace(0, np.nan)
    ndi  = 100*ndm.ewm(alpha=1/n, adjust=False).mean()/a.replace(0, np.nan)
    dx   = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), pdi, ndi

def fibonacci_levels(high_val: float, low_val: float) -> dict:
    diff = high_val - low_val
    return {
        "0%":     high_val,
        "23.6%":  high_val - 0.236*diff,
        "38.2%":  high_val - 0.382*diff,
        "50%":    high_val - 0.500*diff,
        "61.8%":  high_val - 0.618*diff,
        "78.6%":  high_val - 0.786*diff,
        "100%":   low_val,
        "127.2%": low_val  - 0.272*diff,
        "161.8%": low_val  - 0.618*diff,
    }


# ── candlestick patterns ───────────────────────────────────────────────────────

def pattern_engulfing(o, c):
    pb = c.shift(1) < o.shift(1)
    cb = c > o
    bull = pb & cb & (o <= c.shift(1)) & (c >= o.shift(1))
    cb2  = c < o
    pb2  = c.shift(1) > o.shift(1)
    bear = pb2 & cb2 & (o >= c.shift(1)) & (c <= o.shift(1))
    s = pd.Series(0, index=c.index)
    s[bull] = 1; s[bear] = -1
    return s

def pattern_3candle_sniper(o, c, h, l):
    body     = (c-o).abs()
    avg_body = body.rolling(20).mean()
    small    = body < avg_body*0.7
    sig = pd.Series(0, index=c.index)
    for i in range(3, len(c)):
        c1,o1 = c.iloc[i-3],o.iloc[i-3]
        c2,o2 = c.iloc[i-2],o.iloc[i-2]
        c3,o3 = c.iloc[i-1],o.iloc[i-1]
        c4,o4 = c.iloc[i],  o.iloc[i]
        s1,s2,s3 = small.iloc[i-3],small.iloc[i-2],small.iloc[i-1]
        if c1<o1 and c2<o2 and c3<o3 and s1 and s2 and s3 and c4>o4 and c4>o1:
            sig.iloc[i] = 1
        elif c1>o1 and c2>o2 and c3>o3 and s1 and s2 and s3 and c4<o4 and c4<o1:
            sig.iloc[i] = -1
    return sig

def pattern_doji(o, c, h, l):
    body  = (c-o).abs()
    range_ = (h-l).replace(0, np.nan)
    return ((body/range_) < 0.1).astype(int)

def pattern_hammer(o, c, h, l):
    body = (c-o).abs()
    lo_  = pd.DataFrame({"o":o,"c":c}).min(axis=1)
    hi_  = pd.DataFrame({"o":o,"c":c}).max(axis=1)
    lw   = lo_ - l
    uw   = h - hi_
    return ((lw >= 2*body) & (uw < body*0.3) & ((h-l)>0)).astype(int)


# ── ICH+CCI signal (your notebook) ────────────────────────────────────────────

def compute_ich_cci_signal(df: pd.DataFrame) -> pd.Series:
    c = df["Close"]; h = df["High"]; l = df["Low"]
    tenkan, kijun, span_a, span_b, chikou = ichimoku(h, l, c)
    cci_ = cci(h, l, c, 14)
    sig  = pd.Series(0, index=df.index)
    for i in range(52, len(df)):
        sa = float(span_a.iloc[i]) if pd.notna(span_a.iloc[i]) else float(c.iloc[i])
        sb = float(span_b.iloc[i]) if pd.notna(span_b.iloc[i]) else float(c.iloc[i])
        cv = float(c.iloc[i]); cv1 = float(c.iloc[i-1])
        cc = float(cci_.iloc[i]) if pd.notna(cci_.iloc[i]) else 0
        cc1= float(cci_.iloc[i-1]) if pd.notna(cci_.iloc[i-1]) else 0
        top = max(sa,sb); bot = min(sa,sb)
        thick = (top-bot)/cv if cv>0 else 0
        if thick < 0.005: continue  # thin cloud — skip
        bull_bo = cv1 <= top and cv > top
        bear_bo = cv1 >= bot and cv < bot
        ck_idx = max(0, i-26)
        ck = float(chikou.iloc[ck_idx]) if pd.notna(chikou.iloc[ck_idx]) else cv
        chikou_bull = ck > cv; chikou_bear = ck < cv
        cci_bull = cc1 < -100 and cc >= -100
        cci_bear = cc1 > 100  and cc <=  100
        if bull_bo and (cci_bull or cc < -100) and chikou_bull:
            sig.iloc[i] = 1
        elif bear_bo and (cci_bear or cc > 100) and chikou_bear:
            sig.iloc[i] = -1
    return sig


def compute_mark1_signal(df: pd.DataFrame, af: float = 0.04) -> pd.Series:
    c = df["Close"]; h = df["High"]; l = df["Low"]; o = df["Open"]
    sma50 = sma(c, 50); rsi8 = rsi(c, 8)
    psar_, ptrd = parabolic_sar(h, l, af_step=af)
    doji_  = pattern_doji(o, c, h, l)
    hammer_= pattern_hammer(o, c, h, l)
    sig    = pd.Series(0, index=df.index)
    for i in range(3, len(df)):
        pt = int(ptrd.iloc[i]); pt1 = int(ptrd.iloc[i-1])
        if pt == pt1: continue  # no flip
        r = float(rsi8.iloc[i]) if pd.notna(rsi8.iloc[i]) else 50
        rsi_ok = r > 70 or r < 30 or (rsi8.iloc[i] > 50) != (rsi8.iloc[i-1] > 50)
        prev_bull = float(c.iloc[i-1]) > float(o.iloc[i-1])
        candle_ok = (pt == 1 and prev_bull) or (pt == -1 and not prev_bull)
        avoid = bool(doji_.iloc[i-1]) or bool(hammer_.iloc[i-1])
        if rsi_ok and candle_ok and not avoid:
            sig.iloc[i] = pt
    return sig


def backtest(df, signals, rrr=2.0, atr_mult=1.5):
    c = df["Close"]; h = df["High"]; l = df["Low"]
    a = atr(h, l, c, 14)
    trades = []
    for i in range(1, len(signals)):
        s = int(np.sign(signals.iloc[i]))
        if s == 0: continue
        entry = float(c.iloc[i])
        av    = float(a.iloc[i]) if pd.notna(a.iloc[i]) else entry*0.001
        sl    = entry - av*atr_mult*s
        tp    = entry + av*atr_mult*rrr*s
        res   = "OPEN"; pnl = 0
        for j in range(i+1, min(i+60, len(df))):
            hi = float(h.iloc[j]); lo = float(l.iloc[j])
            if s == 1:
                if lo <= sl: res="SL"; pnl=-1; break
                if hi >= tp: res="TP"; pnl=rrr; break
            else:
                if hi >= sl: res="SL"; pnl=-1; break
                if lo <= tp: res="TP"; pnl=rrr; break
        trades.append({"i":i,"dir":"L" if s==1 else "S","entry":entry,
                        "tp":round(tp,5),"sl":round(sl,5),"result":res,"R":pnl})
    return pd.DataFrame(trades) if trades else pd.DataFrame()