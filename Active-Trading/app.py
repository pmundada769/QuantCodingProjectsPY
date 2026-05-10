#app.py — Active Trading Dashboard
# TradingView-style chart: clean price chart + add any indicator you want
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signals import (
    fetch_ohlcv, TIMEFRAME_MAP,
    sma, ema, wma, rsi, cci, williams_r, stochastic,
    macd, bollinger, atr, vwap, supertrend, parabolic_sar, ichimoku,
    mfi, adx, fibonacci_levels,
    pattern_engulfing, pattern_3candle_sniper, pattern_doji, pattern_hammer,
    compute_ich_cci_signal, compute_mark1_signal, backtest,
)

st.set_page_config(page_title="Active Trading", page_icon="📡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'Exo 2',sans-serif;background:#020508;color:#A8D8C0;}
h1{font-family:'Share Tech Mono',monospace!important;color:#00FF88!important;font-size:1.5rem!important;}
h2,h3{color:#1A4A30!important;font-weight:400!important;}
[data-testid="metric-container"]{background:#081410;border:1px solid #0C2018;border-radius:3px;padding:8px 12px;}
[data-testid="metric-container"] label{font-family:'Share Tech Mono',monospace!important;font-size:0.55rem!important;color:#0A2018!important;text-transform:uppercase;letter-spacing:.1em;}
[data-testid="metric-container"] [data-testid="metric-value"]{font-family:'Share Tech Mono',monospace!important;font-size:.95rem!important;color:#00FF88!important;}
[data-testid="stSidebar"]{background:#020408;border-right:1px solid #0C2018;}
.stTabs [data-baseweb="tab"]{font-family:'Share Tech Mono',monospace;font-size:.62rem;background:#081410;border:1px solid #0C2018;color:#0A2018;border-radius:3px;}
.stTabs [aria-selected="true"]{background:#0C2018!important;border-color:#00FF88!important;color:#00FF88!important;}
hr{border-color:#0C2018!important;}
.stMultiSelect [data-baseweb="tag"]{background:#0C2018;color:#00FF88;}
</style>
""", unsafe_allow_html=True)

G="#00FF88"; R="#FF4466"; GOLD="#FFD700"; CY="#00CCFF"; OR="#FF8C00"
PU="#CC88FF"; MU="#1A4A30"; BG="#020508"; GR="#0C2018"; TX="#A8D8C0"

def _base(h=500):
    return dict(plot_bgcolor=BG, paper_bgcolor=BG,
                font=dict(color=TX, size=10),
                margin=dict(l=50,r=20,t=40,b=30),
                legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor=GR,font=dict(size=9)),
                height=h)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Active Trading")
    st.markdown("---")

    ticker = st.text_input("Symbol", value="EURUSD=X",
        help="FX: EURUSD=X GBPUSD=X  |  Gold: GC=F  |  Crypto: BTC-USD ETH-USD  |  Stocks: AAPL NVDA  |  Index: ^GSPC ^NDX")
    tf = st.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=2)
    n_candles = st.slider("Candles to show", 30, 500, 150)

    st.markdown("---")
    st.markdown("#### 📊 Add Indicators")
    st.markdown("<span style='font-size:.7rem;color:#336655;'>Overlay (on price chart)</span>", unsafe_allow_html=True)
    overlays = st.multiselect("Overlay indicators", [
        "SMA 20","SMA 50","SMA 100","SMA 200",
        "EMA 9","EMA 21","EMA 50","EMA 100","EMA 200",
        "WMA 20",
        "Bollinger Bands (20,2)","Bollinger Bands (20,3)",
        "VWAP",
        "PSAR (0.02)","PSAR (0.04)",
        "SuperTrend (10,3)",
        "Ichimoku Cloud",
        "Fibonacci Levels",
    ], default=["EMA 21","Ichimoku Cloud"])

    st.markdown("<span style='font-size:.7rem;color:#336655;'>Panels (below chart)</span>", unsafe_allow_html=True)
    panels = st.multiselect("Panel indicators", [
        "RSI (8)","RSI (14)",
        "CCI (14)","CCI (20)","CCI (50)",
        "Williams %R",
        "Stochastic (14,3)",
        "MACD (12,26,9)",
        "Volume",
        "ATR (14)",
        "MFI (14)",
        "ADX (14)",
    ], default=["RSI (14)","CCI (14)","Volume"])

    st.markdown("---")
    st.markdown("#### 🎯 Signal Strategies")
    show_mark1  = st.checkbox("Mark I  (PSAR+SMA50+RSI8)", value=True)
    show_ichcci = st.checkbox("ICH+CCI  (your notebook)", value=True)
    show_engulf = st.checkbox("Engulfing pattern", value=True)
    show_sniper = st.checkbox("3-Candle Sniper", value=False)

    st.markdown("---")
    st.markdown("#### TP/SL")
    rrr     = st.select_slider("RRR", [1.0,1.5,2.0,2.5,3.0], value=2.0)
    sl_atr  = st.slider("SL × ATR", 0.5, 3.0, 1.5, step=0.25)
    show_tpsl = st.checkbox("Draw TP/SL on chart", value=True)

    st.markdown("---")
    st.caption("All free · No API key · Yahoo Finance\nRefreshes on symbol change")

# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load(ticker, tf):
    return fetch_ohlcv(ticker, tf)

with st.spinner(f"Loading {ticker} {tf}..."):
    df = load(ticker, tf)

if df is None or len(df) < 10:
    st.error(f"No data for **{ticker}**.")
    st.markdown("""
**Common symbols:**
| Type | Examples |
|---|---|
| FX | `EURUSD=X` `GBPUSD=X` `GBPJPY=X` `USDJPY=X` `AUDUSD=X` |
| Gold/Oil | `GC=F` `CL=F` `SI=F` |
| Crypto | `BTC-USD` `ETH-USD` `SOL-USD` |
| US Stocks | `AAPL` `NVDA` `TSLA` `MSFT` |
| Indices | `^GSPC` (S&P) `^NDX` (Nasdaq) `^FTSE` (FTSE 100) |
| You can also type without =X: `GBPJPY` `EURUSD` |
""")
    st.stop()

df = df.tail(n_candles).copy()
c = df["Close"]; h = df["High"]; l = df["Low"]; o = df["Open"]
vol = df["Volume"]

# ── compute selected indicators ────────────────────────────────
computed = {}

for ind in overlays:
    try:
        if ind == "SMA 20":      computed[ind] = sma(c,20)
        elif ind == "SMA 50":    computed[ind] = sma(c,50)
        elif ind == "SMA 100":   computed[ind] = sma(c,100)
        elif ind == "SMA 200":   computed[ind] = sma(c,200)
        elif ind == "EMA 9":     computed[ind] = ema(c,9)
        elif ind == "EMA 21":    computed[ind] = ema(c,21)
        elif ind == "EMA 50":    computed[ind] = ema(c,50)
        elif ind == "EMA 100":   computed[ind] = ema(c,100)
        elif ind == "EMA 200":   computed[ind] = ema(c,200)
        elif ind == "WMA 20":    computed[ind] = wma(c,20)
        elif ind == "VWAP":      computed[ind] = vwap(h,l,c,vol)
        elif ind == "Bollinger Bands (20,2)":  computed[ind] = bollinger(c,20,2.0)
        elif ind == "Bollinger Bands (20,3)":  computed[ind] = bollinger(c,20,3.0)
        elif ind == "PSAR (0.02)": computed[ind] = parabolic_sar(h,l,0.02)
        elif ind == "PSAR (0.04)": computed[ind] = parabolic_sar(h,l,0.04)
        elif ind == "SuperTrend (10,3)":  computed[ind] = supertrend(h,l,c,10,3.0)
        elif ind == "Ichimoku Cloud": computed[ind] = ichimoku(h,l,c)
        elif ind == "Fibonacci Levels":
            computed[ind] = fibonacci_levels(float(h.max()), float(l.min()))
    except Exception as e:
        pass

panel_data = {}
for ind in panels:
    try:
        if ind == "RSI (8)":          panel_data[ind] = rsi(c,8)
        elif ind == "RSI (14)":       panel_data[ind] = rsi(c,14)
        elif ind == "CCI (14)":       panel_data[ind] = cci(h,l,c,14)
        elif ind == "CCI (20)":       panel_data[ind] = cci(h,l,c,20)
        elif ind == "CCI (50)":       panel_data[ind] = cci(h,l,c,50)
        elif ind == "Williams %R":    panel_data[ind] = williams_r(h,l,c,14)
        elif ind == "Stochastic (14,3)": panel_data[ind] = stochastic(h,l,c,14,3)
        elif ind == "MACD (12,26,9)": panel_data[ind] = macd(c,12,26,9)
        elif ind == "Volume":         panel_data[ind] = vol
        elif ind == "ATR (14)":       panel_data[ind] = atr(h,l,c,14)
        elif ind == "MFI (14)":       panel_data[ind] = mfi(h,l,c,vol,14)
        elif ind == "ADX (14)":       panel_data[ind] = adx(h,l,c,14)
    except Exception:
        pass

# ── compute signals ────────────────────────────────────────────
sig_mark1  = compute_mark1_signal(df)  if show_mark1  else pd.Series(0,index=df.index)
sig_ichcci = compute_ich_cci_signal(df) if show_ichcci else pd.Series(0,index=df.index)
sig_engulf = pattern_engulfing(o,c)    if show_engulf else pd.Series(0,index=df.index)
sig_sniper = pattern_3candle_sniper(o,c,h,l) if show_sniper else pd.Series(0,index=df.index)

composite  = sig_mark1 + sig_ichcci*2 + sig_engulf + sig_sniper
longs  = composite[composite >=  2]
shorts = composite[composite <= -2]

atr14 = atr(h,l,c,14)
last_close = float(c.iloc[-1])
last_atr   = float(atr14.dropna().iloc[-1]) if len(atr14.dropna()) > 0 else last_close*0.001

# last signal for TP/SL
last_sig = int(np.sign(float(composite.iloc[-1])))
tp_price = last_close + last_atr*sl_atr*rrr*last_sig if last_sig != 0 else None
sl_price = last_close - last_atr*sl_atr*last_sig     if last_sig != 0 else None

# ══════════════════════════════════════════════════════════════
# HEADER METRICS
# ══════════════════════════════════════════════════════════════
sym = ticker.replace("=X","").replace("^","")
direction_txt = "▲ LONG" if last_sig==1 else ("▼ SHORT" if last_sig==-1 else "— NEUTRAL")
dcol = G if last_sig==1 else (R if last_sig==-1 else GOLD)

st.markdown(f"# 📡 {sym}  <span style='font-size:1.2rem;'>{tf.upper()}</span>", unsafe_allow_html=True)

row = st.columns(9)
row[0].markdown(f"<div style='font-family:Share Tech Mono;font-size:1.5rem;color:{dcol};'>{direction_txt}</div>",unsafe_allow_html=True)
row[1].metric("Price",  f"{last_close:.5g}")
row[2].metric("ATR",    f"{last_atr:.5g}")

rsi_now = float(panel_data.get("RSI (14)", panel_data.get("RSI (8)", pd.Series([50]))).dropna().iloc[-1]) if "RSI (14)" in panel_data or "RSI (8)" in panel_data else None
if rsi_now: row[3].metric("RSI", f"{rsi_now:.1f}")

cci_now = float(panel_data.get("CCI (14)", pd.Series([0])).dropna().iloc[-1]) if "CCI (14)" in panel_data else None
if cci_now is not None: row[4].metric("CCI", f"{cci_now:.0f}")

wpr_now = float(panel_data.get("Williams %R", pd.Series([-50])).dropna().iloc[-1]) if "Williams %R" in panel_data else None
if wpr_now is not None: row[5].metric("WPR", f"{wpr_now:.0f}")

if tp_price: row[6].metric("TP", f"{tp_price:.5g}")
if sl_price: row[7].metric("SL", f"{sl_price:.5g}")
row[8].metric("Signals", f"L:{len(longs)}  S:{len(shorts)}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# BUILD CHART
# ══════════════════════════════════════════════════════════════
n_panels = len(panel_data)
row_heights = [0.60] + [0.40/max(n_panels,1)]*n_panels if n_panels > 0 else [1.0]
row_heights = [r/sum(row_heights) for r in row_heights]

fig = make_subplots(
    rows       = 1 + n_panels,
    cols       = 1,
    row_heights= row_heights,
    shared_xaxes = True,
    vertical_spacing = 0.02,
    subplot_titles = (
        [f"{sym} {tf}"] +
        [ind for ind in panel_data]
    ),
)

# ── CANDLESTICK ────────────────────────────────────────────────
fig.add_trace(go.Candlestick(
    x=df.index, open=o, high=h, low=l, close=c,
    increasing_line_color=G, decreasing_line_color=R,
    increasing_fillcolor=G, decreasing_fillcolor=R,
    line_width=1, name="Price",
), row=1, col=1)

# ── OVERLAY INDICATORS ────────────────────────────────────────
COLORS = [CY, GOLD, OR, PU, "#FF8C00","#88FFAA","#FF88AA","#AACCFF"]
ci = 0
for ind, val in computed.items():
    col = COLORS[ci % len(COLORS)]; ci += 1
    try:
        if "SMA" in ind or "EMA" in ind or "WMA" in ind or "VWAP" in ind:
            fig.add_trace(go.Scatter(x=val.index, y=val, mode="lines", name=ind,
                line=dict(color=col, width=1.5)), row=1, col=1)

        elif "Bollinger" in ind:
            upper, mid, lower = val
            fig.add_trace(go.Scatter(x=upper.index, y=upper, mode="lines", name=f"BB Upper",
                line=dict(color=col, width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=lower.index, y=lower, mode="lines", name="BB Lower",
                line=dict(color=col, width=1, dash="dot"), fill="tonexty",
                fillcolor=f"rgba(0,204,255,0.04)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=mid.index, y=mid, mode="lines", name="BB Mid",
                line=dict(color=col, width=1)), row=1, col=1)

        elif "PSAR" in ind:
            psar_, ptrd = val
            psar_colors = [G if t==1 else R for t in ptrd]
            fig.add_trace(go.Scatter(x=psar_.index, y=psar_, mode="markers",
                marker=dict(color=psar_colors, size=3, symbol="circle"),
                name=ind), row=1, col=1)

        elif "SuperTrend" in ind:
            st_, stdir = val
            st_colors  = [G if d==1 else R for d in stdir]
            fig.add_trace(go.Scatter(x=st_.index, y=st_, mode="lines",
                line=dict(color=G, width=1.5), name="SuperTrend"), row=1, col=1)

        elif "Ichimoku" in ind:
            tenkan, kijun, span_a, span_b, chikou = val
            # cloud fill
            fig.add_trace(go.Scatter(x=span_a.index, y=span_a, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(x=span_b.index, y=span_b, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(0,255,136,0.06)", showlegend=False, hoverinfo="skip"), row=1, col=1)
            # lines
            fig.add_trace(go.Scatter(x=tenkan.index, y=tenkan, mode="lines", name="Tenkan",
                line=dict(color=CY, width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=kijun.index, y=kijun, mode="lines", name="Kijun",
                line=dict(color=GOLD, width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=chikou.index, y=chikou, mode="lines", name="Chikou",
                line=dict(color=PU, width=1.0, dash="dot")), row=1, col=1)

        elif "Fibonacci" in ind:
            fibs = val
            fib_colors = [GOLD, "#FFCC44","#FFAA33","#FF8822","#FF6611","#FF4400","#FF2200","#CC0000","#990000"]
            for (level, price), fc in zip(fibs.items(), fib_colors):
                fig.add_hline(y=price, line=dict(color=fc, dash="dot", width=1),
                              annotation_text=f"Fib {level}", annotation_font_color=fc,
                              row=1, col=1)
    except Exception:
        pass

# ── SIGNAL MARKERS ─────────────────────────────────────────────
if len(longs) > 0:
    entry_lows = l.reindex(longs.index, method="nearest")
    fig.add_trace(go.Scatter(
        x=longs.index, y=entry_lows*0.9995,
        mode="markers+text",
        marker=dict(color=G, size=14, symbol="triangle-up"),
        text=["▲"]*len(longs), textposition="bottom center",
        textfont=dict(color=G, size=8),
        name="BUY Signal",
    ), row=1, col=1)

if len(shorts) > 0:
    entry_highs = h.reindex(shorts.index, method="nearest")
    fig.add_trace(go.Scatter(
        x=shorts.index, y=entry_highs*1.0005,
        mode="markers+text",
        marker=dict(color=R, size=14, symbol="triangle-down"),
        text=["▼"]*len(shorts), textposition="top center",
        textfont=dict(color=R, size=8),
        name="SELL Signal",
    ), row=1, col=1)

# ── TP/SL LINES ────────────────────────────────────────────────
if show_tpsl and last_sig != 0 and tp_price and sl_price:
    fig.add_hline(y=tp_price, line=dict(color=G, dash="dash", width=1.5),
                  annotation_text=f"TP  {tp_price:.5g}", annotation_font_color=G, row=1, col=1)
    fig.add_hline(y=sl_price, line=dict(color=R, dash="dash", width=1.5),
                  annotation_text=f"SL  {sl_price:.5g}", annotation_font_color=R, row=1, col=1)
    fig.add_hline(y=last_close, line=dict(color=GOLD, dash="dot", width=1),
                  annotation_text=f"Entry  {last_close:.5g}", annotation_font_color=GOLD, row=1, col=1)

# ── PANELS ────────────────────────────────────────────────────
for row_i, (ind, val) in enumerate(panel_data.items(), start=2):
    try:
        if "RSI" in ind:
            fig.add_trace(go.Scatter(x=val.index, y=val, mode="lines",
                line=dict(color=G, width=1.5), name=ind, showlegend=False), row=row_i, col=1)
            fig.add_hline(y=70, line=dict(color=R, dash="dot", width=1), row=row_i, col=1)
            fig.add_hline(y=30, line=dict(color=G, dash="dot", width=1), row=row_i, col=1)
            fig.add_hline(y=50, line=dict(color=MU, width=1), row=row_i, col=1)

        elif "CCI" in ind:
            bar_colors = [G if v>0 else R for v in val.fillna(0)]
            fig.add_trace(go.Bar(x=val.index, y=val, marker_color=bar_colors,
                name=ind, showlegend=False), row=row_i, col=1)
            fig.add_hline(y= 100, line=dict(color=R, dash="dot", width=1), row=row_i, col=1)
            fig.add_hline(y=-100, line=dict(color=G, dash="dot", width=1), row=row_i, col=1)
            fig.add_hline(y=0, line=dict(color=MU, width=1), row=row_i, col=1)

        elif "Williams" in ind:
            fig.add_trace(go.Scatter(x=val.index, y=val, mode="lines",
                line=dict(color=OR, width=1.5), showlegend=False), row=row_i, col=1)
            fig.add_hline(y=-20, line=dict(color=R, dash="dot", width=1), row=row_i, col=1)
            fig.add_hline(y=-80, line=dict(color=G, dash="dot", width=1), row=row_i, col=1)

        elif "Stochastic" in ind:
            k_, d_ = val
            fig.add_trace(go.Scatter(x=k_.index, y=k_, mode="lines", name="K",
                line=dict(color=G, width=1.5), showlegend=False), row=row_i, col=1)
            fig.add_trace(go.Scatter(x=d_.index, y=d_, mode="lines", name="D",
                line=dict(color=GOLD, width=1.2, dash="dot"), showlegend=False), row=row_i, col=1)
            fig.add_hline(y=80, line=dict(color=R, dash="dot", width=1), row=row_i, col=1)
            fig.add_hline(y=20, line=dict(color=G, dash="dot", width=1), row=row_i, col=1)

        elif "MACD" in ind:
            ml, ms, mh = val
            hist_colors = [G if v>=0 else R for v in mh.fillna(0)]
            fig.add_trace(go.Bar(x=mh.index, y=mh, marker_color=hist_colors,
                name="Hist", showlegend=False), row=row_i, col=1)
            fig.add_trace(go.Scatter(x=ml.index, y=ml, mode="lines", name="MACD",
                line=dict(color=CY, width=1.5), showlegend=False), row=row_i, col=1)
            fig.add_trace(go.Scatter(x=ms.index, y=ms, mode="lines", name="Signal",
                line=dict(color=OR, width=1.2), showlegend=False), row=row_i, col=1)
            fig.add_hline(y=0, line=dict(color=MU, width=1), row=row_i, col=1)

        elif "Volume" in ind:
            vol_colors = [G if float(c.iloc[i]) >= float(o.iloc[i]) else R
                           for i in range(len(df))]
            fig.add_trace(go.Bar(x=val.index, y=val, marker_color=vol_colors,
                name="Volume", showlegend=False), row=row_i, col=1)

        elif "ATR" in ind:
            fig.add_trace(go.Scatter(x=val.index, y=val, mode="lines",
                line=dict(color=PU, width=1.5), showlegend=False), row=row_i, col=1)

        elif "MFI" in ind:
            fig.add_trace(go.Scatter(x=val.index, y=val, mode="lines",
                line=dict(color=CY, width=1.5), showlegend=False), row=row_i, col=1)
            fig.add_hline(y=80, line=dict(color=R, dash="dot", width=1), row=row_i, col=1)
            fig.add_hline(y=20, line=dict(color=G, dash="dot", width=1), row=row_i, col=1)

        elif "ADX" in ind:
            adx_val, pdi, ndi = val
            fig.add_trace(go.Scatter(x=adx_val.index, y=adx_val, mode="lines", name="ADX",
                line=dict(color=GOLD, width=1.5), showlegend=False), row=row_i, col=1)
            fig.add_trace(go.Scatter(x=pdi.index, y=pdi, mode="lines", name="+DI",
                line=dict(color=G, width=1.2), showlegend=False), row=row_i, col=1)
            fig.add_trace(go.Scatter(x=ndi.index, y=ndi, mode="lines", name="-DI",
                line=dict(color=R, width=1.2), showlegend=False), row=row_i, col=1)
            fig.add_hline(y=25, line=dict(color=MU, dash="dot", width=1), row=row_i, col=1)

        fig.update_xaxes(gridcolor=GR, zerolinecolor=GR, rangeslider=dict(visible=False),
                         row=row_i, col=1)
        fig.update_yaxes(gridcolor=GR, zerolinecolor=GR, row=row_i, col=1)
    except Exception:
        pass

# ── FINAL LAYOUT ────────────────────────────────────────────────
chart_height = 480 + n_panels * 140

# detect if this is a daily/weekly chart (stocks have weekend gaps)
is_daily_or_weekly = tf in ["1d", "1wk"]

fig.update_layout(
    **_base(chart_height),
    xaxis_rangeslider_visible=False,
)

# remove weekend/holiday gaps on daily+ charts
if is_daily_or_weekly:
    fig.update_xaxes(
        type="category",  # treat x as categories = no gaps
        tickangle=-45,
        tickfont=dict(size=9),
        gridcolor=GR,
    )
else:
    fig.update_xaxes(
        gridcolor=GR,
        zerolinecolor=GR,
        rangeslider_visible=False,
    )

fig.update_yaxes(gridcolor=GR, zerolinecolor=GR)

st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# BOTTOM TABS
# ══════════════════════════════════════════════════════════════
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🎯  TP/SL & Position Size","📋  Active Signals","🔬  Backtest"])

with tab1:
    c1,c2,c3 = st.columns(3)
    entry_  = c1.number_input("Entry", value=last_close, format="%.5f")
    dir_    = c1.radio("Direction", ["LONG","SHORT"], horizontal=True)
    atr_inp = c2.number_input("ATR", value=last_atr, format="%.5f")
    rrr_inp = c2.select_slider("RRR", [1.0,1.5,2.0,2.5,3.0,4.0], value=rrr)
    sl_m    = c3.slider("SL Multiplier (× ATR)", 0.5, 4.0, sl_atr, step=0.25)

    d  = 1 if dir_=="LONG" else -1
    sl = entry_ - atr_inp*sl_m*d
    tp = entry_ + atr_inp*sl_m*rrr_inp*d

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Entry",       f"{entry_:.5g}")
    m2.metric("Stop Loss",   f"{sl:.5g}",   delta=f"-{abs(entry_-sl):.5g}")
    m3.metric("Take Profit", f"{tp:.5g}",   delta=f"+{abs(tp-entry_):.5g}")
    m4.metric("Breakeven WR",f"{1/(1+rrr_inp)*100:.1f}%")

    st.markdown("#### Position Sizing")
    pa,pb,pc = st.columns(3)
    acc  = pa.number_input("Account ($)", value=10_000.0)
    risk = pb.slider("Risk per trade (%)", 0.25, 5.0, 1.0, step=0.25)
    pip  = pc.number_input("Pip value ($ / std lot)", value=10.0)

    risk_amt = acc*risk/100
    sl_dist = abs(entry_-sl)
    st.markdown(f"**Risk Amount:** ${risk_amt:,.2f} &nbsp;|&nbsp; **SL Distance:** {sl_dist:.5g} &nbsp;|&nbsp; **Kelly (55% WR):** {max(0, 0.55-0.45/rrr_inp)*100:.1f}% of capital")

with tab2:
    sig_rows = []
    sig_map = {
        "Mark I (PSAR+RSI)": sig_mark1,
        "ICH+CCI": sig_ichcci,
        "Engulfing": sig_engulf,
        "3-Candle Sniper": sig_sniper,
    }
    for name, s in sig_map.items():
        last_s = int(s.dropna().iloc[-1]) if len(s.dropna()) > 0 else 0
        n_l    = int((s == 1).sum())
        n_s    = int((s == -1).sum())
        recent = s[s != 0]
        last_date = str(recent.index[-1].date()) if len(recent) > 0 else "none"
        sig_rows.append({
            "Strategy": name,
            "Current": "▲ LONG" if last_s==1 else ("▼ SHORT" if last_s==-1 else "—"),
            "Longs (total)": n_l,
            "Shorts (total)": n_s,
            "Last signal": last_date,
        })
    st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)

with tab3:
    st.caption("Walks through every signal on historical data. TP/SL set by ATR.")
    col_a, col_b = st.columns(2)
    bt_strat = col_a.selectbox("Strategy to backtest", list(sig_map.keys()))
    bt_rrr   = col_b.select_slider("RRR", [1.0,1.5,2.0,2.5,3.0], value=2.0, key="bt_rrr")

    if st.button("▶  Run Backtest"):
        sig_to_bt = sig_map[bt_strat]
        bt_df = backtest(df, sig_to_bt, rrr=bt_rrr, atr_mult=sl_atr)

        if bt_df is not None and len(bt_df) > 0:
            closed = bt_df[bt_df["result"].isin(["TP","SL"])]
            wr = (closed["R"] > 0).mean() if len(closed) > 0 else 0
            total_r = closed["R"].sum() if len(closed) > 0 else 0

            ca,cb,cc,cd = st.columns(4)
            ca.metric("Trades", str(len(closed)))
            cb.metric("Win Rate", f"{wr*100:.1f}%")
            cc.metric("Total R", f"{total_r:+.1f}R")
            cd.metric("Sharpe (approx)", f"{(closed['R'].mean()/closed['R'].std()*16):.2f}" if closed['R'].std()>0 else "0")

            st.dataframe(bt_df, hide_index=True, use_container_width=True)

            # equity curve
            if len(closed) > 0:
                eq = go.Figure(go.Scatter(
                    x=closed.index, y=closed["R"].cumsum(),
                    mode="lines+markers",
                    line=dict(color=G, width=2),
                    marker=dict(color=[G if r>0 else R for r in closed["R"]], size=7),
                ))
                eq.add_hline(y=0, line=dict(color=MU, width=1))
                eq.update_layout(**_base(250), title=f"{bt_strat} — Equity (R)")
                st.plotly_chart(eq, use_container_width=True)
        else:
            st.info("No signals generated on this dataset. Try a longer timeframe or different start date.")