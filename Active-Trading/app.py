#app.py

# Active Trading Dashboard
# Your Mark I-IV rules + ICH+CCI + 3-Candle Sniper + Engulfing
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from signals import (
    fetch_ohlcv, sma, ema, rsi, cci, williams_r,
    parabolic_sar, macd, bollinger_bands, volume_ma, atr, ichimoku,
    compute_all_signals, calculate_tp_sl, backtest_signals,
    TIMEFRAME_MAP,
)

st.set_page_config(page_title="Active Trading", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; background-color: #030608; color: #A8D8C0; }
h1 { font-family: 'Share Tech Mono', monospace !important; color: #00FF88 !important; font-size: 1.6rem !important; }
h2, h3 { color: #1A4A30 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #081410; border: 1px solid #0C2018; border-radius: 3px; padding: 10px 14px; }
[data-testid="metric-container"] label { font-family: 'Share Tech Mono', monospace !important; font-size: 0.58rem !important; letter-spacing: 0.1em; color: #0A2018 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Share Tech Mono', monospace !important; font-size: 1.0rem !important; color: #00FF88 !important; }
[data-testid="stSidebar"] { background: #020608; border-right: 1px solid #0C2018; }
.stTabs [data-baseweb="tab"] { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; background: #081410; border-radius: 3px; border: 1px solid #0C2018; color: #0A2018; }
.stTabs [aria-selected="true"] { background: #0C2018 !important; border-color: #00FF88 !important; color: #00FF88 !important; }
hr { border-color: #0C2018 !important; }
</style>
""", unsafe_allow_html=True)

GREEN="#00FF88"; RED="#FF4466"; GOLD="#FFD700"; CYAN="#00CCFF"
ORANGE="#FF8C00"; PURPLE="#CC88FF"; MUTED="#1A4A30"
BG="#030608"; GRID="#0C2018"; TEXT="#A8D8C0"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Share Tech Mono, monospace", color=TEXT, size=10),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, rangeslider=dict(visible=False)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, font=dict(size=9)),
)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Active Trading")
    st.markdown("---")

    ticker    = st.text_input("Ticker / Symbol", value="EURUSD=X",
                              help="FX: EURUSD=X  Gold: GC=F  Crypto: BTC-USD  Index: ^GSPC")
    timeframe = st.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=2,
                             help="15m default — matches your IG chart")

    st.markdown("#### Active Strategies")
    use_mark1   = st.checkbox("Mark I  (PSAR + SMA50 + RSI8)", value=True)
    use_ichcci  = st.checkbox("ICH+CCI  (your notebook)", value=True)
    use_mark4   = st.checkbox("Mark IV  (EMA ribbon + RSI div)", value=True)
    use_sniper  = st.checkbox("3-Candle Sniper", value=True)
    use_engulf  = st.checkbox("Engulfing Pattern", value=True)

    st.markdown("#### TP/SL Settings")
    rrr      = st.slider("RRR (Risk:Reward)", 1.0, 4.0, 2.0, step=0.5)
    sl_mult  = st.slider("SL (× ATR)", 0.5, 3.0, 1.5, step=0.25)
    af_step  = st.select_slider("PSAR AF Step", [0.02, 0.04, 0.06, 0.08, 0.1], value=0.04)

    st.markdown("---")
    st.caption("Data: Yahoo Finance\nAuto-refreshes on symbol change\nAll strategies run in parallel")

# ── data ────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)   # refresh every 60s
def load_data(ticker, timeframe):
    return fetch_ohlcv(ticker, timeframe)

with st.spinner(f"Loading {ticker} {timeframe}..."):
    df = load_data(ticker, timeframe)

if df is None or len(df) < 30:
    st.error(f"No data for {ticker}. Try: EURUSD=X, GC=F, BTC-USD, AAPL, ^GSPC")
    st.stop()

# ── compute indicators ─────────────────────────────────────────────────────────
close = df["Close"]; high = df["High"]; low = df["Low"]; open_ = df["Open"]
vol   = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)

sma50   = sma(close, 50)
ema21   = ema(close, 21)
ema100  = ema(close, 100)
rsi8    = rsi(close, 8)
rsi14   = rsi(close, 14)
cci14   = cci(high, low, close, 14)
wpr     = williams_r(high, low, close, 14)
psar, psar_trend = parabolic_sar(high, low, af_step=af_step)
macd_l, macd_s, macd_h = macd(close)
bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2)
vol_ma_ = volume_ma(vol, 20)
atr14   = atr(high, low, close, 14)
tenkan, kijun, span_a, span_b, chikou, cloud_thick = ichimoku(high, low, close)

# compute all signals
sig_df  = compute_all_signals(df)
latest  = sig_df.iloc[-1] if len(sig_df) > 0 else None

# current signal summary
composite_now = int(np.sign(float(latest["composite"]))) if latest is not None else 0
atr_now       = float(atr14.dropna().iloc[-1])
entry_price   = float(close.iloc[-1])
tpsl          = calculate_tp_sl(entry_price, composite_now if composite_now != 0 else 1, atr_now, rrr, sl_mult)

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown(f"# 📡 {ticker.replace('=X','').replace('^','')}  {timeframe.upper()}")

# signal boxes
direction_col  = GREEN if composite_now == 1 else (RED if composite_now == -1 else GOLD)
direction_text = "▲ LONG" if composite_now == 1 else ("▼ SHORT" if composite_now == -1 else "— NEUTRAL")

c0,c1,c2,c3,c4,c5,c6,c7 = st.columns(8)
c0.markdown(f"<div style='font-family:Share Tech Mono;font-size:1.6rem;color:{direction_col};padding:8px 0;'>{direction_text}</div>", unsafe_allow_html=True)
c1.metric("Price",    f"{entry_price:.5g}")
c2.metric("RSI(8)",   f"{float(rsi8.dropna().iloc[-1]):.1f}")
c3.metric("CCI(14)",  f"{float(cci14.dropna().iloc[-1]):.1f}")
c4.metric("WPR",      f"{float(wpr.dropna().iloc[-1]):.1f}")
c5.metric("ATR(14)",  f"{atr_now:.5g}")
c6.metric("TP",       f"{tpsl['tp']:.5g}")
c7.metric("SL",       f"{tpsl['sl']:.5g}")

st.markdown("---")

# signal agreement table
if latest is not None:
    sig_names  = ["mark1","ich_cci","mark4","sniper","engulf"]
    sig_labels = ["Mark I\nPSAR+RSI", "ICH+CCI\nYour Note", "Mark IV\nEMA Div", "3-Candle\nSniper", "Engulfing"]
    cols = st.columns(5)
    for col, name, label in zip(cols, sig_names, sig_labels):
        v = int(latest[name])
        icon  = "▲" if v == 1 else ("▼" if v == -1 else "—")
        color = GREEN if v == 1 else (RED if v == -1 else GOLD)
        col.markdown(
            f"<div style='text-align:center;background:#081410;border:1px solid #0C2018;border-radius:4px;padding:10px;'>"
            f"<div style='font-family:Share Tech Mono;font-size:1.4rem;color:{color};'>{icon}</div>"
            f"<div style='font-size:0.65rem;color:#336655;white-space:pre;'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # show active reasons
    active_reasons = [str(latest[f"reason_{n}"]) for n in sig_names if latest[f"reason_{n}"] != ""]
    if active_reasons:
        for r in active_reasons:
            st.markdown(f"<span style='font-family:Share Tech Mono;font-size:0.7rem;color:#00FF88;'>● {r}</span>", unsafe_allow_html=True)

st.markdown("---")

# ── main chart tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🕯️  Price + Signals",
    "⚡  Ichimoku + CCI",
    "📊  Oscillators",
    "🎯  TP/SL Calculator",
    "🔬  Backtest",
])

with tab1:
    n_candles = st.slider("Candles to show", 50, min(500, len(df)), 120, key="n1")
    df_plot   = df.tail(n_candles)
    sp_a      = span_a.tail(n_candles)
    sp_b      = span_b.tail(n_candles)
    psar_plot = psar.tail(n_candles)
    sma_plot  = sma50.tail(n_candles)
    ema21_plot= ema21.tail(n_candles)

    fig = go.Figure()

    # candlesticks
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="Price",
    ))

    # Ichimoku cloud
    fig.add_trace(go.Scatter(x=sp_a.index, y=sp_a, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sp_b.index, y=sp_b, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,255,136,0.06)", showlegend=False, hoverinfo="skip"))

    # SMA50 and EMA21
    fig.add_trace(go.Scatter(x=sma_plot.index, y=sma_plot, mode="lines",
        line=dict(color=CYAN, width=1.5, dash="dot"), name="SMA50"))
    fig.add_trace(go.Scatter(x=ema21_plot.index, y=ema21_plot, mode="lines",
        line=dict(color=ORANGE, width=1.5), name="EMA21"))

    # PSAR dots
    fig.add_trace(go.Scatter(x=psar_plot.index, y=psar_plot, mode="markers",
        marker=dict(color=[GREEN if t==1 else RED for t in psar_trend.tail(n_candles)],
                    size=4, symbol="circle"), name="PSAR"))

    # signal markers
    if len(sig_df) > 0:
        sig_plot = sig_df.tail(n_candles)
        longs  = sig_plot[sig_plot["composite"] >  2]
        shorts = sig_plot[sig_plot["composite"] < -2]
        if len(longs) > 0:
            fig.add_trace(go.Scatter(
                x=longs.index,
                y=df_plot["Low"].reindex(longs.index) * 0.9995,
                mode="markers", marker=dict(color=GREEN, size=14, symbol="triangle-up"),
                name="BUY Signal",
            ))
        if len(shorts) > 0:
            fig.add_trace(go.Scatter(
                x=shorts.index,
                y=df_plot["High"].reindex(shorts.index) * 1.0005,
                mode="markers", marker=dict(color=RED, size=14, symbol="triangle-down"),
                name="SELL Signal",
            ))

    # TP/SL lines if signal exists
    if composite_now != 0:
        fig.add_hline(y=tpsl["tp"], line=dict(color=GREEN, dash="dash", width=1),
                      annotation_text=f"TP {tpsl['tp']:.5g}", annotation_font_color=GREEN)
        fig.add_hline(y=tpsl["sl"], line=dict(color=RED, dash="dash", width=1),
                      annotation_text=f"SL {tpsl['sl']:.5g}", annotation_font_color=RED)
        fig.add_hline(y=entry_price, line=dict(color=GOLD, dash="dot", width=1),
                      annotation_text=f"Entry {entry_price:.5g}", annotation_font_color=GOLD)

    fig.update_layout(**{**_base, "height": 550, "title": f"{ticker} {timeframe} — Price + Signals"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    n2 = st.slider("Candles", 50, min(500, len(df)), 120, key="n2")
    df2 = df.tail(n2)

    fig2 = sp.make_subplots(rows=3, cols=1,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=["Ichimoku + Price", "CCI (14) — ±100 extreme zones", "Williams %R — ±20/±80 zones"],
        vertical_spacing=0.05, shared_xaxes=True)

    # price + ichimoku
    fig2.add_trace(go.Candlestick(
        x=df2.index, open=df2["Open"], high=df2["High"],
        low=df2["Low"], close=df2["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED, showlegend=False), row=1, col=1)

    sa2 = span_a.tail(n2); sb2 = span_b.tail(n2)
    tk2 = tenkan.tail(n2); kj2 = kijun.tail(n2); ch2 = chikou.tail(n2)

    fig2.add_trace(go.Scatter(x=sa2.index, y=sa2, line=dict(width=0), showlegend=False), row=1, col=1)
    fig2.add_trace(go.Scatter(x=sb2.index, y=sb2, line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,255,136,0.06)", showlegend=False), row=1, col=1)
    fig2.add_trace(go.Scatter(x=tk2.index, y=tk2, mode="lines",
        line=dict(color=CYAN, width=1.2), name="Tenkan"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=kj2.index, y=kj2, mode="lines",
        line=dict(color=GOLD, width=1.2), name="Kijun"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=ch2.index, y=ch2, mode="lines",
        line=dict(color=PURPLE, width=1.0, dash="dot"), name="Chikou"), row=1, col=1)

    # CCI
    cci2 = cci14.tail(n2)
    cci_colours = [GREEN if v > 0 else RED for v in cci2]
    fig2.add_trace(go.Bar(x=cci2.index, y=cci2, marker_color=cci_colours,
        name="CCI", showlegend=False), row=2, col=1)
    for level, col in [(100, RED), (-100, GREEN)]:
        fig2.add_hline(y=level, line=dict(color=col, dash="dot", width=1), row=2, col=1)
    fig2.add_hline(y=0, line=dict(color=MUTED, width=1), row=2, col=1)

    # WPR
    wpr2 = wpr.tail(n2)
    fig2.add_trace(go.Scatter(x=wpr2.index, y=wpr2, mode="lines",
        line=dict(color=ORANGE, width=1.5), showlegend=False), row=3, col=1)
    for level, col in [(-20, RED), (-80, GREEN)]:
        fig2.add_hline(y=level, line=dict(color=col, dash="dot", width=1), row=3, col=1)

    for r in range(1, 4):
        fig2.update_xaxes(gridcolor=GRID, row=r, col=1)
        fig2.update_yaxes(gridcolor=GRID, row=r, col=1)

    fig2.update_layout(**{**_base, "height": 660, "title": "Ichimoku + CCI + WPR"})
    st.plotly_chart(fig2, use_container_width=True)

    # cloud thickness note
    thick = float(cloud_thick.dropna().iloc[-1]) / float(close.iloc[-1]) * 100
    if thick < 0.5:
        st.warning(f"⚠️ Cloud thickness = {thick:.2f}% — THIN CLOUD — high uncertainty, avoid trading")
    else:
        st.success(f"✅ Cloud thickness = {thick:.2f}% — cloud width OK")

with tab3:
    n3 = min(200, len(df))
    fig3 = sp.make_subplots(rows=3, cols=1,
        subplot_titles=["MACD (12,26,9)", "RSI (8 + 14)", "Bollinger Bands"],
        vertical_spacing=0.08, shared_xaxes=True)

    # MACD
    ml3 = macd_l.tail(n3); ms3 = macd_s.tail(n3); mh3 = macd_h.tail(n3)
    fig3.add_trace(go.Bar(x=mh3.index, y=mh3, marker_color=[GREEN if v>0 else RED for v in mh3],
        showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=ml3.index, y=ml3, mode="lines",
        line=dict(color=CYAN, width=1.5), name="MACD"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=ms3.index, y=ms3, mode="lines",
        line=dict(color=ORANGE, width=1.5), name="Signal"), row=1, col=1)

    # RSI
    r8 = rsi8.tail(n3); r14 = rsi14.tail(n3)
    fig3.add_trace(go.Scatter(x=r8.index, y=r8, mode="lines",
        line=dict(color=GREEN, width=1.8), name="RSI8"), row=2, col=1)
    fig3.add_trace(go.Scatter(x=r14.index, y=r14, mode="lines",
        line=dict(color=PURPLE, width=1.2, dash="dot"), name="RSI14"), row=2, col=1)
    for level, col in [(70, RED), (30, GREEN), (50, MUTED)]:
        fig3.add_hline(y=level, line=dict(color=col, dash="dot", width=1), row=2, col=1)

    # BB
    cl3 = close.tail(n3); bu3 = bb_u.tail(n3); bm3 = bb_m.tail(n3); bl3 = bb_l.tail(n3)
    fig3.add_trace(go.Scatter(x=cl3.index, y=cl3, mode="lines",
        line=dict(color=CYAN, width=1.5), showlegend=False), row=3, col=1)
    fig3.add_trace(go.Scatter(x=bu3.index, y=bu3, line=dict(width=0), showlegend=False), row=3, col=1)
    fig3.add_trace(go.Scatter(x=bl3.index, y=bl3, line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,255,136,0.05)", showlegend=False), row=3, col=1)

    for r in range(1, 4):
        fig3.update_xaxes(gridcolor=GRID, row=r, col=1)
        fig3.update_yaxes(gridcolor=GRID, row=r, col=1)

    fig3.update_layout(**{**_base, "height": 620, "title": "Oscillator Dashboard"})
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.markdown("### 🎯 TP / SL Calculator")
    c1, c2, c3 = st.columns(3)
    manual_entry = c1.number_input("Entry Price", value=entry_price, format="%.5f")
    manual_dir   = c2.radio("Direction", ["LONG","SHORT"], horizontal=True)
    manual_atr   = c3.number_input("ATR", value=atr_now, format="%.5f")

    manual_rrr  = st.slider("RRR", 1.0, 5.0, rrr, step=0.5, key="calc_rrr")
    manual_sl   = st.slider("SL mult (× ATR)", 0.5, 4.0, sl_mult, step=0.25, key="calc_sl")

    d = 1 if manual_dir == "LONG" else -1
    result = calculate_tp_sl(manual_entry, d, manual_atr, manual_rrr, manual_sl)

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Entry",    f"{result['entry']:.5g}")
    col_b.metric("Stop Loss", f"{result['sl']:.5g}", delta=f"-{result['sl_pips']:.5g}")
    col_c.metric("Take Profit", f"{result['tp']:.5g}", delta=f"+{result['tp_pips']:.5g}")
    col_d.metric("SL Distance", f"{result['sl_pips']:.5g}")
    col_e.metric("RRR", f"1:{manual_rrr}")

    # position size calculator
    st.markdown("#### Position Size")
    acc_size  = st.number_input("Account size ($)", value=10_000)
    risk_pct  = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, step=0.25)
    pip_value = st.number_input("Pip value ($)", value=10.0)

    risk_amount = acc_size * risk_pct / 100
    sl_pips_val = result["sl_pips"]
    lot_size    = risk_amount / (sl_pips_val * pip_value / sl_pips_val) if sl_pips_val > 0 else 0

    st.markdown(f"""
| Parameter | Value |
|---|---|
| Account Size | ${acc_size:,.0f} |
| Risk per Trade | {risk_pct}% = **${risk_amount:,.0f}** |
| SL Distance | {sl_pips_val:.5g} |
| Max Position Value | **${risk_amount / (risk_pct/100):.0f}** |
| Kelly (approx 55% WR, {manual_rrr}:1) | {max(0, 0.55 - 0.45/manual_rrr)*100:.1f}% of capital |
""")

with tab5:
    st.markdown("### 🔬 Strategy Backtest")
    st.caption("Tests each signal on historical data. TP and SL set by ATR × your multipliers.")

    if st.button("▶  Run Backtest"):
        with st.spinner("Backtesting all strategies..."):
            results_table = []
            for signal_name in ["mark1","ich_cci","mark4","sniper","engulf","composite"]:
                try:
                    bt = backtest_signals(df, sig_df, signal_col=signal_name,
                                          atr_mult_sl=sl_mult, rrr=rrr)
                    results_table.append({
                        "Strategy":   signal_name,
                        "Trades":     bt["n_trades"],
                        "Wins":       bt["n_wins"],
                        "Win Rate":   f"{bt['win_rate']*100:.1f}%",
                        "Total R":    f"{bt['total_r']:+.2f}R",
                        "Sharpe":     f"{bt['sharpe']:.3f}",
                    })
                except Exception:
                    pass

        if results_table:
            st.dataframe(pd.DataFrame(results_table), hide_index=True, use_container_width=True)

            # show individual trades for selected strategy
            sel_strat = st.selectbox("View trades for", ["mark1","ich_cci","mark4","sniper","engulf"])
            bt_detail = backtest_signals(df, sig_df, signal_col=sel_strat,
                                          atr_mult_sl=sl_mult, rrr=rrr)
            if bt_detail["trades"]:
                trades_df = pd.DataFrame(bt_detail["trades"])
                st.dataframe(trades_df, hide_index=True, use_container_width=True)

                # equity curve
                trades_df["cumR"] = trades_df["pnl_r"].cumsum()
                fig_eq = go.Figure(go.Scatter(x=trades_df.index, y=trades_df["cumR"],
                    mode="lines+markers",
                    line=dict(color=GREEN, width=2),
                    marker=dict(color=[GREEN if v>0 else RED for v in trades_df["pnl_r"]], size=8)))
                fig_eq.add_hline(y=0, line=dict(color=MUTED, width=1))
                fig_eq.update_layout(**{**_base, "title": f"{sel_strat} — Equity Curve (R multiple)",
                                         "height": 300})
                st.plotly_chart(fig_eq, use_container_width=True)