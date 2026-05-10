#app.py

# Unified Trading Signal Bot — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from signal_bot import run_bot, generate_orders, DEFAULT_WEIGHTS, ALPACA_AVAILABLE

st.set_page_config(page_title="Signal Bot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; background-color: #04080C; color: #A0D4B8; }
h1 { font-family: 'Share Tech Mono', monospace !important; color: #00FF88 !important; text-shadow: 0 0 12px rgba(0,255,136,0.25); }
h2, h3 { color: #1A4A30 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #081410; border: 1px solid #0C2018; border-radius: 3px; padding: 12px 16px; }
[data-testid="metric-container"] label { font-family: 'Share Tech Mono', monospace !important; font-size: 0.58rem !important; letter-spacing: 0.12em; color: #0A2018 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Share Tech Mono', monospace !important; font-size: 1.0rem !important; color: #00FF88 !important; }
[data-testid="stSidebar"] { background: #020608; border-right: 1px solid #0C2018; }
.stTabs [data-baseweb="tab"] { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; background: #081410; border-radius: 3px; border: 1px solid #0C2018; color: #0A2018; }
.stTabs [aria-selected="true"] { background: #0C2018 !important; border-color: #00FF88 !important; color: #00FF88 !important; }
hr { border-color: #0C2018 !important; }
</style>
""", unsafe_allow_html=True)

GREEN="#00FF88"; RED="#FF4466"; GOLD="#FFD700"; MUTED="#1A4A30"; BG="#04080C"; GRID="#0C2018"; TEXT="#A0D4B8"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Share Tech Mono, monospace", color=TEXT, size=10),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## 🤖 Signal Bot")
    st.markdown("---")

    ticker_input = st.text_area("Asset Universe", height=180,
        value="SPY\nQQQ\nTLT\nGLD\nEEM\nVNQ\nDBC\nHYG")
    tickers      = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split("\n") if t.strip()]))

    start_year   = st.selectbox("Backtest start", [2010, 2012, 2015, 2018], index=2)
    target_vol   = st.slider("Target Vol (%)",   5, 25, 15) / 100
    max_leverage = st.slider("Max Leverage",     0.5, 3.0, 2.0, step=0.5)
    dd_stop      = st.slider("DD Stop (%)",      5, 50, 20) / 100

    st.markdown("---")
    st.markdown("#### Signal Weights")
    w_tsmom = st.slider("TSMOM",       0.0, 1.0, 0.30, step=0.05)
    w_xsmom = st.slider("Cross-sect",  0.0, 1.0, 0.25, step=0.05)
    w_vol   = st.slider("Vol Regime",  0.0, 1.0, 0.20, step=0.05)
    w_trend = st.slider("SMA Trend",   0.0, 1.0, 0.15, step=0.05)
    w_sent  = st.slider("Sentiment",   0.0, 0.3, 0.10, step=0.05)
    incl_sent = st.checkbox("Fetch live sentiment (RSS)", value=False)

    st.markdown("---")
    st.markdown("#### Alpaca Paper Trading")
    alpaca_key    = st.text_input("API Key",    type="password")
    alpaca_secret = st.text_input("Secret Key", type="password")
    port_value    = st.number_input("Portfolio ($)", value=100_000, step=10_000)
    dry_run       = st.checkbox("Dry run (no orders submitted)", value=True)
    if not ALPACA_AVAILABLE:
        st.caption("pip install alpaca-py for live orders")

start = f"{start_year}-01-01"

# normalise weights
raw_w = {"tsmom": w_tsmom, "xsmom": w_xsmom, "vol_regime": w_vol,
         "trend": w_trend, "sentiment": w_sent}
total = sum(raw_w.values())
weights = {k: v/total for k, v in raw_w.items()} if total > 0 else DEFAULT_WEIGHTS

@st.cache_data(ttl=300)
def load_bot(tickers_t, start, tv, ml, dds, weights_t, incl_s):
    return run_bot(list(tickers_t), start=start, target_vol=tv,
                   max_leverage=ml, dd_threshold=dds,
                   weights=dict(weights_t), include_sentiment=incl_s)

with st.spinner("Running signal ensemble..."):
    try:
        result = load_bot(tuple(tickers), start, target_vol, max_leverage,
                          dd_stop, tuple(sorted(weights.items())), incl_sent)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 UNIFIED TRADING SIGNAL BOT")
st.markdown(
    f"`{len(result.tickers)} assets` · "
    f"`target vol={target_vol*100:.0f}%` · "
    f"`DD stop={dd_stop*100:.0f}%` · "
    f"`{len(result.weights_used)} signals`"
)
st.markdown("---")

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Ann. Return",   f"{result.ann_return*100:.2f}%")
c2.metric("Ann. Vol",      f"{result.ann_vol*100:.2f}%")
c3.metric("Sharpe",        f"{result.sharpe:.3f}")
c4.metric("Sortino",       f"{result.sortino:.3f}")
c5.metric("Max Drawdown",  f"{result.max_drawdown*100:.2f}%")
c6.metric("Calmar",        f"{result.calmar:.3f}")

st.markdown("---")

# ── live signal table ──────────────────────────────────────────────────────────
st.markdown("### 📡 Current Signal Snapshot")
sig_rows = []
for t, s in result.signals.items():
    direction = ("▲ LONG" if s.final_position > 0.05
                 else ("▼ SHORT" if s.final_position < -0.05 else "— FLAT"))
    if s.dd_stop_active:
        direction = "🛑 STOPPED"
    sig_rows.append({
        "Ticker":    t,
        "Direction": direction,
        "Position":  f"{s.final_position:+.3f}",
        "TSMOM":     f"{s.tsmom:+.2f}",
        "X-Sect":    f"{s.xsmom:+.2f}",
        "Vol Regime":f"{s.vol_regime:+.2f}",
        "Trend":     f"{s.trend:+.2f}",
        "Composite": f"{s.composite:+.3f}",
        "Realised σ":f"{s.realised_vol*100:.1f}%",
        "Agreement": f"{s.signal_agreement*100:.0f}%",
    })
st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Equity Curve",
    "🎛️  Signal History",
    "🗓️  Position Heatmap",
    "📉  Drawdown",
    "⚡  Weight Sensitivity",
    "🚀  Execute Orders",
])

with tab1:
    cum = result.portfolio_cumret
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=(cum-1)*100, mode="lines",
        name="Signal Bot",
        line=dict(color=GREEN, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,255,136,0.05)"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**{**_base, "title": "Signal Bot — Cumulative Return (backtest)"})
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    # rolling 252-day Sharpe
    r   = result.portfolio_returns
    rs  = (r.rolling(252).mean() * 252) / (r.rolling(252).std() * np.sqrt(252))
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines",
        line=dict(color=GOLD, width=2), showlegend=False))
    fig_rs.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig_rs.add_hline(y=1, line=dict(color=GREEN, dash="dot", width=1),
                     annotation_text="Sharpe=1")
    fig_rs.update_layout(**{**_base, "title": "Rolling 252-Day Sharpe", "height": 280})
    st.plotly_chart(fig_rs, use_container_width=True)

with tab2:
    t_sel = st.selectbox("Asset", result.tickers)
    if t_sel in result.composite_history.columns:
        ch = result.composite_history[t_sel].dropna()
        colours = [GREEN if v > 0 else RED for v in ch.values]
        fig2 = go.Figure(go.Bar(x=ch.index, y=ch.values,
            marker_color=colours, showlegend=False))
        fig2.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig2.update_layout(**{**_base, "title": f"{t_sel} — Daily Composite Signal"})
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    wh  = result.weights_history.resample("ME").last().dropna(how="all")
    fig3 = go.Figure(go.Heatmap(
        z=wh.T.values, x=wh.index, y=wh.columns.tolist(),
        colorscale="RdYlGn", zmid=0, zmin=-2, zmax=2,
        colorbar=dict(title=dict(text="Position", font=dict(color=TEXT)),
                      tickfont=dict(color=TEXT), thickness=12),
    ))
    fig3.update_layout(**{**_base,
        "title": "Monthly Position Heatmap  (Green=Long, Red=Short)",
        "height": max(300, len(result.tickers)*45)})
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Clusters of red = strategy correctly went short during drawdown periods.")

with tab4:
    cum   = result.portfolio_cumret
    peak  = cum.cummax()
    dd    = (cum - peak) / peak * 100
    fig4  = go.Figure()
    fig4.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
        line=dict(color=RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,68,102,0.10)"))
    fig4.add_hline(y=-dd_stop*100, line=dict(color=GOLD, dash="dash", width=1.5),
                   annotation_text=f"DD Stop: -{dd_stop*100:.0f}%",
                   annotation_font_color=GOLD)
    fig4.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig4.update_layout(**{**_base, "title": "Portfolio Drawdown + Stop Level"})
    fig4.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    st.caption("Compare how results change when you adjust a single signal weight to 0 vs 0.5.")
    if st.button("▶  Run Weight Sensitivity (5 simulations)"):
        sensitivity_rows = []
        prog = st.progress(0)
        for i, (signal_name, _) in enumerate(weights.items()):
            # run with this signal weight = 0 (turned off) vs default
            w_off = {k: (0.0 if k == signal_name else v) for k, v in weights.items()}
            t_off = sum(w_off.values())
            w_off = {k: v/t_off for k, v in w_off.items()} if t_off > 0 else weights
            try:
                r_off = run_bot(result.tickers, start=start, target_vol=target_vol,
                                max_leverage=max_leverage, dd_threshold=dd_stop,
                                weights=w_off)
                sensitivity_rows.append({
                    "Signal removed":    signal_name,
                    "Sharpe (without)":  round(r_off.sharpe, 3),
                    "Sharpe (baseline)": round(result.sharpe, 3),
                    "Δ Sharpe":          round(r_off.sharpe - result.sharpe, 3),
                    "Ann Return (without)": f"{r_off.ann_return*100:.2f}%",
                })
            except Exception:
                pass
            prog.progress((i+1)/len(weights))

        if sensitivity_rows:
            sens_df = pd.DataFrame(sensitivity_rows)
            st.dataframe(sens_df, hide_index=True, use_container_width=True)
            st.caption("Negative Δ Sharpe = removing that signal hurt performance = it adds value.")

with tab6:
    st.markdown("### 🚀 Order Generation")
    if not ALPACA_AVAILABLE:
        st.warning("Install alpaca-py for live execution:\n```\npip install alpaca-py\n```")

    orders = generate_orders(result.signals, portfolio_value=port_value,
                              api_key=alpaca_key, secret_key=alpaca_secret,
                              dry_run=True)
    orders_df = pd.DataFrame(orders)
    st.dataframe(orders_df, hide_index=True, use_container_width=True)

    if alpaca_key and alpaca_secret and not dry_run:
        if st.button("🚀 SUBMIT TO ALPACA PAPER ACCOUNT", type="primary"):
            live = generate_orders(result.signals, portfolio_value=port_value,
                                   api_key=alpaca_key, secret_key=alpaca_secret,
                                   dry_run=False)
            submitted = sum(1 for o in live if "submitted" in str(o.get("status","")))
            st.success(f"Submitted {submitted} orders to Alpaca paper account")
            st.dataframe(pd.DataFrame(live), hide_index=True)
    elif dry_run:
        st.caption("Dry run mode — uncheck to submit real orders to Alpaca paper account.")