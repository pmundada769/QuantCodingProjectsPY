#app.py

# Macro Intelligence Dashboard
# FRED + VIX term structure + FedWatch + DXY + credit spreads + real yields
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
import requests
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Macro Dashboard", page_icon="🌍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #060A0E; color: #C0D0E0; }
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #00CCFF !important; }
h2, h3 { color: #1A3050 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #0A1018; border: 1px solid #0E2030; border-radius: 3px; padding: 10px 14px; }
[data-testid="metric-container"] label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.58rem !important; color: #0E2030 !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.0rem !important; color: #00CCFF !important; }
[data-testid="stSidebar"] { background: #040810; border-right: 1px solid #0E2030; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; background: #0A1018; border-radius: 3px; border: 1px solid #0E2030; color: #0E2030; }
.stTabs [aria-selected="true"] { background: #0E2030 !important; border-color: #00CCFF !important; color: #00CCFF !important; }
hr { border-color: #0E2030 !important; }
</style>
""", unsafe_allow_html=True)

CYAN="#00CCFF"; GREEN="#00FF88"; RED="#FF4466"; GOLD="#FFD700"; ORANGE="#FF8C00"
PURPLE="#CC88FF"; MUTED="#1A3050"; BG="#060A0E"; GRID="#0E2030"; TEXT="#C0D0E0"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=10),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

@st.cache_data(ttl=3600)
def fred(series_id: str, years: int = 5) -> pd.Series:
    try:
        resp = requests.get(FRED_BASE + series_id, timeout=10,
                             headers={"User-Agent": "MacroDashboard/1.0"})
        lines = resp.text.strip().split("\n")
        rows  = [l.split(",") for l in lines[1:] if "." in l]
        df    = pd.DataFrame(rows, columns=["date","value"])
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()
        cutoff = datetime.now() - timedelta(days=365*years)
        return s[s.index >= cutoff]
    except Exception as e:
        print(f"FRED {series_id}: {e}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=300)
def yf_price(ticker: str, period: str = "1y") -> pd.Series:
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True,
                           threads=False, progress=False)["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.squeeze()
        return raw.dropna()
    except Exception:
        return pd.Series(dtype=float)

with st.sidebar:
    st.markdown("## 🌍 Macro Dashboard")
    st.markdown("---")
    lookback = st.selectbox("History", ["1 Year","2 Years","5 Years","10 Years"], index=1)
    years = {"1 Year":1,"2 Years":2,"5 Years":5,"10 Years":10}[lookback]
    st.markdown("---")
    st.caption("Data: FRED (Federal Reserve)\nYahoo Finance\nAll free, no API key")

st.markdown("# 🌍 Macro Intelligence Dashboard")
st.markdown(f"`{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC` · `FRED + Yahoo Finance` · `{lookback}`")
st.markdown("---")

# ── load key data ──────────────────────────────────────────────────────────────
with st.spinner("Loading macro data from FRED..."):
    yc_10y2y  = fred("T10Y2Y",    years)    # yield curve 10Y-2Y
    real_yield = fred("DFII10",   years)    # 10Y real yield (TIPS)
    nom_10y    = fred("DGS10",    years)    # 10Y nominal yield
    cpi_yoy    = fred("CPIAUCSL", years)    # CPI
    pmi        = fred("NAPM",     years)    # ISM PMI
    unrate     = fred("UNRATE",   years)    # unemployment
    fed_funds   = fred("FEDFUNDS", years)   # Fed funds rate

# VIX term structure from Yahoo
vix_spot = yf_price("^VIX",  "1y")
vix3m    = yf_price("^VIX3M","1y")
vix9d    = yf_price("^VIX9D","1y")
dxy      = yf_price("DX-Y.NYB","2y")   # DXY
spy      = yf_price("SPY",   "2y")
hyg      = yf_price("HYG",   "2y")     # high yield — credit proxy
lqd      = yf_price("LQD",   "2y")     # investment grade — credit proxy
tlt      = yf_price("TLT",   "2y")     # long duration bonds

# ── quick metrics ──────────────────────────────────────────────────────────────
def last(s, default="N/A"):
    return f"{float(s.dropna().iloc[-1]):.2f}" if len(s.dropna()) > 0 else default

def chg(s):
    if len(s.dropna()) < 2: return ""
    d = float(s.dropna().iloc[-1]) - float(s.dropna().iloc[-2])
    return f"{'▲' if d>0 else '▼'} {abs(d):.3f}"

c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("Yield Curve\n10Y-2Y",  last(yc_10y2y)+"%",  chg(yc_10y2y))
c2.metric("Real Yield\n10Y TIPS", last(real_yield)+"%", chg(real_yield))
c3.metric("10Y Nom\nYield",       last(nom_10y)+"%",    chg(nom_10y))
c4.metric("CPI\nYoY",             last(cpi_yoy)+"%",    chg(cpi_yoy))
c5.metric("Fed Funds\nRate",      last(fed_funds)+"%",  chg(fed_funds))
c6.metric("DXY",                  last(dxy),            chg(dxy))
c7.metric("VIX\nSpot",            last(vix_spot),       chg(vix_spot))
c8.metric("PMI\nManufacturing",   last(pmi),            chg(pmi))

# regime label
try:
    yc_now = float(yc_10y2y.dropna().iloc[-1])
    cpi_now = float(cpi_yoy.dropna().iloc[-1])
    pmi_now = float(pmi.dropna().iloc[-1])
    regime = ("🔴 STAGFLATION"   if pmi_now < 50 and cpi_now > 3.0 else
              "🟡 OVERHEATING"   if pmi_now > 50 and cpi_now > 3.0 else
              "🟢 GOLDILOCKS"    if pmi_now > 50 and cpi_now < 3.0 else
              "🔵 DEFLATION")
    inv = "⚠️ INVERTED" if yc_now < 0 else "✅ Normal"
    st.markdown(f"**Macro Regime:** {regime} &nbsp;&nbsp;|&nbsp;&nbsp; **Yield Curve:** {inv} ({yc_now:+.2f}%)")
except Exception:
    pass

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Yield Curve & Rates",
    "⚡  VIX Term Structure",
    "💀  Credit Spreads",
    "🌐  DXY & Global Liquidity",
    "📊  Economic Indicators",
    "🔗  Cross-Asset",
])

with tab1:
    fig1 = sp.make_subplots(rows=2, cols=2,
        subplot_titles=["10Y-2Y Yield Curve (inversion = recession warning)",
                        "10Y Nominal vs Real Yield",
                        "Inflation Expectations (Nom - Real)",
                        "Fed Funds Rate"],
        vertical_spacing=0.12, horizontal_spacing=0.08)

    # yield curve
    yc = yc_10y2y.dropna()
    fig1.add_trace(go.Scatter(x=yc.index, y=yc, mode="lines",
        line=dict(color=CYAN, width=2)), row=1, col=1)
    fig1.add_hline(y=0, line=dict(color=RED, dash="dash", width=1.5),
                   annotation_text="Inversion", row=1, col=1)
    fig1.add_hrect(y0=-5, y1=0, fillcolor="rgba(255,68,102,0.05)", row=1, col=1)

    # nominal vs real
    fig1.add_trace(go.Scatter(x=nom_10y.dropna().index, y=nom_10y.dropna(),
        mode="lines", name="Nominal 10Y", line=dict(color=GOLD, width=2)), row=1, col=2)
    fig1.add_trace(go.Scatter(x=real_yield.dropna().index, y=real_yield.dropna(),
        mode="lines", name="Real 10Y", line=dict(color=GREEN, width=2)), row=1, col=2)

    # breakeven inflation (nom - real)
    try:
        be = nom_10y.reindex(real_yield.index).dropna() - real_yield.dropna()
        fig1.add_trace(go.Scatter(x=be.index, y=be, mode="lines",
            line=dict(color=ORANGE, width=2), showlegend=False), row=2, col=1)
        fig1.add_hline(y=2.0, line=dict(color=MUTED, dash="dot"), row=2, col=1,
                       annotation_text="Fed 2% target")
    except Exception:
        pass

    # fed funds
    fig1.add_trace(go.Scatter(x=fed_funds.dropna().index, y=fed_funds.dropna(),
        mode="lines", line=dict(color=PURPLE, width=2), showlegend=False), row=2, col=2)

    for r in range(1, 3):
        for c in range(1, 3):
            fig1.update_xaxes(gridcolor=GRID, row=r, col=c)
            fig1.update_yaxes(gridcolor=GRID, ticksuffix="%", row=r, col=c)

    fig1.update_layout(**{**_base, "height": 560, "title": "Interest Rates & Yield Curve"})
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Yield curve inversion (10Y-2Y < 0) has preceded every US recession since 1955, typically by 12-18 months.")

with tab2:
    st.markdown("### VIX Term Structure — shape matters more than level")

    fig2 = go.Figure()
    if len(vix9d.dropna()) > 0 and len(vix_spot.dropna()) > 0 and len(vix3m.dropna()) > 0:
        # plot most recent 60 days
        n = 60
        for name, s, col in [("VIX9D (9-day)", vix9d, RED),
                              ("VIX Spot (30-day)", vix_spot, GOLD),
                              ("VIX3M (3-month)", vix3m, GREEN)]:
            ts = s.dropna().tail(n)
            fig2.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines",
                name=name, line=dict(color=col, width=2)))

        fig2.add_hline(y=20, line=dict(color=MUTED, dash="dot"),
                       annotation_text="VIX 20 = historical average")
        fig2.add_hline(y=30, line=dict(color=RED, dash="dot"),
                       annotation_text="VIX 30 = elevated fear")

    fig2.update_layout(**{**_base, "title": "VIX Term Structure (9D / 30D / 3M)", "height": 380})
    fig2.update_yaxes(title_text="VIX Level")
    st.plotly_chart(fig2, use_container_width=True)

    # contango/backwardation
    try:
        v9  = float(vix9d.dropna().iloc[-1])
        vsp = float(vix_spot.dropna().iloc[-1])
        v3m = float(vix3m.dropna().iloc[-1])
        shape = "CONTANGO (normal)" if v9 < vsp < v3m else (
                "BACKWARDATION (stress)" if v9 > vsp else "FLAT")
        col   = GREEN if "CONTANGO" in shape else RED
        st.markdown(f"**Current VIX structure:** <span style='color:{col};font-family:Share Tech Mono;'>{shape}</span> — 9D={v9:.1f} | Spot={vsp:.1f} | 3M={v3m:.1f}", unsafe_allow_html=True)
        st.caption("Contango = markets calm. Backwardation = near-term fear spikes — often at turning points.")
    except Exception:
        pass

with tab3:
    st.markdown("### Credit Spreads as Leading Equity Indicator")

    fig3 = sp.make_subplots(rows=2, cols=1,
        subplot_titles=["HYG (High Yield) vs LQD (Investment Grade) — Credit Health",
                        "SPY Equity vs HYG Credit — Divergence = Warning"],
        vertical_spacing=0.12)

    if len(hyg.dropna()) > 0 and len(lqd.dropna()) > 0:
        hyg_n = hyg.dropna() / float(hyg.dropna().iloc[0]) * 100
        lqd_n = lqd.dropna() / float(lqd.dropna().iloc[0]) * 100
        fig3.add_trace(go.Scatter(x=hyg_n.index, y=hyg_n, mode="lines",
            name="HYG (High Yield)", line=dict(color=RED, width=2)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=lqd_n.index, y=lqd_n, mode="lines",
            name="LQD (Inv Grade)", line=dict(color=GOLD, width=2)), row=1, col=1)

    if len(spy.dropna()) > 0 and len(hyg.dropna()) > 0:
        spy_n = spy.dropna() / float(spy.dropna().iloc[0]) * 100
        hyg_n2 = hyg.dropna() / float(hyg.dropna().iloc[0]) * 100
        fig3.add_trace(go.Scatter(x=spy_n.index, y=spy_n, mode="lines",
            name="SPY (Equity)", line=dict(color=GREEN, width=2)), row=2, col=1)
        fig3.add_trace(go.Scatter(x=hyg_n2.index, y=hyg_n2, mode="lines",
            name="HYG (Credit)", line=dict(color=RED, width=1.5, dash="dot")), row=2, col=1)

    for r in [1,2]:
        fig3.update_xaxes(gridcolor=GRID, row=r, col=1)
        fig3.update_yaxes(gridcolor=GRID, row=r, col=1)

    fig3.update_layout(**{**_base, "height": 520, "title": "Credit Markets"})
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Credit leads equity. When HYG falls while SPY holds up, equity often follows credit down. This is a classic divergence warning.")

with tab4:
    st.markdown("### DXY — The Global Liquidity Gauge")

    fig4 = sp.make_subplots(rows=2, cols=1,
        subplot_titles=["DXY (US Dollar Index)", "DXY vs Gold (GLD) — Inverse Relationship"],
        vertical_spacing=0.12)

    gld = yf_price("GLD", "2y")

    dxy_ = dxy.dropna()
    fig4.add_trace(go.Scatter(x=dxy_.index, y=dxy_, mode="lines",
        line=dict(color=CYAN, width=2), showlegend=False), row=1, col=1)
    fig4.add_hline(y=100, line=dict(color=MUTED, dash="dot"),
                   annotation_text="DXY 100 = parity", row=1, col=1)

    if len(gld.dropna()) > 0:
        dxy_n = dxy_.reindex(gld.dropna().index).ffill()
        dxy_n = dxy_n / float(dxy_n.dropna().iloc[0]) * 100
        gld_n = gld.dropna() / float(gld.dropna().iloc[0]) * 100
        fig4.add_trace(go.Scatter(x=dxy_n.index, y=dxy_n, mode="lines",
            name="DXY", line=dict(color=CYAN, width=2)), row=2, col=1)
        fig4.add_trace(go.Scatter(x=gld_n.index, y=gld_n, mode="lines",
            name="GLD", line=dict(color=GOLD, width=2)), row=2, col=1)

    for r in [1,2]:
        fig4.update_xaxes(gridcolor=GRID, row=r, col=1)
        fig4.update_yaxes(gridcolor=GRID, row=r, col=1)

    fig4.update_layout(**{**_base, "height": 500, "title": "DXY & Dollar"})
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Strong DXY = dollar strength = pressure on EM, gold, commodities, and risk assets. Weak DXY = global liquidity expansion.")

with tab5:
    fig5 = sp.make_subplots(rows=2, cols=2,
        subplot_titles=["ISM Manufacturing PMI (>50=expansion)",
                        "CPI Year-over-Year (%)",
                        "Unemployment Rate (%)",
                        "Real Yield (10Y TIPS) — Discounts equities"],
        vertical_spacing=0.12, horizontal_spacing=0.08)

    for (s, r, c, col, note) in [
        (pmi,       1, 1, GREEN,  50),
        (cpi_yoy,   1, 2, ORANGE, 2.0),
        (unrate,    2, 1, PURPLE, None),
        (real_yield,2, 2, RED,    0),
    ]:
        sd = s.dropna()
        fig5.add_trace(go.Scatter(x=sd.index, y=sd, mode="lines",
            line=dict(color=col, width=2), showlegend=False), row=r, col=c)
        if note is not None:
            fig5.add_hline(y=note, line=dict(color=MUTED, dash="dot"), row=r, col=c)
        fig5.update_xaxes(gridcolor=GRID, row=r, col=c)
        fig5.update_yaxes(gridcolor=GRID, row=r, col=c)

    fig5.update_layout(**{**_base, "height": 540, "title": "Economic Indicators (FRED)"})
    st.plotly_chart(fig5, use_container_width=True)

with tab6:
    st.markdown("### Cross-Asset Correlation — 60-day rolling")

    assets = {"SPY":spy, "TLT":tlt, "GLD":gld, "DXY":dxy, "HYG":hyg}
    rets   = {k: v.pct_change().dropna() for k, v in assets.items() if len(v.dropna()) > 30}

    if len(rets) >= 3:
        rets_df = pd.DataFrame(rets).dropna()
        corr    = rets_df.tail(60).corr()

        fig6 = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values,2), texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            colorbar=dict(title=dict(text="ρ",font=dict(color=TEXT)), tickfont=dict(color=TEXT), thickness=12),
        ))
        fig6.update_layout(**{**_base, "title": "60-Day Cross-Asset Return Correlation", "height": 420})
        st.plotly_chart(fig6, use_container_width=True)
        st.caption("SPY↑ + TLT↓ = risk-on. SPY↓ + TLT↑ = risk-off. DXY↑ + GLD↓ = dollar strength. Correlations shift in regimes.")