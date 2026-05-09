#app.py

# Financial Statement Analyser — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
import yfinance as yf
from financials import (
    fetch_financials, piotroski_fscore, altman_zscore,
    get_market_cap, revenue_growth, margin_trends,
    fetch_mda_text, analyse_mda_with_llm,
)

st.set_page_config(page_title="Financial Analyser", page_icon="📑", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #080C10; color: #C8D8E0; }
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #E8C468 !important; }
h2, h3 { color: #2A3840 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #0E1418; border: 1px solid #141E24; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #1A2830 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.15rem !important; color: #E8C468 !important; }
[data-testid="stSidebar"] { background: #050810; border-right: 1px solid #141E24; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; background: #0E1418; border-radius: 3px; border: 1px solid #141E24; color: #1A2830; }
.stTabs [aria-selected="true"] { background: #141E24 !important; border-color: #E8C468 !important; color: #E8C468 !important; }
hr { border-color: #141E24 !important; }
</style>
""", unsafe_allow_html=True)

GOLD="#E8C468"; TEAL="#2ECC9A"; CORAL="#E74C3C"; BLUE="#3498DB"; MUTED="#2A3840"
BG="#080C10"; GRID="#141E24"; TEXT="#C8D8E0"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## 📑 Financial Analyser")
    st.markdown("---")

    tickers_input = st.text_area("Tickers (one per line)", height=140,
                                  value="AAPL\nMSFT\nNVDA\nMETA\nGOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]
    primary = st.selectbox("Primary ticker to analyse", tickers, index=0)

    st.markdown("---")
    st.markdown("#### LLM MD&A Analysis (optional)")
    anthropic_key = st.text_input("Anthropic API key", type="password",
        help="Free at console.anthropic.com — leave blank for regex fallback")
    run_mda = st.checkbox("Fetch & analyse MD&A section", value=False)

    st.markdown("---")
    st.caption("Data: SEC EDGAR API\nNo API key required for financials\nPiotroski (2000) + Altman (1968)")


@st.cache_data(ttl=86400)
def load_company(ticker):
    fd    = fetch_financials(ticker)
    mcap  = get_market_cap(ticker)
    if fd is None:
        return None, None, None
    fs = piotroski_fscore(fd)
    zs = altman_zscore(fd, market_cap=mcap)
    return fd, fs, zs


@st.cache_data(ttl=86400)
def load_mda(cik, ticker, company, api_key):
    text, date = fetch_mda_text(cik)
    if not text:
        return None
    result = analyse_mda_with_llm(text, ticker, company, api_key)
    if result:
        result.filing_date = date
    return result


with st.spinner(f"Fetching SEC EDGAR data for {primary}..."):
    fd, fscore, zscore = load_company(primary)

if fd is None:
    st.error(f"Could not fetch SEC data for {primary}. Must be a US-listed company with 10-K filings.")
    st.info("Tip: try AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, BAC, JNJ, XOM, KO")
    st.stop()

# run MD&A if requested
mda_result = None
if run_mda and fd.cik:
    with st.spinner("Fetching MD&A from SEC filing..."):
        mda_result = load_mda(fd.cik, primary, fd.company_name, anthropic_key)

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("# 📑 Financial Statement Analyser")
st.markdown(f"`{fd.company_name}` · `{primary}` · `SEC EDGAR 10-K filings`")
st.markdown("---")

f_val  = fscore.total_score if fscore else "N/A"
z_val  = f"{zscore.z_score:.2f}" if zscore else "N/A"
z_zone = zscore.zone if zscore else "N/A"
z_icon = {"Safe":"🟢","Grey":"🟡","Distress":"🔴"}.get(z_zone, "⚪")

rv_growth_val = "N/A"
if fd.revenue is not None and len(fd.revenue) >= 2:
    rv_growth_val = f"{float(fd.revenue.pct_change().dropna().iloc[-1])*100:+.1f}%"

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Company",        fd.company_name[:22])
c2.metric("Piotroski F",    f"{f_val} / 9")
c3.metric("Altman Z",       z_val)
c4.metric("Z Zone",         f"{z_icon} {z_zone}")
c5.metric("Revenue Growth", rv_growth_val)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Piotroski Score",
    "⚠️  Altman Z-Score",
    "📈  Revenue & Margins",
    "🔍  Peer Comparison",
    "🤖  MD&A Analysis",
    "📋  Raw Financials",
])

with tab1:
    if fscore:
        score_colour = TEAL if fscore.total_score >= 7 else (CORAL if fscore.total_score <= 2 else GOLD)
        st.markdown(f"### F-Score: **{fscore.total_score}/9** — {fscore.interpretation}")

        cats = {
            "Profitability": {
                "ROA positive":      fscore.roa_positive,
                "OCF positive":      fscore.ocf_positive,
                "ROA improved":      fscore.roa_increase,
                "Low accruals":      fscore.accruals_low,
            },
            "Leverage / Liquidity": {
                "Debt ratio fell":   fscore.leverage_dec,
                "Current ratio up":  fscore.liquidity_inc,
                "No dilution":       fscore.no_dilution,
            },
            "Operating Efficiency": {
                "Gross margin up":   fscore.margin_inc,
                "Asset turnover up": fscore.turnover_inc,
            },
        }

        for cat, signals in cats.items():
            st.markdown(f"**{cat}**")
            cols = st.columns(len(signals))
            for col, (name, val) in zip(cols, signals.items()):
                col.metric(name, f"{'✅' if val else '❌'} {val}/1")

        # radar chart
        labels = [k for d in cats.values() for k in d]
        values = [v for d in cats.values() for v in d.values()]
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=labels + [labels[0]],
            fill="toself", fillcolor="rgba(232,196,104,0.15)",
            line=dict(color=GOLD, width=2),
        ))
        fig.update_layout(**{**_base,
            "polar": dict(bgcolor=GRID,
                          radialaxis=dict(range=[0,1], gridcolor=MUTED, color=TEXT),
                          angularaxis=dict(gridcolor=MUTED, color=TEXT)),
            "title": f"Piotroski Radar — {primary}", "height": 420})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient EDGAR data for Piotroski F-Score.")

with tab2:
    if zscore:
        zone_col = TEAL if zscore.zone == "Safe" else (MUTED if zscore.zone == "Grey" else CORAL)
        st.markdown(f"### Altman Z-Score: **{zscore.z_score:.4f}** — {z_icon} {zscore.zone}")
        st.markdown(f"*{zscore.interpretation}*")

        d1,d2,d3,d4,d5 = st.columns(5)
        for col, label, val, tip in [
            (d1, "X1 Working Cap/Assets", zscore.x1, "Liquidity"),
            (d2, "X2 Retained E/Assets",  zscore.x2, "Accumulated profit"),
            (d3, "X3 EBIT/Assets",        zscore.x3, "Operating profitability"),
            (d4, "X4 MktCap/Liabilities", zscore.x4, "Leverage buffer"),
            (d5, "X5 Revenue/Assets",     zscore.x5, "Asset efficiency"),
        ]:
            col.metric(label, f"{val:.4f}", help=tip)

        # gauge
        fig2 = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = zscore.z_score,
            title = dict(text="Altman Z-Score", font=dict(color=TEXT)),
            gauge = dict(
                axis  = dict(range=[0, 5], tickcolor=TEXT, tickfont=dict(color=TEXT)),
                bar   = dict(color=zone_col),
                steps = [
                    dict(range=[0, 1.81],    color="rgba(231,76,60,0.25)"),
                    dict(range=[1.81, 2.99], color="rgba(42,56,64,0.25)"),
                    dict(range=[2.99, 5],    color="rgba(46,204,154,0.25)"),
                ],
                threshold = dict(line=dict(color=GOLD, width=3), value=2.99),
            ),
            number = dict(font=dict(color=GOLD)),
        ))
        fig2.update_layout(**{**_base, "height": 360})
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Red zone (< 1.81) = distress. Grey zone (1.81–2.99) = uncertain. Green zone (> 2.99) = safe.")
    else:
        st.warning("Insufficient EDGAR data for Altman Z-Score.")

with tab3:
    rg = revenue_growth(fd)
    mt = margin_trends(fd)

    if rg is not None and len(rg) > 0:
        fig3 = go.Figure(go.Bar(
            x=[str(d.year) for d in rg.index],
            y=rg.values * 100,
            marker_color=[TEAL if v > 0 else CORAL for v in rg.values],
            text=[f"{v*100:+.1f}%" for v in rg.values],
            textposition="outside", textfont=dict(color=TEXT),
        ))
        fig3.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig3.update_layout(**{**_base, "title": f"{primary} — Annual Revenue Growth"})
        fig3.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)

    if mt is not None and len(mt) > 0:
        fig4 = go.Figure()
        colours = [GOLD, TEAL, BLUE]
        for i, col in enumerate(mt.columns):
            fig4.add_trace(go.Scatter(
                x=[str(d.year) for d in mt.index],
                y=mt[col].values * 100,
                mode="lines+markers", name=col,
                line=dict(color=colours[i % len(colours)], width=2),
            ))
        fig4.update_layout(**{**_base, "title": f"{primary} — Margin Trends"})
        fig4.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig4, use_container_width=True)

    if rg is None and mt is None:
        st.warning("Not enough revenue data from EDGAR for this ticker.")

with tab4:
    st.caption(f"Compares Piotroski and Altman scores across: {', '.join(tickers)}")
    if st.button("▶  Run Peer Comparison"):
        peer_rows = []
        prog = st.progress(0)
        for i, t in enumerate(tickers):
            try:
                fd_p, fs_p, zs_p = load_company(t)
                if fd_p:
                    peer_rows.append({
                        "Ticker":     t,
                        "Company":    fd_p.company_name[:25],
                        "Piotroski":  fs_p.total_score if fs_p else "N/A",
                        "Altman Z":   f"{zs_p.z_score:.2f}" if zs_p else "N/A",
                        "Z Zone":     zs_p.zone if zs_p else "N/A",
                        "Rev Growth": (f"{float(fd_p.revenue.pct_change().dropna().iloc[-1])*100:+.1f}%"
                                       if fd_p.revenue is not None and len(fd_p.revenue) >= 2 else "N/A"),
                    })
            except Exception:
                pass
            prog.progress((i+1)/len(tickers))

        if peer_rows:
            peer_df = pd.DataFrame(peer_rows)
            st.dataframe(peer_df, hide_index=True, use_container_width=True)

            # bar chart of Piotroski scores
            numeric = peer_df[peer_df["Piotroski"] != "N/A"].copy()
            numeric["Piotroski"] = numeric["Piotroski"].astype(int)
            if len(numeric) > 0:
                fig_peer = go.Figure(go.Bar(
                    x=numeric["Ticker"], y=numeric["Piotroski"],
                    marker_color=[TEAL if v >= 7 else (CORAL if v <= 2 else GOLD)
                                  for v in numeric["Piotroski"]],
                    text=numeric["Piotroski"].astype(str),
                    textposition="outside", textfont=dict(color=TEXT),
                ))
                fig_peer.add_hline(y=7, line=dict(color=TEAL, dash="dot", width=1),
                                   annotation_text="Strong (≥7)")
                fig_peer.add_hline(y=2, line=dict(color=CORAL, dash="dot", width=1),
                                   annotation_text="Weak (≤2)")
                fig_peer.update_layout(**{**_base, "title": "Piotroski F-Score Peer Comparison"})
                fig_peer.update_yaxes(range=[0, 10])
                st.plotly_chart(fig_peer, use_container_width=True)

with tab5:
    st.markdown("### 🤖 MD&A Section Analysis")
    if not run_mda:
        st.info("Enable 'Fetch & analyse MD&A section' in the sidebar to run this tab.")
        st.caption("Fetches the Management Discussion & Analysis from the most recent 10-K filing via SEC EDGAR, "
                   "then uses Claude to extract risks, growth drivers, and assess management tone.")
    elif mda_result is None:
        st.warning("Could not fetch MD&A for this ticker. SEC filing may not be in standard format.")
    else:
        st.markdown(f"**Filing date:** {mda_result.filing_date or 'Most recent 10-K'}")
        st.markdown(f"**Tone:** {'🟢 Positive' if mda_result.tone=='positive' else ('🔴 Negative' if mda_result.tone=='negative' else '🟡 Cautious')}")

        col_r, col_g = st.columns(2)
        with col_r:
            st.markdown("#### ⚠️ Key Risks")
            for risk in mda_result.key_risks:
                st.markdown(f"- {risk}")
            if not mda_result.key_risks:
                st.caption("No risks extracted")

        with col_g:
            st.markdown("#### 🚀 Growth Drivers")
            for driver in mda_result.growth_drivers:
                st.markdown(f"- {driver}")
            if not mda_result.growth_drivers:
                st.caption("No growth drivers extracted")

        st.markdown("#### 📝 Summary")
        st.markdown(mda_result.llm_summary)

        with st.expander("Raw MD&A excerpt"):
            st.text(mda_result.mda_excerpt)

        if not anthropic_key:
            st.caption("ℹ️ Regex extraction used. Add Anthropic API key for Claude analysis.")

with tab6:
    st.markdown("#### Latest Annual Figures (from 10-K)")
    rows = []
    for name, series in [
        ("Revenue",           fd.revenue),
        ("Net Income",        fd.net_income),
        ("Operating Income",  fd.operating_income),
        ("Gross Profit",      fd.gross_profit),
        ("Total Assets",      fd.total_assets),
        ("Total Liabilities", fd.total_liab),
        ("Current Assets",    fd.current_assets),
        ("Current Liabilities",fd.current_liab),
        ("Long-Term Debt",    fd.long_term_debt),
        ("Operating Cash Flow",fd.operating_cf),
        ("Retained Earnings", fd.retained_earnings),
    ]:
        if series is not None and len(series) >= 1:
            val  = float(series.dropna().iloc[-1])
            year = str(series.dropna().index[-1].year)
            rows.append({"Item": name, "Year": year,
                         "Value ($M)": f"${val/1e6:,.1f}M",
                         "Value ($B)": f"${val/1e9:,.2f}B"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.warning("No financial data retrieved from EDGAR.")