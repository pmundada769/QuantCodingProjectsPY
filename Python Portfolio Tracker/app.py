#app.py

# Python Portfolio Tracker — Streamlit Dashboard
# Run with: streamlit run app.py
# Edit holdings.csv to match your actual portfolio.

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tracker import load_holdings, fetch_prices, fetch_history, portfolio_daily_returns, holdings_to_df, sector_summary, portfolio_metrics
from charts import cumulative_return_chart, pnl_bar_chart, sector_pie, drawdown_chart, daily_returns_histogram, top_movers_bar

st.set_page_config(page_title="Portfolio Tracker", page_icon="💼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0A080E; color: #D5CCE8; }
h1 { font-family: 'Space Mono', monospace !important; color: #9B59B6 !important; letter-spacing: -0.03em; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #6C5A7C !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #120F18; border: 1px solid #1C1626; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'Space Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #3A2A4A !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Space Mono', monospace !important; font-size: 1.15rem !important; color: #9B59B6 !important; }
[data-testid="stSidebar"] { background: #07060A; border-right: 1px solid #1C1626; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.72rem; background: #120F18; border-radius: 3px; border: 1px solid #1C1626; color: #3A2A4A; }
.stTabs [aria-selected="true"] { background: #1C1626 !important; border-color: #9B59B6 !important; color: #9B59B6 !important; }
hr { border-color: #1C1626 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 💼 Portfolio")
    st.markdown("---")

    uploaded = st.file_uploader("Upload holdings CSV", type="csv")
    st.markdown("#### Or use default `holdings.csv`")
    period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y"], index=2)
    vs_spy = st.checkbox("Compare vs SPY benchmark", value=True)
    st.markdown("---")

    st.markdown("**CSV format:**")
    st.code("ticker,shares,avg_cost,sector\nAAPL,50,145.20,Technology")
    st.caption("Save your own CSV and upload above.")

# load holdings
try:
    if uploaded:
        import io
        holdings = load_holdings(io.StringIO(uploaded.read().decode("utf-8")))
    else:
        holdings = load_holdings("holdings.csv")
except Exception as e:
    st.error(f"Could not load holdings: {e}")
    st.stop()

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_market_data(tickers_tuple, period):
    holdings_local = load_holdings("holdings.csv")
    holdings_local = fetch_prices(holdings_local)
    prices         = fetch_history(holdings_local, period=period)
    spy_raw        = yf.download("SPY", period=period, auto_adjust=True, threads=False, progress=False)["Close"]
    spy_returns    = spy_raw.pct_change().dropna().squeeze()
    return holdings_local, prices, spy_returns

with st.spinner("Fetching live prices..."):
    tickers_key = tuple(h.ticker for h in holdings)
    try:
        holdings, prices, spy_returns = load_market_data(tickers_key, period)
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        st.stop()

port_returns = portfolio_daily_returns(holdings, prices)
metrics      = portfolio_metrics(port_returns, spy_returns if vs_spy else None)
holdings_df  = holdings_to_df(holdings)
sectors_df   = sector_summary(holdings)

# totals
total_value    = holdings_df["Cur Value"].sum()
total_cost     = holdings_df["Cost Basis"].sum()
total_pnl      = holdings_df["P&L ($)"].sum()
total_pnl_pct  = (total_pnl / total_cost * 100) if total_cost > 0 else 0

# header
st.markdown("# 💼 Portfolio Tracker")
st.markdown(f"`{len(holdings)} holdings` · `{period} history` · live prices via yfinance")
st.markdown("---")

# summary metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Value",    f"${total_value:,.0f}")
c2.metric("Total Cost",     f"${total_cost:,.0f}")
pnl_col = "normal" if total_pnl >= 0 else "inverse"
c3.metric("Total P&L",      f"${total_pnl:+,.0f}", delta=f"{total_pnl_pct:+.2f}%")
c4.metric("Sharpe Ratio",   metrics["Sharpe Ratio"])
c5.metric("Max Drawdown",   metrics["Max Drawdown"])
c6.metric("Ann. Return",    metrics["Ann. Return"])

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Cumulative Return",
    "💰  P&L",
    "🥧  Sectors",
    "📉  Drawdown",
    "📊  Distribution",
    "📋  Holdings Table",
])

with tab1:
    fig1 = cumulative_return_chart(port_returns, spy_returns if vs_spy else None)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig2a = pnl_bar_chart(holdings_df)
        st.plotly_chart(fig2a, use_container_width=True)
    with col_b:
        fig2b = top_movers_bar(holdings_df)
        st.plotly_chart(fig2b, use_container_width=True)

with tab3:
    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = sector_pie(sectors_df)
        st.plotly_chart(fig3, use_container_width=True)
    with col_d:
        st.markdown("#### Sector Breakdown")
        st.dataframe(sectors_df.style.format({
            "Value": "${:,.0f}", "PnL": "${:+,.0f}",
            "Weight (%)": "{:.1f}%", "Return (%)": "{:+.2f}%",
        }), hide_index=True, use_container_width=True)

with tab4:
    fig4 = drawdown_chart(port_returns)
    st.plotly_chart(fig4, use_container_width=True)

with tab5:
    fig5 = daily_returns_histogram(port_returns)
    st.plotly_chart(fig5, use_container_width=True)

with tab6:
    st.markdown("#### All Holdings")
    st.dataframe(holdings_df.style.format({
        "Avg Cost": "${:.2f}", "Cur Price": "${:.2f}",
        "Cur Value": "${:,.2f}", "Cost Basis": "${:,.2f}",
        "P&L ($)": "${:+,.2f}", "P&L (%)": "{:+.2f}%",
    }).applymap(lambda v: "color: #1ABC9C" if isinstance(v, str) and v.startswith("$+")
                else ("color: #E74C3C" if isinstance(v, str) and v.startswith("$-") else "")),
    hide_index=True, use_container_width=True)

    with st.expander("📊 Full Performance Metrics"):
        metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)