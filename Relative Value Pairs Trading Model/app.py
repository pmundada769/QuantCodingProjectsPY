#app.py

# Statistical Arbitrage — Pairs Trading Dashboard
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
from pairs import fetch_prices, analyse_pair, scan_all_pairs, johansen_test
from charts import spread_chart, cumulative_pnl, price_comparison, scan_summary_chart

st.set_page_config(page_title="Pairs Trading", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #080810; color: #D0D0F0; }
h1 { font-family: 'JetBrains Mono', monospace !important; color: #6C5CE7 !important; }
h2, h3 { color: #636E72 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #0E0E1C; border: 1px solid #10101E; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'JetBrains Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #2A2A5A !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.15rem !important; color: #6C5CE7 !important; }
[data-testid="stSidebar"] { background: #050508; border-right: 1px solid #10101E; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; background: #0E0E1C; border-radius: 3px; border: 1px solid #10101E; color: #2A2A5A; }
.stTabs [aria-selected="true"] { background: #10101E !important; border-color: #6C5CE7 !important; color: #6C5CE7 !important; }
hr { border-color: #10101E !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚖️ Pairs Trading")
    st.markdown("---")

    st.markdown("#### Universe")
    ticker_input = st.text_area("Tickers (one per line)",
        value="XOM\nCVX\nGS\nMS\nJPM\nBAC\nKO\nPEP\nMCD\nYUM", height=200)
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    start_year = st.selectbox("Start Year", [2015, 2016, 2017, 2018, 2019, 2020], index=0)
    z_window   = st.slider("Z-score window (days)", 20, 120, 60, step=10)
    entry_z    = st.slider("Entry threshold (σ)", 1.0, 3.0, 2.0, step=0.25)
    exit_z     = st.slider("Exit threshold (σ)",  0.0, 1.5, 0.5, step=0.25)

    st.markdown("---")
    st.markdown("#### Specific Pair")
    pa = st.text_input("Ticker A", value="XOM")
    pb = st.text_input("Ticker B", value="CVX")
    st.caption("Engle-Granger + Johansen\nstatsmodels cointegration")

start = f"{start_year}-01-01"

@st.cache_data(ttl=600)
def load_prices(tickers_tuple, start):
    return fetch_prices(list(tickers_tuple), start=start)

@st.cache_data(ttl=600)
def load_pair(a, b, start, window, entry, exit_t):
    prices = fetch_prices([a, b], start=start)
    return prices, analyse_pair(a, b, prices, window=window, entry=entry, exit=exit_t)

@st.cache_data(ttl=600)
def load_scan(tickers_tuple, start, window, entry, exit_t):
    prices = fetch_prices(list(tickers_tuple), start=start)
    scan   = scan_all_pairs(prices, window=window, entry=entry, exit=exit_t)
    joh    = johansen_test(prices) if len(prices.columns) >= 2 else None
    return prices, scan, joh

with st.spinner("Loading data..."):
    try:
        pair_prices, pair_result = load_pair(
            pa.upper(), pb.upper(), start, z_window, entry_z, exit_z
        )
    except Exception as e:
        st.error(f"Pair load error: {e}")
        st.stop()

# header
st.markdown("# ⚖️ Statistical Arbitrage — Pairs Trading")
st.markdown(f"`{pa.upper()} / {pb.upper()}` · `{start} → today` · `z-window={z_window}d` · `entry=±{entry_z}σ`")
st.markdown("---")

coint_label = "✅ COINTEGRATED" if pair_result.cointegrated else "❌ NOT COINTEGRATED"
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Cointegration",  coint_label)
c2.metric("EG p-value",     f"{pair_result.eg_pvalue:.4f}")
c3.metric("Hedge Ratio",    f"{pair_result.hedge_ratio:.4f}")
c4.metric("Half-Life",      f"{pair_result.half_life:.1f}d" if not np.isnan(pair_result.half_life) else "N/A")
c5.metric("Sharpe Ratio",   f"{pair_result.sharpe:.3f}")
c6.metric("# Trades",       str(pair_result.n_trades))

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Spread & Signal",
    "📈  P&L",
    "🔍  Price Comparison",
    "🌐  Universe Scan",
    "📋  Stats",
])

with tab1:
    fig1 = spread_chart(pair_result)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = cumulative_pnl(pair_result, pair_prices[pa.upper()])
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = price_comparison(pair_prices, pa.upper(), pb.upper())
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"Hedge ratio: {pair_result.hedge_ratio:.4f} — for every 1 share of {pa.upper()}, hedge with {pair_result.hedge_ratio:.4f} shares of {pb.upper()}.")

with tab4:
    st.caption(f"Scanning all {len(tickers)*(len(tickers)-1)//2} pairs in the universe for cointegration.")
    if st.button("▶  Run Universe Scan"):
        with st.spinner("Scanning all pairs..."):
            try:
                all_prices, scan_result, joh = load_scan(tuple(tickers), start, z_window, entry_z, exit_z)
                fig4 = scan_summary_chart(scan_result)
                st.plotly_chart(fig4, use_container_width=True)

                st.markdown(f"**{len(scan_result.cointegrated_pairs)} cointegrated pairs** out of {len(scan_result.all_pairs)} tested")

                if scan_result.cointegrated_pairs:
                    scan_df = pd.DataFrame([{
                        "Pair":       f"{r.ticker_a}/{r.ticker_b}",
                        "EG p-val":   f"{r.eg_pvalue:.4f}",
                        "Hedge Ratio":f"{r.hedge_ratio:.4f}",
                        "Half-Life":  f"{r.half_life:.1f}d" if not np.isnan(r.half_life) else "N/A",
                        "Sharpe":     f"{r.sharpe:.3f}",
                        "Max DD":     f"{r.max_drawdown*100:.2f}%",
                        "Trades":     r.n_trades,
                    } for r in sorted(scan_result.cointegrated_pairs, key=lambda x: -x.sharpe)])
                    st.dataframe(scan_df, hide_index=True, use_container_width=True)

                if joh:
                    st.markdown(f"**Johansen test:** {joh['n_cointegrating_vectors']} cointegrating vector(s) at 95% confidence")
            except Exception as e:
                st.error(f"Scan error: {e}")

with tab5:
    stats = {
        "Ticker A":         pa.upper(),
        "Ticker B":         pb.upper(),
        "Cointegrated":     str(pair_result.cointegrated),
        "EG p-value":       f"{pair_result.eg_pvalue:.6f}",
        "Hedge Ratio":      f"{pair_result.hedge_ratio:.6f}",
        "Half-Life (days)": f"{pair_result.half_life:.2f}" if not np.isnan(pair_result.half_life) else "N/A",
        "Sharpe Ratio":     f"{pair_result.sharpe:.4f}",
        "Max Drawdown":     f"{pair_result.max_drawdown*100:.2f}%",
        "# Round Trips":    str(pair_result.n_trades),
        "Spread Mean":      f"{pair_result.spread.mean():.4f}",
        "Spread Std":       f"{pair_result.spread.std():.4f}",
    }
    st.dataframe(pd.DataFrame(stats.items(), columns=["Metric","Value"]),
                 hide_index=True, use_container_width=True)