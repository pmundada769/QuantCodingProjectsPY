#app.py

# CAPM / Fama-French 3-Factor Regression Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
from regression import (
    fetch_ff3_factors, fetch_stock_returns,
    run_capm, run_ff3, rolling_beta, factor_decomposition,
)
from charts import (
    scatter_regression, rolling_beta_chart,
    factor_betas_bar, decomposition_waterfall, multi_ticker_summary,
)

st.set_page_config(page_title="Factor Regression", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Fira+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Fira Sans', sans-serif; background-color: #080E0A; color: #C8DDD0; }
h1 { font-family: 'Fira Code', monospace !important; color: #2ECC71 !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Fira Sans', sans-serif !important; color: #5D7A6A !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #0D1610; border: 1px solid #142018; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'Fira Code', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #2D4A38 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Fira Code', monospace !important; font-size: 1.15rem !important; color: #2ECC71 !important; }
[data-testid="stSidebar"] { background: #060C08; border-right: 1px solid #142018; }
.stTabs [data-baseweb="tab"] { font-family: 'Fira Code', monospace; font-size: 0.72rem; background: #0D1610; border-radius: 3px; border: 1px solid #142018; color: #2D4A38; }
.stTabs [aria-selected="true"] { background: #142018 !important; border-color: #2ECC71 !important; color: #2ECC71 !important; }
hr { border-color: #142018 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📡 Parameters")
    st.markdown("---")

    ticker_input = st.text_input("Ticker (single)", value="AAPL")
    multi_input  = st.text_area("Multi-ticker compare\n(one per line)", value="AAPL\nMSFT\nNVDA\nMETA\nJPM", height=120)
    start_year   = st.selectbox("Start Year", [2010, 2012, 2014, 2015, 2016, 2018], index=0)
    roll_window  = st.slider("Rolling window (months)", 12, 48, 24, step=6)
    st.markdown("---")
    st.caption("Fama-French 3-Factor\nKen French Data Library\nMonthly returns")

start = f"{start_year}-01-01"
ticker = ticker_input.strip().upper()

@st.cache_data(ttl=3600)
def load_factors(start):
    return fetch_ff3_factors(start)

@st.cache_data(ttl=3600)
def load_results(ticker, start):
    factors = load_factors(start)
    capm    = run_capm(ticker, start, factors)
    ff3     = run_ff3(ticker, start, factors)
    rolling = rolling_beta(ticker, start, window=24, factors_df=factors)
    stock_r = fetch_stock_returns(ticker, start)
    return factors, capm, ff3, rolling, stock_r

try:
    with st.spinner(f"Running regressions for {ticker}..."):
        factors, capm, ff3, rolling, stock_r = load_results(ticker, start)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

excess_r = stock_r - factors.reindex(stock_r.index)["RF"].fillna(0)
mkt_rf   = factors.reindex(stock_r.index)["Mkt-RF"].dropna()
aligned  = pd.concat([excess_r, mkt_rf], axis=1).dropna()

# header
st.markdown("# 📡 CAPM / Factor Regression")
st.markdown(f"`{ticker}` · `{start} → today` · `{ff3.n_obs} monthly obs`")
st.markdown("---")

# CAPM metrics
st.markdown("### CAPM")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Alpha (monthly)",  f"{capm.alpha*100:.4f}%")
c2.metric("Alpha t-stat",     f"{capm.alpha_tstat:.3f}")
c3.metric("Beta (Market)",    f"{capm.beta_market:.4f}")
c4.metric("R²",               f"{capm.r_squared:.4f}")
c5.metric("Obs",              f"{capm.n_obs}")

st.markdown("### Fama-French 3-Factor")
d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("Alpha (monthly)",  f"{ff3.alpha*100:.4f}%")
d2.metric("Alpha t-stat",     f"{ff3.alpha_tstat:.3f}")
d3.metric("β Market",         f"{ff3.beta_market:.4f}")
d4.metric("β SMB",            f"{ff3.beta_smb:.4f}")
d5.metric("β HML",            f"{ff3.beta_hml:.4f}")
d6.metric("R²",               f"{ff3.r_squared:.4f}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  CAPM Scatter",
    "📉  Rolling Beta",
    "⚖️  Factor Betas",
    "🧩  Decomposition",
    "🌐  Multi-Ticker",
])

with tab1:
    fig1 = scatter_regression(
        aligned.iloc[:, 0], aligned.iloc[:, 1], capm
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Each dot is one month. Slope = Beta. Intercept = Alpha. R² shows how much of the return is explained by the market.")

with tab2:
    fig2 = rolling_beta_chart(rolling, ticker)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Rolling 24-month estimates. A rising beta means the stock is becoming more market-sensitive over time.")

with tab3:
    fig3 = factor_betas_bar(capm, ff3)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Positive SMB = behaves like a small-cap. Positive HML = behaves like a value stock.")

with tab4:
    decomp = factor_decomposition(ff3, factors)
    fig4   = decomposition_waterfall(decomp, ticker)
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(decomp.style.format({"Contribution": "{:.5f}", "Contribution %": "{:.4f}%"}),
                 hide_index=True, use_container_width=True)

with tab5:
    multi_tickers = [t.strip().upper() for t in multi_input.split("\n") if t.strip()]
    if st.button("Run Multi-Ticker FF3"):
        results = []
        prog = st.progress(0)
        for i, t in enumerate(multi_tickers):
            try:
                r = run_ff3(t, start, factors)
                results.append(r)
            except Exception:
                pass
            prog.progress((i+1)/len(multi_tickers))
        if results:
            fig5 = multi_ticker_summary(results)
            st.plotly_chart(fig5, use_container_width=True)

            summary_df = pd.DataFrame([{
                "Ticker": r.ticker,
                "Alpha (ann %)": f"{r.alpha*100*12:.2f}%",
                "α t-stat": f"{r.alpha_tstat:.2f}",
                "β Market": f"{r.beta_market:.3f}",
                "β SMB": f"{r.beta_smb:.3f}",
                "β HML": f"{r.beta_hml:.3f}",
                "R²": f"{r.r_squared:.3f}",
            } for r in results])
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
    else:
        st.caption("Click 'Run Multi-Ticker FF3' to compare all tickers above.")