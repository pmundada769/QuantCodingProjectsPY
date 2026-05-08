#app.py

'''
Monte Carlo Portfolio Simulator — Streamlit Dashboard
Run with: streamlit run app.py
'''
import streamlit as st
import numpy as np
import pandas as pd
from simulator import run_simulation, run_multi_asset_simulation
from charts import (
    paths_chart, terminal_distribution, drawdown_chart,
    var_cvar_bar, cdf_chart, vol_sensitivity_chart,
)

'''page config'''
st.set_page_config(
    page_title = "Monte Carlo Simulator",
    page_icon  = "🎲",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

'''custom CSS - IBM Plex Mono, amber on dark, different from Options Pricer'''
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0B0E14;
    color: #DDE4EE;
}
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #F5A623 !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #8A9BB0 !important; font-weight: 400 !important; }

[data-testid="metric-container"] {
    background: #111520;
    border: 1px solid #1A2030;
    border-radius: 6px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em;
    color: #4A5A70 !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.25rem !important;
    font-weight: 600;
    color: #F5A623 !important;
}
[data-testid="stSidebar"] { background: #080B10; border-right: 1px solid #1A2030; }

.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    background: #111520;
    border-radius: 4px;
    border: 1px solid #1A2030;
    color: #4A5A70;
}
.stTabs [aria-selected="true"] {
    background: #1A2030 !important;
    border-color: #F5A623 !important;
    color: #F5A623 !important;
}
hr { border-color: #1A2030 !important; }
</style>
""", unsafe_allow_html=True)


'''sidebar inputs'''
with st.sidebar:
    st.markdown("## 🎲 Parameters")
    st.markdown("---")

    mode = st.radio("Mode", ["Single Asset", "Multi-Asset Portfolio"], index=0)

    st.markdown("#### Portfolio")
    initial_value  = st.number_input("Initial Value ($)", value=100_000, min_value=1_000, step=1_000)
    n_simulations  = st.select_slider("Simulations", options=[1_000, 2_000, 5_000, 10_000, 20_000], value=10_000)
    horizon_days   = st.slider("Horizon (trading days)", min_value=21, max_value=1260, value=252, step=21)
    ruin_threshold = st.slider("Ruin Threshold (%)", min_value=5, max_value=80, value=20) / 100

    st.markdown("---")

    if mode == "Single Asset":
        st.markdown("#### Return & Risk")
        annual_return = st.slider("Expected Annual Return (%)", min_value=-10, max_value=40, value=8)  / 100
        annual_vol    = st.slider("Annual Volatility (%)",      min_value=1,  max_value=80, value=15)  / 100

        '''preset buttons for common asset classes'''
        st.markdown("#### Quick Presets")
        col1, col2 = st.columns(2)
        if col1.button("S&P 500"):
            annual_return = 0.10
            annual_vol    = 0.18
        if col2.button("60/40"):
            annual_return = 0.07
            annual_vol    = 0.10
        col3, col4 = st.columns(2)
        if col3.button("Bonds"):
            annual_return = 0.04
            annual_vol    = 0.05
        if col4.button("EM Equity"):
            annual_return = 0.09
            annual_vol    = 0.25

    else:
        st.markdown("#### Asset Allocation")
        st.caption("Weights must sum to 100%")

        n_assets = st.number_input("Number of assets", min_value=2, max_value=5, value=3)

        asset_names, weights, ret_list, vol_list = [], [], [], []
        for i in range(int(n_assets)):
            st.markdown(f"**Asset {i+1}**")
            c1, c2, c3 = st.columns(3)
            name   = c1.text_input(f"Name",   value=["Equity","Bonds","Cash","Alts","Cmdty"][i], key=f"name{i}")
            weight = c2.number_input(f"Weight %", value=[60,30,10,0,0][i], min_value=0, max_value=100, key=f"w{i}")
            ret    = c3.number_input(f"Return %", value=[10,4,2,8,6][i],   min_value=-20, max_value=50, key=f"r{i}")
            vol    = st.number_input(f"Vol % (Asset {i+1})", value=[18,5,1,20,22][i], min_value=1, max_value=80, key=f"v{i}")
            asset_names.append(name)
            weights.append(weight / 100)
            ret_list.append(ret / 100)
            vol_list.append(vol / 100)

        st.markdown("**Correlation (between Asset 1 & 2)**")
        corr_12 = st.slider("ρ Asset 1–2", min_value=-1.0, max_value=1.0, value=0.2, step=0.05)

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", use_container_width=True)
    st.caption("Geometric Brownian Motion\nIto-corrected drift")


'''run simulation'''
if mode == "Single Asset":
    result = run_simulation(
        initial_value  = initial_value,
        annual_return  = annual_return,
        annual_vol     = annual_vol,
        n_simulations  = n_simulations,
        n_days         = horizon_days,
        ruin_threshold = ruin_threshold,
    )
else:
    '''build correlation matrix - simplified 2-asset correlation for n assets'''
    n = int(n_assets)
    corr = np.eye(n)
    if n >= 2:
        corr[0, 1] = corr[1, 0] = corr_12

    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total_weight*100:.0f}% — normalising to 100%")
        weights = [w / total_weight for w in weights]

    result = run_multi_asset_simulation(
        weights        = weights,
        annual_returns = ret_list,
        annual_vols    = vol_list,
        correlations   = corr,
        initial_value  = initial_value,
        n_simulations  = n_simulations,
        n_days         = horizon_days,
        ruin_threshold = ruin_threshold,
    )


'''header'''
st.markdown("# 🎲 Monte Carlo Portfolio Simulator")
horizon_label = f"{horizon_days}d" if horizon_days < 252 else f"{horizon_days//252}yr"
st.markdown(
    f"`{n_simulations:,} paths` · `{horizon_label}` · "
    f"`μ={result.annual_return*100:.1f}%` · `σ={result.annual_vol*100:.1f}%`"
)
st.markdown("---")

'''top metrics row'''
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Expected Value",    f"${result.expected_return:,.0f}")
c2.metric("Median Value",      f"${result.median_return:,.0f}")
c3.metric("VaR 95%",           f"${result.var_95:,.0f}")
c4.metric("CVaR 95%",          f"${result.cvar_95:,.0f}")
c5.metric("Prob of Ruin",      f"{result.prob_ruin*100:.1f}%")
c6.metric("Prob of Profit",    f"{result.prob_profit*100:.1f}%")

st.markdown("---")

'''second row of metrics'''
c7, c8, c9, c10, c11, c12 = st.columns(6)
c7.metric("VaR 99%",           f"${result.var_99:,.0f}")
c8.metric("CVaR 99%",          f"${result.cvar_99:,.0f}")
c9.metric("Best Case (p95)",   f"${result.best_case:,.0f}")
c10.metric("Worst Case (p5)",  f"${result.worst_case:,.0f}")
c11.metric("Sharpe (sim)",     f"{result.sharpe:.3f}")
gain = result.expected_return - result.initial_value
c12.metric("Expected Gain",    f"${gain:+,.0f}")

st.markdown("---")

'''tabs'''
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Path Fan",
    "📊  Terminal Distribution",
    "📉  Drawdowns",
    "⚖️  VaR vs CVaR",
    "〰️  CDF",
    "🌡️  Vol Sensitivity",
])

with tab1:
    n_show = st.slider("Paths to display", min_value=50, max_value=500, value=200, step=50)
    fig = paths_chart(result, n_display=n_show)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Amber band = 25th–75th percentile. Outer band = 5th–95th percentile. Solid line = median.")

with tab2:
    fig2 = terminal_distribution(result)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Red region = worst 5% of outcomes. VaR is the boundary. CVaR is the average within it.")

with tab3:
    fig3 = drawdown_chart(result)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Maximum peak-to-trough loss for each simulated path over the full horizon.")

with tab4:
    fig4 = var_cvar_bar(result)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("CVaR is always larger than VaR — it averages the entire tail rather than just its edge.")

with tab5:
    fig5 = cdf_chart(result)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Read as: probability that the final value is below X. VaR is where the CDF crosses 5% and 1%.")

with tab6:
    st.caption("Reruns the simulation at 20 different volatility levels to show how tail risk scales with vol.")
    fig6 = vol_sensitivity_chart(
        initial_value   = initial_value,
        annual_return   = result.annual_return,
        n_simulations   = n_simulations,
        n_days          = horizon_days,
        ruin_threshold  = ruin_threshold,
    )
    st.plotly_chart(fig6, use_container_width=True)

'''risk summary table'''
with st.expander("📋  Full Risk Summary Table"):
    summary = {
        "Metric": [
            "Initial Value", "Expected Final Value", "Median Final Value",
            "Best Case (p95)", "Worst Case (p5)",
            "VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%",
            "Prob of Profit", "Prob of Ruin", "Sharpe Ratio",
            "Expected Gain / Loss",
        ],
        "Value": [
            f"${result.initial_value:,.2f}",
            f"${result.expected_return:,.2f}",
            f"${result.median_return:,.2f}",
            f"${result.best_case:,.2f}",
            f"${result.worst_case:,.2f}",
            f"${result.var_95:,.2f}  ({result.var_95/result.initial_value*100:.1f}% of portfolio)",
            f"${result.var_99:,.2f}  ({result.var_99/result.initial_value*100:.1f}% of portfolio)",
            f"${result.cvar_95:,.2f}  ({result.cvar_95/result.initial_value*100:.1f}% of portfolio)",
            f"${result.cvar_99:,.2f}  ({result.cvar_99/result.initial_value*100:.1f}% of portfolio)",
            f"{result.prob_profit*100:.2f}%",
            f"{result.prob_ruin*100:.2f}%",
            f"{result.sharpe:.4f}",
            f"${result.expected_return - result.initial_value:+,.2f}",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)