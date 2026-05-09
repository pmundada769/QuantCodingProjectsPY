#app.py

# PCA Factor Model — Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from pca_factors import fetch_prices, compute_returns, run_pca, rolling_pca, factor_returns, variance_attribution

st.set_page_config(page_title="PCA Factors", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&family=Roboto:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Roboto', sans-serif; background-color: #08080E; color: #C0C8E0; }
h1 { font-family: 'Roboto Mono', monospace !important; color: #7B8EF7 !important; }
h2, h3 { color: #3A4060 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #0E0E18; border: 1px solid #14142A; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'Roboto Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #202040 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Roboto Mono', monospace !important; font-size: 1.15rem !important; color: #7B8EF7 !important; }
[data-testid="stSidebar"] { background: #060610; border-right: 1px solid #14142A; }
.stTabs [data-baseweb="tab"] { font-family: 'Roboto Mono', monospace; font-size: 0.72rem; background: #0E0E18; border-radius: 3px; border: 1px solid #14142A; color: #202040; }
.stTabs [aria-selected="true"] { background: #14142A !important; border-color: #7B8EF7 !important; color: #7B8EF7 !important; }
hr { border-color: #14142A !important; }
</style>
""", unsafe_allow_html=True)

INDIGO="#7B8EF7"; TEAL="#2ECC9A"; CORAL="#E74C3C"; GOLD="#F1C40F"; MUTED="#3A4060"
BG="#08080E"; GRID="#14142A"; TEXT="#C0C8E0"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="Roboto Mono, monospace", color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## 🔬 PCA Factors")
    st.markdown("---")
    ticker_input = st.text_area("Stock Universe", height=200,
        value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nMETA\nJPM\nBAC\nJNJ\nPFE\nXOM\nCVX\nKO\nPG\nHD")
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    start_year  = st.selectbox("Start Year", [2015, 2018, 2019, 2020], index=0)
    n_components = st.slider("Number of PCs", 2, min(10, len(tickers)), 5)
    roll_window  = st.slider("Rolling window (days)", 60, 252, 126, step=21)
    st.markdown("---")
    st.caption("sklearn PCA on standardised daily returns\nRolling window detects regime shifts")

start = f"{start_year}-01-01"

@st.cache_data(ttl=600)
def load_data(tickers_tuple, start, n_comp, window):
    prices  = fetch_prices(list(tickers_tuple), start=start)
    returns = compute_returns(prices)
    pca_res = run_pca(returns, n_components=n_comp)
    roll    = rolling_pca(returns, window=window)
    f_rets  = factor_returns(pca_res, returns)
    var_att = variance_attribution(pca_res)
    return prices, returns, pca_res, roll, f_rets, var_att

with st.spinner("Running PCA..."):
    try:
        prices, returns, pca_res, roll, f_rets, var_att = load_data(
            tuple(tickers), start, n_components, roll_window
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

st.markdown("# 🔬 PCA Risk Factor Model")
st.markdown(f"`{len(pca_res.tickers)} stocks` · `{len(returns)} days` · `{n_components} PCs`")
st.markdown("---")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("PC1 Variance",  f"{pca_res.explained_variance[0]*100:.1f}%")
c2.metric("PC2 Variance",  f"{pca_res.explained_variance[1]*100:.1f}%" if len(pca_res.explained_variance) > 1 else "N/A")
c3.metric(f"Top {n_components} PCs Total", f"{pca_res.cumulative_variance[-1]*100:.1f}%")
c4.metric("Stocks",        str(len(pca_res.tickers)))
c5.metric("PC1 Top Stock", pca_res.top_contributors["PC1"][0][0] if pca_res.top_contributors.get("PC1") else "N/A")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Scree Plot",
    "🌡️  Factor Loadings",
    "📈  Factor Returns",
    "🔄  Rolling PC1",
    "🎯  Variance Attribution",
    "💡  Interpretation",
])

with tab1:
    fig = go.Figure()
    pc_labels = [f"PC{i+1}" for i in range(len(pca_res.explained_variance))]
    fig.add_trace(go.Bar(x=pc_labels, y=pca_res.explained_variance*100,
        marker_color=INDIGO, name="Individual",
        text=[f"{v*100:.1f}%" for v in pca_res.explained_variance],
        textposition="outside", textfont=dict(color=TEXT)))
    fig.add_trace(go.Scatter(x=pc_labels, y=pca_res.cumulative_variance*100,
        mode="lines+markers", name="Cumulative",
        line=dict(color=TEAL, width=2.5), marker=dict(size=8)))
    fig.add_hline(y=80, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text="80% threshold")
    fig.update_layout(**{**_base, "title": "Scree Plot — Variance Explained by Each PC", "height": 400})
    fig.update_yaxes(title_text="Variance Explained (%)", ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PC1 ≈ market factor. PC2 ≈ growth vs value. PC3+ ≈ sector/style rotation.")

with tab2:
    loadings = pca_res.loadings
    fig2 = go.Figure(go.Heatmap(
        z=loadings.T.values,
        x=loadings.index.tolist(),
        y=loadings.columns.tolist(),
        colorscale="RdBu_r", zmid=0,
        text=np.round(loadings.T.values, 3),
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorbar=dict(title=dict(text="Loading", font=dict(color=TEXT)),
                      tickfont=dict(color=TEXT), thickness=12),
    ))
    fig2.update_layout(**{**_base, "title": "Factor Loadings Heatmap", "height": max(300, n_components*60)})
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Red = positive loading (stock moves with factor). Blue = negative (moves opposite).")

with tab3:
    if len(f_rets) > 0:
        fig3 = go.Figure()
        colours = [INDIGO, TEAL, CORAL, GOLD, "#B388FF"]
        for i, pc in enumerate(f_rets.columns[:5]):
            cum = (1 + f_rets[pc]).cumprod()
            fig3.add_trace(go.Scatter(x=cum.index, y=(cum-1)*100,
                mode="lines", name=pc, line=dict(color=colours[i%5], width=2)))
        fig3.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig3.update_layout(**{**_base, "title": "Long-Short Factor Portfolio Returns"})
        fig3.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = sp.make_subplots(rows=2, cols=1,
        subplot_titles=["PC1 Variance Explained — Rolling (crisis = spikes)",
                        "Average Pairwise Correlation — Rolling"],
        vertical_spacing=0.12)

    pc1v = roll.pc1_variance_explained.dropna()
    avg_c = roll.rolling_correlation.dropna()

    fig4.add_trace(go.Scatter(x=pc1v.index, y=pc1v.values*100,
        mode="lines", line=dict(color=INDIGO, width=2),
        fill="tozeroy", fillcolor="rgba(123,142,247,0.08)", showlegend=False), row=1, col=1)

    fig4.add_trace(go.Scatter(x=avg_c.index, y=avg_c.values,
        mode="lines", line=dict(color=TEAL, width=2),
        fill="tozeroy", fillcolor="rgba(46,204,154,0.08)", showlegend=False), row=2, col=1)
    fig4.add_hline(y=0.6, line=dict(color=CORAL, dash="dot", width=1),
                   annotation_text="Crisis threshold", row=2, col=1)

    for r in [1, 2]:
        fig4.update_xaxes(gridcolor=GRID, row=r, col=1)
        fig4.update_yaxes(gridcolor=GRID, row=r, col=1)
    fig4.update_layout(**{**_base, "height": 520, "title": "Rolling Factor Structure"})
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Spikes in PC1 variance = everything moving together = crisis regime. Normal PC1 ≈ 30-50%.")

with tab5:
    st.dataframe(var_att.style.background_gradient(subset=["PC1 exposure"], cmap="Blues")
                              .format({"PC1 exposure":"{:.4f}", "Total R² (all PCs)":"{:.4f}", "Idiosyncratic":"{:.4f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("High idiosyncratic = stock has more company-specific risk. High PC1 = more market-driven.")

with tab6:
    st.markdown("### How to interpret PCA factors")
    st.markdown(f"""
**PC1 explains {pca_res.explained_variance[0]*100:.1f}% of variance** — this is almost certainly the **market factor**. All stocks load positively on it. It represents the systematic risk that cannot be diversified away.

**PC2 explains {pca_res.explained_variance[1]*100:.1f}%** — typically captures **growth vs value** or **tech vs defensive** split. Stocks that load positively include growth names; negative loadings = defensive/value.

**PC3+** — sector rotation, style factors, or macro themes. Their interpretation depends on which stocks load highly.

**Rolling PC1 variance** spikes during market crises (2020 COVID, 2022 rate shock). When everything moves together, PC1 explains 60-80% of variance instead of the normal 30-50%. This is when diversification breaks down.

**Top contributors to PC{1}:** {", ".join([f"{t} ({v:+.3f})" for t, v in pca_res.top_contributors.get("PC1", [])])}
    """)