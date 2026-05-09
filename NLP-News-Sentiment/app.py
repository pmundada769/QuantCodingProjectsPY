#app.py

# NLP News Sentiment → Market Signal
# Run with: streamlit run app.py

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
import yfinance as yf
from sentiment import (
    SentimentClassifier, fetch_rss, fetch_newsapi, fetch_reddit_wsb,
    generate_sample_headlines, score_headlines, build_signal,
    sentiment_backtest, train_keyword_model, FINBERT_AVAILABLE, SKLEARN_AVAILABLE,
)

st.set_page_config(page_title="NLP Sentiment", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #060C14; color: #B8D4E8; }
h1 { font-family: 'JetBrains Mono', monospace !important; color: #00D4FF !important; }
h2, h3 { color: #2A4A60 !important; font-weight: 400 !important; }
[data-testid="metric-container"] { background: #0C1420; border: 1px solid #0E1820; border-radius: 4px; padding: 14px 18px; }
[data-testid="metric-container"] label { font-family: 'JetBrains Mono', monospace !important; font-size: 0.62rem !important; letter-spacing: 0.1em; color: #1A3040 !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.15rem !important; color: #00D4FF !important; }
[data-testid="stSidebar"] { background: #040810; border-right: 1px solid #0E1820; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; background: #0C1420; border-radius: 3px; border: 1px solid #0E1820; color: #1A3040; }
.stTabs [aria-selected="true"] { background: #0E1820 !important; border-color: #00D4FF !important; color: #00D4FF !important; }
hr { border-color: #0E1820 !important; }
</style>
""", unsafe_allow_html=True)

CYAN="#00D4FF"; LIME="#39FF14"; CORAL="#FF4757"; GOLD="#FFA502"; MUTED="#2A4A60"
BG="#060C14"; GRID="#0E1820"; TEXT="#B8D4E8"

_base = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

with st.sidebar:
    st.markdown("## 🧠 NLP Sentiment")
    st.markdown("---")

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    source = st.radio("News source", [
        "Yahoo Finance RSS (no key)",
        "Reddit r/wallstreetbets (no key)",
        "NewsAPI (free key at newsapi.org)",
        "Sample data (offline demo)",
    ], index=0)

    newsapi_key = ""
    if "NewsAPI" in source:
        newsapi_key = st.text_input("NewsAPI key", type="password")

    model_choice = st.radio(
        "Sentiment model",
        ["FinBERT (HuggingFace)" if FINBERT_AVAILABLE else "FinBERT — install transformers torch",
         "Lexicon (fast, offline)"],
        index=0 if FINBERT_AVAILABLE else 1,
    )
    use_finbert  = "FinBERT" in model_choice and FINBERT_AVAILABLE

    threshold   = st.slider("Signal threshold (σ)", 0.2, 1.5, 0.5, step=0.1)
    roll_window = st.slider("Rolling window (days)", 3, 21, 7)
    start_year  = st.selectbox("Price history start", [2018, 2019, 2020, 2021, 2022], index=2)

    st.markdown("---")
    if not FINBERT_AVAILABLE:
        st.warning("FinBERT needs:\n```\npip install transformers torch\n```")
    if not SKLEARN_AVAILABLE:
        st.warning("Keyword model needs:\n```\npip install scikit-learn\n```")
    st.caption("FinBERT: ProsusAI/finbert (HuggingFace)\nIC = Spearman rank correlation")

start = f"{start_year}-01-01"


@st.cache_resource
def get_classifier(use_fb):
    return SentimentClassifier(use_finbert=use_fb)


@st.cache_data(ttl=1800)
def load_headlines(ticker, source, newsapi_key):
    if "Yahoo" in source:
        return fetch_rss(ticker, n=100)
    elif "Reddit" in source:
        return fetch_reddit_wsb(n=100)
    elif "NewsAPI" in source and newsapi_key:
        return fetch_newsapi(ticker, newsapi_key, days=30)
    else:
        return generate_sample_headlines(ticker, days=120)


@st.cache_data(ttl=1800)
def load_spy_returns(start):
    prices = yf.download("SPY", start=start, auto_adjust=True,
                          threads=False, progress=False)["Close"]
    r = pd.Series(prices.values.flatten(), index=prices.index).pct_change().dropna()
    r.name = "SPY"
    return r


with st.spinner("Loading headlines..."):
    headlines = load_headlines(ticker, source, newsapi_key)

classifier = get_classifier(use_finbert)

with st.spinner(f"Classifying {len(headlines)} headlines..."):
    headlines = score_headlines(headlines, classifier)

with st.spinner("Building signal..."):
    try:
        sig          = build_signal(headlines, ticker, start=start, roll_window=roll_window)
        strat_ret    = sentiment_backtest(sig, threshold=threshold)
    except Exception as e:
        st.error(f"Signal build failed: {e}")
        st.stop()

# keyword model on SPY
kw_model = None
with st.spinner("Training keyword model (Fed+hike → SPY direction)..."):
    try:
        spy_ret  = load_spy_returns(start)
        kw_model = train_keyword_model(headlines, spy_ret)
    except Exception:
        pass

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 NLP News Sentiment → Market Signal")
st.markdown(f"`{ticker}` · `{len(headlines)} headlines` · "
            f"`{'FinBERT' if use_finbert else 'lexicon'}` · `{source.split('(')[0].strip()}`")
st.markdown("---")

pos_n    = sum(1 for h in headlines if h.sentiment == "positive")
neg_n    = sum(1 for h in headlines if h.sentiment == "negative")
avg_sent = float(np.mean([h.compound for h in headlines])) if headlines else 0.0
cur_sig  = float(sig.signal.dropna().iloc[-1]) if len(sig.signal.dropna()) > 0 else 0.0
ann_ret  = float(strat_ret.mean()) * 252
sharpe   = ann_ret / (float(strat_ret.std()) * np.sqrt(252)) if strat_ret.std() > 0 else 0.0

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Headlines",      str(len(headlines)))
c2.metric("Positive",       f"{pos_n} ({pos_n/max(len(headlines),1)*100:.0f}%)")
c3.metric("Negative",       f"{neg_n} ({neg_n/max(len(headlines),1)*100:.0f}%)")
c4.metric("IC (5-day)",     f"{sig.ic_5d:.3f}")
c5.metric("Current Signal", f"{cur_sig:+.2f}σ")
c6.metric("Strategy Sharpe",f"{sharpe:.3f}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  Timeline",
    "🔍  Signal vs Returns",
    "💰  Strategy P&L",
    "📰  Headlines",
    "🤖  Keyword Model",
    "🥧  Distribution",
])

with tab1:
    # price + daily sentiment + signal
    prices_raw  = (1 + sig.returns.fillna(0)).cumprod()
    daily_sent  = sig.daily_sentiment.dropna()
    signal_s    = sig.signal.dropna()

    fig = sp.make_subplots(rows=3, cols=1,
        subplot_titles=[f"{ticker} Price (rebased)",
                        "Daily Sentiment Score (green=positive, red=negative)",
                        "Rolling Signal Z-Score  (±0.5σ thresholds)"],
        vertical_spacing=0.10)

    fig.add_trace(go.Scatter(x=prices_raw.index, y=prices_raw.values,
        mode="lines", line=dict(color=CYAN, width=1.8), showlegend=False), row=1, col=1)

    bar_colours = [LIME if v > 0 else CORAL for v in daily_sent.values]
    fig.add_trace(go.Bar(x=daily_sent.index, y=daily_sent.values,
        marker_color=bar_colours, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=signal_s.index, y=signal_s.values,
        mode="lines", line=dict(color=GOLD, width=1.8), showlegend=False,
        fill="tozeroy", fillcolor="rgba(255,165,2,0.07)"), row=3, col=1)
    for lvl, col in [(0.5, LIME), (-0.5, CORAL)]:
        fig.add_hline(y=lvl, line=dict(color=col, dash="dot", width=1), row=3, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=3, col=1)

    for r in range(1, 4):
        fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)

    fig.update_layout(**{**_base, "height": 700, "title": f"{ticker} — Sentiment Signal Timeline"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fwd_5d  = sig.returns.rolling(5).sum().shift(-5)
    aligned = pd.concat([sig.signal, fwd_5d], axis=1).dropna()
    aligned.columns = ["signal", "fwd_5d"]

    fig2 = go.Figure(go.Scatter(
        x=aligned["signal"].values, y=aligned["fwd_5d"].values * 100,
        mode="markers",
        marker=dict(color=[LIME if v > 0 else CORAL for v in aligned["signal"].values],
                    size=5, opacity=0.6),
        hovertemplate="Signal: %{x:.3f}<br>5d Fwd Return: %{y:.2f}%<extra></extra>",
    ))
    fig2.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig2.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig2.update_layout(**{**_base,
        "title": f"{ticker} — Sentiment Signal vs 5-Day Forward Return  (IC={sig.ic_5d:.3f})"})
    fig2.update_xaxes(title_text="Sentiment Signal (z-score)")
    fig2.update_yaxes(title_text="5-Day Forward Return (%)", ticksuffix="%")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"IC = {sig.ic_5d:.4f} · IC > 0.05 = meaningful · IC > 0.10 = strong")

with tab3:
    if len(strat_ret) > 0:
        cum = (1 + strat_ret).cumprod()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=cum.index, y=(cum-1)*100, mode="lines",
            line=dict(color=CYAN, width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)"))
        fig3.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig3.update_layout(**{**_base, "title": f"{ticker} Sentiment Strategy — Cumulative P&L"})
        fig3.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f"Ann. Return: **{ann_ret*100:.2f}%** · Sharpe: **{sharpe:.3f}**")
    else:
        st.info("Not enough data for backtest — try an earlier start year or different source.")

with tab4:
    recent = sorted(headlines, key=lambda h: h.date, reverse=True)[:30]
    scores = [h.compound for h in recent]
    titles = [h.title[:70] + "..." if len(h.title) > 70 else h.title for h in recent]
    colours = [LIME if s > 0.05 else (CORAL if s < -0.05 else GOLD) for s in scores]

    fig4 = go.Figure(go.Bar(
        x=scores, y=titles, orientation="h",
        marker_color=colours,
        text=[f"{s:+.3f}" for s in scores],
        textposition="outside", textfont=dict(color=TEXT, size=9),
    ))
    fig4.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig4.update_layout(**{**_base, "title": "Recent Headlines — Sentiment Scores",
                           "height": max(400, len(recent)*22)})
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("All headlines table"):
        hdf = pd.DataFrame([{
            "Date": h.date, "Headline": h.title[:80],
            "Sentiment": h.sentiment, "Score": f"{h.compound:+.3f}", "Source": h.source,
        } for h in sorted(headlines, key=lambda x: x.date, reverse=True)])
        st.dataframe(hdf, hide_index=True, use_container_width=True)

with tab5:
    st.markdown("### 🤖 Keyword Model: Fed + Hike → SPY 5-Day Direction")
    st.caption("Logistic regression trained on keyword frequency features to predict whether SPY goes up over the next 5 days.")

    if not SKLEARN_AVAILABLE:
        st.warning("Install scikit-learn:\n```\npip install scikit-learn\n```")
    elif kw_model is None:
        st.info("Not enough dated headlines to train keyword model. Try 'Sample data' source with a 2020 start date.")
    else:
        st.markdown(f"**Test Accuracy: {kw_model.accuracy*100:.1f}%** · "
                    f"Training samples: {kw_model.n_train} · Test samples: {kw_model.n_test}")
        st.caption("50% = random. >55% = meaningful edge for a daily macro signal.")

        # feature importance bar
        coef_df = pd.DataFrame({
            "Feature": kw_model.feature_names,
            "Coefficient": kw_model.coefficients,
        }).sort_values("Coefficient")

        fig5 = go.Figure(go.Bar(
            x=coef_df["Coefficient"], y=coef_df["Feature"],
            orientation="h",
            marker_color=[LIME if v > 0 else CORAL for v in coef_df["Coefficient"]],
        ))
        fig5.add_vline(x=0, line=dict(color=MUTED, width=1))
        fig5.update_layout(**{**_base,
            "title": "Keyword Coefficients  (positive = bullish for SPY 5-day)",
            "height": max(400, len(coef_df)*22)})
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("Large positive coefficient on 'pivot' or 'cut' means those words appear more often before SPY rises. "
                   "Large negative on 'hike' means Fed hike language precedes declines.")

with tab6:
    pos_n2 = sum(1 for h in headlines if h.sentiment == "positive")
    neg_n2 = sum(1 for h in headlines if h.sentiment == "negative")
    neu_n2 = sum(1 for h in headlines if h.sentiment == "neutral")

    col_a, col_b = st.columns(2)
    with col_a:
        fig6 = go.Figure(go.Pie(
            labels=["Positive","Negative","Neutral"],
            values=[pos_n2, neg_n2, neu_n2],
            hole=0.45,
            marker=dict(colors=[LIME, CORAL, GOLD]),
            textfont=dict(color=TEXT),
        ))
        fig6.update_layout(**{**_base, "title": "Headline Sentiment Distribution", "height": 360})
        st.plotly_chart(fig6, use_container_width=True)

    with col_b:
        st.markdown("### What is FinBERT?")
        st.markdown(f"""
**FinBERT** is BERT fine-tuned on financial text — 10-K filings, earnings calls, and financial news.
It classifies text as **positive**, **negative**, or **neutral** with a probability score for each.

Key advantage over a word list: FinBERT understands context.
*"The stock failed to meet expectations"* → **negative**, even though no negative words appear.

**Model:** `ProsusAI/finbert` on HuggingFace  
**IC threshold:** > 0.05 meaningful · > 0.10 strong

**FinBERT available:** {"✅ Yes" if FINBERT_AVAILABLE else "❌ No — pip install transformers torch"}  
**Keyword model:** {"✅ sklearn available" if SKLEARN_AVAILABLE else "❌ No — pip install scikit-learn"}
        """)