# NLP News Sentiment → Market Signal

Scrapes financial headlines from Yahoo Finance RSS, Reddit r/wallstreetbets, or NewsAPI. Classifies each headline using FinBERT (BERT fine-tuned on financial text), aggregates daily sentiment, builds a z-score signal, and evaluates it with IC. Also trains a logistic regression on keyword features (Fed, hike, pivot, CPI) to predict 5-day SPY direction.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/NLP_Sentiment
source venv/bin/activate
pip install -r requirements.txt

# Dashboard (recommended)
streamlit run app.py

# CLI
python cli.py --ticker AAPL --source rss --model lexicon
python cli.py --ticker SPY  --source sample --start 2020-01-01

# For FinBERT (downloads ~400MB model first run):
pip install transformers torch
```

No API key needed for Yahoo RSS, Reddit, or sample data. Free key at newsapi.org for NewsAPI.

---

## Files

| File | What it does |
|---|---|
| `sentiment.py` | Headline fetchers, FinBERT classifier, lexicon fallback, IC calculation, keyword model |
| `app.py` | Streamlit dashboard — 6 tabs |
| `cli.py` | Terminal interface |
| `requirements.txt` | Dependencies |

---

## The pipeline

```
Headlines (RSS / Reddit / NewsAPI / sample)
    ↓
FinBERT classification → compound score (positive − negative)
    ↓
Daily aggregation → 7-day rolling average → z-score signal
    ↓
IC evaluation: Spearman(signal_t, return_{t+1}) and Spearman(signal_t, return_{t+5})
    ↓
Backtest: long when z > +0.5σ, short when z < −0.5σ
    ↓
Keyword model: logistic regression on Fed/hike/pivot/CPI keyword counts → SPY 5-day direction
```

---

## Dashboard tabs

- **Timeline** — price, daily sentiment bars, rolling z-score signal
- **Signal vs Returns** — scatter of signal vs 5-day forward return, IC shown
- **Strategy P&L** — cumulative return of long/short sentiment strategy
- **Headlines** — horizontal bar chart of recent headlines coloured by score
- **Keyword Model** — coefficient bar chart: which keywords predict SPY direction
- **Distribution** — pie chart of positive/negative/neutral + FinBERT explainer

---

## IC interpretation

| IC | Meaning |
|---|---|
| > 0.10 | Strong — institutional quality alpha signal |
| 0.05–0.10 | Meaningful — worth trading with risk management |
| < 0.05 | Weak — may be noise |

---

## Why FinBERT over a word list

A lexicon scores *"the stock failed to meet expectations"* as neutral — no negative words. FinBERT understands financial context and correctly scores it as negative. It also handles negation, hedging language, and financial jargon. Fine-tuned on 10-K filings, earnings call transcripts, and financial news.

---

## What to say in an interview

I built an end-to-end NLP alpha research pipeline. Headlines are scraped from Yahoo Finance RSS and NewsAPI, classified using FinBERT (ProsusAI/finbert on HuggingFace), and aggregated to a daily z-score sentiment signal. I evaluated signal quality using IC — Spearman rank correlation against 1-day and 5-day forward returns. The lexicon fallback ensures the pipeline runs offline without GPU resources. The keyword model is a logistic regression on Fed/inflation keyword frequencies to predict 5-day SPY direction — this is a genuine macro NLP pipeline, not a toy sentiment project.

---

## Dependencies

```bash
pip install numpy pandas scipy yfinance plotly streamlit requests scikit-learn
pip install transformers torch   # optional — for FinBERT
```

---

*Pranshu Mundada*