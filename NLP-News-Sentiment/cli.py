#cli.py

# NLP Sentiment CLI
# python cli.py --ticker AAPL
# python cli.py --ticker SPY --source sample --model lexicon

import argparse
import numpy as np
from sentiment import (
    SentimentClassifier, fetch_rss, fetch_reddit_wsb,
    generate_sample_headlines, score_headlines, build_signal,
    sentiment_backtest, train_keyword_model, FINBERT_AVAILABLE, SKLEARN_AVAILABLE,
)
import yfinance as yf
import pandas as pd

RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"; LIME="\033[92m"
CORAL="\033[91m"; GOLD="\033[93m"; GREY="\033[90m"; WHITE="\033[97m"

def hr(): print(f"{GREY}{'─'*58}{RESET}")
def section(t): hr(); print(f"{BOLD}{WHITE}  {t}{RESET}"); hr()
def row(l, v, col=WHITE): print(f"  {GREY}{l:<22}{RESET}{col}{v}{RESET}")

parser = argparse.ArgumentParser()
parser.add_argument("--ticker",  default="AAPL")
parser.add_argument("--source",  default="rss", choices=["rss","reddit","sample"])
parser.add_argument("--model",   default="lexicon", choices=["finbert","lexicon"])
parser.add_argument("--start",   default="2020-01-01")
args = parser.parse_args()

ticker = args.ticker.upper()
print(f"\n  {BOLD}{CYAN}NLP NEWS SENTIMENT → MARKET SIGNAL{RESET}\n")

if args.source == "rss":
    headlines = fetch_rss(ticker)
elif args.source == "reddit":
    headlines = fetch_reddit_wsb()
else:
    headlines = generate_sample_headlines(ticker)

print(f"  Loaded {len(headlines)} headlines from {args.source}")

use_finbert = (args.model == "finbert") and FINBERT_AVAILABLE
classifier  = SentimentClassifier(use_finbert=use_finbert)
headlines   = score_headlines(headlines, classifier)

section(f"SENTIMENT SUMMARY — {ticker}")
pos_n = sum(1 for h in headlines if h.sentiment == "positive")
neg_n = sum(1 for h in headlines if h.sentiment == "negative")
neu_n = sum(1 for h in headlines if h.sentiment == "neutral")
avg   = float(sum(h.compound for h in headlines) / max(len(headlines), 1))

row("Total headlines",  str(len(headlines)))
row("Positive",         f"{pos_n} ({pos_n/max(len(headlines),1)*100:.0f}%)", LIME)
row("Negative",         f"{neg_n} ({neg_n/max(len(headlines),1)*100:.0f}%)", CORAL)
row("Neutral",          f"{neu_n} ({neu_n/max(len(headlines),1)*100:.0f}%)")
row("Avg compound",     f"{avg:+.4f}", LIME if avg > 0 else CORAL)

try:
    sig    = build_signal(headlines, ticker, start=args.start)
    cur_sig = float(sig.signal.dropna().iloc[-1]) if len(sig.signal.dropna()) > 0 else 0
    direction = "▲ LONG" if cur_sig > 0.5 else ("▼ SHORT" if cur_sig < -0.5 else "— FLAT")
    sig_col   = LIME if cur_sig > 0.5 else (CORAL if cur_sig < -0.5 else WHITE)

    section("SIGNAL METRICS")
    row("IC (1-day fwd)",  f"{sig.ic_1d:.4f}")
    row("IC (5-day fwd)",  f"{sig.ic_5d:.4f}")
    row("Current signal",  f"{cur_sig:+.2f}σ  →  {direction}", sig_col)

    strat = sentiment_backtest(sig)
    if len(strat) > 0:
        ann_ret = float(strat.mean()) * 252
        sharpe  = ann_ret / (float(strat.std()) * np.sqrt(252)) if strat.std() > 0 else 0
        row("Strategy ann ret", f"{ann_ret*100:.2f}%", LIME if ann_ret > 0 else CORAL)
        row("Strategy Sharpe",  f"{sharpe:.4f}", LIME if sharpe > 0.5 else WHITE)
except Exception as e:
    print(f"\n  {CORAL}Signal build failed: {e}{RESET}")

if SKLEARN_AVAILABLE:
    try:
        prices = yf.download("SPY", start=args.start, auto_adjust=True,
                              threads=False, progress=False)["Close"]
        spy_ret = pd.Series(prices.values.flatten(), index=prices.index).pct_change().dropna()
        km = train_keyword_model(headlines, spy_ret)
        if km:
            section("KEYWORD MODEL — FED/HIKE → SPY 5-DAY DIRECTION")
            row("Test accuracy", f"{km.accuracy*100:.1f}%",
                LIME if km.accuracy > 0.55 else WHITE)
            row("Training samples", str(km.n_train))
            row("Test samples",     str(km.n_test))
            top_idx = int(max(range(len(km.coefficients)), key=lambda i: abs(km.coefficients[i])))
            row("Most predictive keyword",
                f"{km.feature_names[top_idx]}  ({km.coefficients[top_idx]:+.3f})")
    except Exception:
        pass

hr(); print()