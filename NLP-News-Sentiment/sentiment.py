#natural language processing - NLP
#sentiment.py

# NLP News Sentiment → Market Signal
#
# Pipeline:
#   1. Scrape headlines from Yahoo Finance RSS, Reddit WSB, or NewsAPI
#   2. Classify with FinBERT (ProsusAI/finbert) — falls back to lexicon
#   3. Aggregate to daily compound score, build z-score signal
#   4. Evaluate via IC (Spearman rank correlation vs forward returns)
#   5. Train logistic regression: keyword features → 5-day SPY direction

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import re
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline as hf_pipeline # type: ignore
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression # type: ignore
    from sklearn.preprocessing import StandardScaler # type: ignore
    from sklearn.metrics import accuracy_score # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class Headline:
    date:      str
    title:     str
    source:    str
    ticker:    str   = ""
    sentiment: str   = "neutral"
    pos_score: float = 0.0
    neg_score: float = 0.0
    neu_score: float = 1.0
    compound:  float = 0.0


@dataclass
class SentimentSignal:
    ticker:          str
    daily_sentiment: pd.Series
    daily_volume:    pd.Series
    rolling_7d:      pd.Series
    signal:          pd.Series
    returns:         pd.Series
    ic_1d:           float
    ic_5d:           float


@dataclass
class KeywordModel:
    accuracy:       float
    feature_names:  list
    coefficients:   list
    n_train:        int
    n_test:         int
    predictions:    pd.Series
    actual:         pd.Series


class SentimentClassifier:

    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert and FINBERT_AVAILABLE
        self._pipe = None
        if self.use_finbert:
            try:
                self._pipe = hf_pipeline(
                    "text-classification",
                    model = "ProsusAI/finbert",
                    top_k = None,
                    device = -1,
                )
            except Exception as e:
                print(f"[sentiment] FinBERT load failed: {e} — using lexicon")
                self.use_finbert = False

    def classify_batch(self, texts: list) -> list:
        if self.use_finbert and self._pipe:
            return self._finbert(texts)
        return self._lexicon(texts)

    def _finbert(self, texts: list) -> list:
        results = []
        for i in range(0, len(texts), 32):
            batch = [t[:512] for t in texts[i:i+32]]
            try:
                preds = self._pipe(batch)
                for pred_list in preds:
                    s   = {p["label"].lower(): p["score"] for p in pred_list}
                    pos = s.get("positive", 0.0)
                    neg = s.get("negative", 0.0)
                    neu = s.get("neutral",  0.0)
                    results.append({"sentiment": max(s, key=s.get),
                                    "pos_score": pos, "neg_score": neg,
                                    "neu_score": neu, "compound": pos - neg})
            except Exception:
                results.extend([{"sentiment":"neutral","pos_score":0.0,
                                  "neg_score":0.0,"neu_score":1.0,"compound":0.0}]*len(batch))
        return results

    def _lexicon(self, texts: list) -> list:
        POS = {"beat","beats","record","growth","profit","surge","rally","strong",
               "upgrade","outperform","bullish","buy","raise","positive","exceeds",
               "better","gain","rises","climbs","acquisition","dividend","recovery"}
        NEG = {"miss","misses","loss","decline","fall","plunge","drop","weak",
               "downgrade","underperform","bearish","sell","cut","negative","below",
               "worse","warning","layoff","lawsuit","investigation","fraud","debt",
               "bankruptcy","recession","hike","tightening","fears","concern"}
        results = []
        for text in texts:
            words    = set(re.findall(r'\b\w+\b', text.lower()))
            n_pos    = len(words & POS)
            n_neg    = len(words & NEG)
            total    = max(n_pos + n_neg, 1)
            pos      = n_pos / total
            neg      = n_neg / total
            compound = pos - neg
            sentiment = ("positive" if compound > 0.1 else
                         "negative" if compound < -0.1 else "neutral")
            results.append({"sentiment": sentiment, "pos_score": pos,
                             "neg_score": neg, "neu_score": max(0.0, 1-pos-neg),
                             "compound": compound})
        return results


def fetch_rss(ticker: str, n: int = 100) -> list:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        import xml.etree.ElementTree as ET
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        headlines = []
        for item in root.findall(".//item")[:n]:
            title = item.findtext("title") or ""
            pub   = item.findtext("pubDate") or ""
            try:
                date = datetime.strptime(pub[:16], "%a, %d %b %Y").strftime("%Y-%m-%d")
            except Exception:
                date = datetime.now().strftime("%Y-%m-%d")
            headlines.append(Headline(date=date, title=title, source="Yahoo RSS", ticker=ticker))
        return headlines
    except Exception as e:
        print(f"[sentiment] RSS error: {e}")
        return []


def fetch_newsapi(query: str, api_key: str, days: int = 30) -> list:
    end   = datetime.now()
    start = end - timedelta(days=days)
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"), "language": "en",
            "sortBy": "publishedAt", "pageSize": 100, "apiKey": api_key,
        }, timeout=10)
        resp.raise_for_status()
        return [Headline(date=a["publishedAt"][:10], title=a["title"] or "",
                         source="NewsAPI")
                for a in resp.json().get("articles", []) if a.get("title")]
    except Exception as e:
        print(f"[sentiment] NewsAPI error: {e}")
        return []


def fetch_reddit_wsb(n: int = 100) -> list:
    try:
        resp = requests.get(
            "https://www.reddit.com/r/wallstreetbets/hot.json",
            params  = {"limit": n},
            headers = {"User-Agent": "Mozilla/5.0 FinBot/1.0"},
            timeout = 10,
        )
        resp.raise_for_status()
        posts = resp.json()["data"]["children"]
        return [Headline(
            date   = datetime.fromtimestamp(p["data"]["created_utc"]).strftime("%Y-%m-%d"),
            title  = p["data"]["title"],
            source = "r/wallstreetbets",
        ) for p in posts if p["data"].get("title")]
    except Exception as e:
        print(f"[sentiment] Reddit error: {e}")
        return []


def generate_sample_headlines(ticker: str, days: int = 120) -> list:
    # synthetic headlines with realistic sentiment for demo without API key
    templates = [
        (f"{ticker} beats Q3 earnings, raises full-year guidance",        0.70),
        (f"{ticker} revenue growth accelerates to record levels",          0.60),
        (f"Analysts upgrade {ticker} to Strong Buy on margin expansion",   0.55),
        (f"{ticker} announces $3B share buyback programme",                0.45),
        (f"Fed rate hike fears weigh on {ticker} valuation",              -0.50),
        (f"{ticker} misses earnings expectations, shares fall after hours",-0.70),
        (f"{ticker} warns of revenue headwinds in Q4",                    -0.55),
        (f"Regulatory investigation into {ticker} business practices",    -0.65),
        (f"Fed signals further hikes, {ticker} drops on rate sensitivity", -0.45),
        (f"{ticker} CPI inflation concerns drag on growth outlook",        -0.40),
        (f"{ticker} to present at Goldman Sachs Technology Conference",     0.05),
        (f"{ticker} dividend increased 12% year-over-year",                0.50),
        (f"{ticker} files quarterly 10-Q with SEC",                        0.00),
        (f"Fed pivot speculation boosts {ticker} and growth stocks",       0.55),
        (f"Strong jobs report reduces Fed hike expectations for {ticker}", 0.30),
    ]
    np.random.seed(42)
    headlines = []
    for i in range(days):
        date    = (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d")
        n_today = np.random.randint(1, 5)
        for _ in range(n_today):
            tmpl, base = templates[np.random.randint(len(templates))]
            compound   = float(np.clip(base + np.random.normal(0, 0.12), -1, 1))
            sentiment  = ("positive" if compound > 0.1 else
                          "negative" if compound < -0.1 else "neutral")
            headlines.append(Headline(
                date=date, title=tmpl, source="sample", ticker=ticker,
                sentiment=sentiment, compound=compound,
                pos_score=max(0, compound), neg_score=max(0, -compound),
                neu_score=max(0, 1 - abs(compound)),
            ))
    return headlines


def score_headlines(headlines: list, classifier: SentimentClassifier) -> list:
    # only score headlines that haven't already been scored
    unscored = [h for h in headlines if h.source != "sample" and h.compound == 0.0]
    scored   = [h for h in headlines if not (h.source != "sample" and h.compound == 0.0)]
    if unscored:
        results = classifier.classify_batch([h.title for h in unscored])
        for h, r in zip(unscored, results):
            h.sentiment = r["sentiment"]
            h.pos_score = r["pos_score"]
            h.neg_score = r["neg_score"]
            h.neu_score = r["neu_score"]
            h.compound  = r["compound"]
    return scored + unscored


def build_signal(headlines: list, ticker: str, start: str = "2020-01-01",
                 roll_window: int = 7) -> SentimentSignal:
    from scipy.stats import spearmanr # type: ignore

    df = pd.DataFrame([{"date": h.date, "compound": h.compound} for h in headlines])
    df["date"] = pd.to_datetime(df["date"])
    agg = df.groupby("date").agg(compound=("compound","mean"), volume=("compound","count"))

    prices  = yf.download(ticker, start=start, auto_adjust=True,
                           threads=False, progress=False)["Close"]
    returns = pd.Series(prices.values.flatten(), index=prices.index).pct_change().dropna()

    idx_all    = agg.index.union(returns.index)
    daily_sent = agg["compound"].reindex(idx_all).fillna(0.0)
    daily_vol  = agg["volume"].reindex(idx_all).fillna(0.0)
    daily_ret  = returns.reindex(idx_all)

    roll_sent = daily_sent.rolling(roll_window, min_periods=1).mean()
    roll_mean = roll_sent.rolling(30, min_periods=5).mean()
    roll_std  = roll_sent.rolling(30, min_periods=5).std().replace(0, np.nan)
    signal    = (roll_sent - roll_mean) / roll_std

    def safe_ic(s, r):
        a = pd.concat([s, r], axis=1).dropna()
        if len(a) < 10:
            return 0.0
        ic, _ = spearmanr(a.iloc[:,0].values, a.iloc[:,1].values)
        return float(ic) if not np.isnan(ic) else 0.0

    fwd_1d = daily_ret.shift(-1)
    fwd_5d = daily_ret.rolling(5).sum().shift(-5)

    return SentimentSignal(
        ticker          = ticker,
        daily_sentiment = daily_sent,
        daily_volume    = daily_vol,
        rolling_7d      = roll_sent,
        signal          = signal,
        returns         = daily_ret,
        ic_1d           = safe_ic(signal, fwd_1d),
        ic_5d           = safe_ic(signal, fwd_5d),
    )


def sentiment_backtest(sig: SentimentSignal, threshold: float = 0.5) -> pd.Series:
    pos = pd.Series(0.0, index=sig.signal.index)
    pos[sig.signal >  threshold] =  1.0
    pos[sig.signal < -threshold] = -1.0
    return (pos.shift(1) * sig.returns).dropna()


# ─── keyword model: Fed + hike → SPY 5-day direction ──────────────────────────

KEYWORDS = [
    "fed", "federal reserve", "hike", "rate", "inflation", "cpi",
    "tightening", "dovish", "hawkish", "pivot", "pause", "cut",
    "recession", "gdp", "unemployment", "jobs", "nfp",
    "earnings", "beat", "miss", "guidance", "outlook", "tariff",
]


def build_keyword_features(headlines: list, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = {}
    for date in dates:
        date_str     = date.strftime("%Y-%m-%d")
        day_lines    = [h for h in headlines if h.date == date_str]
        text_blob    = " ".join(h.title.lower() for h in day_lines)
        row          = {kw: text_blob.count(kw) for kw in KEYWORDS}
        row["avg_sentiment"] = (float(np.mean([h.compound for h in day_lines]))
                                if day_lines else 0.0)
        row["n_headlines"]   = len(day_lines)
        rows[date]   = row
    return pd.DataFrame(rows).T.fillna(0.0)


def train_keyword_model(headlines: list, spy_returns: pd.Series) -> Optional[KeywordModel]:
    if not SKLEARN_AVAILABLE:
        return None

    fwd_5d = spy_returns.rolling(5).sum().shift(-5)
    target = (fwd_5d > 0).astype(int)
    X_df   = build_keyword_features(headlines, spy_returns.index).reindex(spy_returns.index).fillna(0.0)

    aligned = pd.concat([X_df, target.rename("target")], axis=1).dropna()
    if len(aligned) < 40:
        return None

    X = aligned.drop("target", axis=1).values.astype(float)
    y = aligned["target"].values.astype(int)

    split   = int(len(X) * 0.70)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_train)
    X_te    = scaler.transform(X_test)

    model   = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    acc     = float(accuracy_score(y_test, y_pred))

    test_dates = aligned.index[split:]

    return KeywordModel(
        accuracy      = acc,
        feature_names = aligned.drop("target", axis=1).columns.tolist(),
        coefficients  = model.coef_[0].tolist(),
        n_train       = len(X_train),
        n_test        = len(X_test),
        predictions   = pd.Series(y_pred, index=test_dates, name="predicted", dtype=int),
        actual        = pd.Series(y_test, index=test_dates, name="actual",    dtype=int),
    )