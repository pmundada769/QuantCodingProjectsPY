# WALKTHROUGH.md
# Everything explained from scratch — assuming only CS50

---

## What this project actually does

You are building a **quantitative factor model**. That sounds intimidating. Here is what it means in plain English.

Every month, you rank a list of stocks using a mathematical signal — for example, "which stocks went up the most over the past year." Then you pretend to **buy the top ranked stocks** and **short-sell the bottom ranked stocks.** Then you wait one month, record how much money you made or lost, and repeat.

At the end you have a return history. You measure whether the signal — the ranking method — actually predicted which stocks would go up. If it did, the signal is a **factor**. If it did not, it is noise.

This project tests five factors: momentum, volatility, value, quality, and a composite of all four. It then prints metrics telling you how good each one was.

---

## The mental model before reading any code

Think of the project like a pipeline:

```
PRICES  →  FACTOR SIGNAL  →  RANK STOCKS  →  LONG TOP, SHORT BOTTOM  →  MEASURE RETURNS  →  PRINT METRICS
```

Each file in the project is one stage of that pipeline.

- `data.py` — gets the prices
- `factors.py` — turns prices into a signal
- `backtest.py` — runs the long/short portfolio and measures performance
- `main.py` — connects all of them together and produces output

---

---

# data.py — Getting price data

## What problem does this file solve?

Before you can rank stocks by momentum, you need their price history. This file downloads it from Yahoo Finance.

## The main function: `get_price_data`

```python
def get_price_data(tickers, start_date):
    data = yf.download(tickers, start=start_date, auto_adjust=True, threads=False)["Close"]
    returns = data.pct_change()
    return data, returns
```

**What is `tickers`?**
A Python list of stock ticker symbols. For example `["AAPL", "MSFT"]`. A ticker is the short code a stock trades under on an exchange.

**What is `start_date`?**
A string like `"2015-01-01"`. The function downloads every day of price data from that date until today.

**What is `yf.download(...)`?**
`yf` is the `yfinance` library — a tool that fetches historical price data from Yahoo Finance. `["Close"]` at the end means you only keep the closing price each day, not the open, high, or low.

**What is `auto_adjust=True`?**
Stock prices get adjusted backwards when a company does a stock split or pays a dividend. Without this, a 2-for-1 stock split would look like the price halved overnight, which is wrong. `auto_adjust=True` corrects for this automatically.

**What is `pct_change()`?**
This converts a column of prices into a column of daily percentage returns. So if a stock went from $100 to $102, `pct_change()` gives you `0.02` (2%). You need returns rather than raw prices for most calculations, because a $1 move means very different things for a $10 stock vs a $1,000 stock.

**What does it return?**
Two things: `data` (a table of daily prices, one column per stock) and `returns` (a table of daily percentage changes). The word `return` in finance just means the percentage profit or loss over a period. It is nothing to do with Python's `return` statement, though confusingly both exist in this code.

---

## `get_benchmark_data`

```python
def get_benchmark_data(ticker, start_date):
    data = yf.download(ticker, start=start_date, auto_adjust=True, threads=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    returns = data.pct_change()
    return data, returns
```

This is the same as `get_price_data` but for a single ticker — used to download SPY (the S&P 500 ETF). The `.squeeze()` call converts a one-column DataFrame into a simple Series (a single list of values), because the regime filter later expects a Series, not a table.

---

## `clean_price_data`

```python
def clean_price_data(price_data, max_missing_pct=0.05):
    missing_pct = price_data.isna().mean()
    clean_tickers = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    return price_data[clean_tickers]
```

Some stocks in your list may not have traded for the full history, or Yahoo Finance may have gaps. If a stock is missing more than 5% of its price days, this function removes it before any factors are computed. `isna()` returns True/False for each cell — True if the value is missing. `.mean()` on a True/False column gives you the fraction that are True, which is the fraction missing.

---

---

# factors.py — Turning prices into signals

## What is a "factor"?

A factor is a measurable characteristic of a stock that predicts its future return. Academic research starting in the 1990s found that certain simple calculations — like "stocks that went up a lot last year tend to keep going up" — reliably predicted returns over large samples and long time periods.

Your code tests five of them.

---

## `momentum_factor`

```python
def momentum_factor(price_data):
    long_term  = price_data.pct_change(252)
    short_term = price_data.pct_change(21)
    momentum   = long_term - short_term
    return momentum
```

**The idea:** Stocks that went up a lot over the past year (but not including the most recent month) tend to keep going up next month.

**Why 252?** There are approximately 252 trading days in a calendar year (markets are closed on weekends and holidays).

**Why subtract the last month (`short_term`)?** Research found that the very recent month (last 21 days) actually reverses — stocks that went up last week tend to dip next week. This is called "short-term reversal." Subtracting it leaves you with a cleaner signal. This is the standard "12-minus-1 momentum" used in academic literature.

**What does the output look like?** A table the same shape as your price table. Each cell contains a number like `0.35` (stock is up 35% on a 12-minus-1 basis) or `-0.12` (down 12%). You rank stocks by this number each month.

---

## `volatility_factor`

```python
def volatility_factor(returns):
    volatility = returns.rolling(63).std()
    return volatility
```

**The idea:** Sort stocks by how volatile they have been recently. The low-volatility anomaly in academic research says that boring, stable stocks often outperform exciting, jumpy ones on a risk-adjusted basis — the opposite of what you might expect.

**What is `.rolling(63).std()`?** This computes a rolling standard deviation. `rolling(63)` means "use the last 63 days of data." `.std()` means standard deviation — a measure of how much values spread around their average. A high standard deviation means the stock is very jumpy. A low one means it is stable.

**Why 63?** 63 trading days ≈ 3 months (21 days × 3).

---

## `value_factor`

```python
def value_factor(price_data):
    two_year_return = price_data.pct_change(504)
    value = -two_year_return
    return value
```

**The idea:** "Cheap" stocks (those that have not risen much in price) tend to outperform "expensive" ones over time.

**The honest caveat:** True value investing uses accounting data — a stock's price relative to its book value, earnings, or cash flows. You do not have that data here (it requires a Bloomberg terminal or a paid API like Compustat). Instead, this file uses a proxy: stocks that have not gone up much over the last 2 years are treated as "cheap."

**Why `-two_year_return`?** Because a low past return = high value score. Inverting the sign means "the less it rose, the higher its value rank."

**Why 504?** 504 trading days ≈ 2 years.

---

## `quality_factor`

```python
def quality_factor(returns):
    rolling_mean = returns.rolling(126).mean()
    rolling_std  = returns.rolling(126).std()
    quality = rolling_mean / (rolling_std + 1e-8)
    return quality
```

**The idea:** High-quality companies — stable, profitable, with consistent earnings — tend to outperform erratic ones.

**The proxy:** A rolling Sharpe ratio computed per stock. Mean return divided by standard deviation of returns over the last 6 months. A stock with steady, positive returns gets a high score. A stock with huge swings gets a low score.

**What is `1e-8`?** That is `0.00000001`. It is added to the denominator (the bottom of the division) to prevent dividing by zero if a stock had perfectly constant returns (std = 0). This is a standard defensive coding pattern.

---

## `composite_factor`

```python
def composite_factor(price_data, returns):
    mom  = momentum_factor(price_data)
    val  = value_factor(price_data)
    qual = quality_factor(returns)

    def cross_sectional_zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)

    z_mom  = cross_sectional_zscore(mom)
    z_val  = cross_sectional_zscore(val)
    z_qual = cross_sectional_zscore(qual)

    composite = (z_mom + z_val + z_qual) / 3
    return composite
```

**The idea:** No single factor is perfect. Momentum crashes during sharp reversals. Value underperforms in growth markets. By blending multiple uncorrelated signals, you can smooth out each one's weaknesses.

**What is a z-score?** A z-score rescales a number to say "how many standard deviations above or below the average is this?" For example, if the average momentum score is 0.10 and the standard deviation is 0.05, then a momentum score of 0.15 has a z-score of `(0.15 - 0.10) / 0.05 = 1.0` — it is 1 standard deviation above average.

**Why z-score before blending?** Because momentum scores are in different units than quality scores. Without normalising, one factor's large numbers would dominate. Z-scoring puts them all on the same scale (mean 0, standard deviation 1) so they contribute equally.

**What is `axis=1`?** In a pandas DataFrame, `axis=0` means "go down rows" and `axis=1` means "go across columns." Here `df.mean(axis=1)` computes the average across all stocks for a given date — that is, the cross-sectional average, meaning the average across the universe at that point in time.

---

---

# backtest.py — Testing whether the signal works

## What is a backtest?

A backtest simulates trading as if you had used a strategy in the past. You take your historical factor signals, pretend you rebalanced a portfolio every month based on them, and record the hypothetical returns. This tells you whether the signal had predictive power historically.

**Important caveat:** Past performance does not guarantee future results. Backtests are optimistic by nature. They should be used to understand a strategy's characteristics, not as a guarantee it will work.

---

## `form_long_short_portfolio` — the core function

This is the most important function in the project. It takes a factor (like momentum scores) and simulates the long-short strategy.

```python
monthly_factor  = factor_data.resample("ME").last()
monthly_returns = returns_data.resample("ME").sum()
```

**What is `resample("ME").last()`?** Your factor is computed daily, but you only trade monthly. `resample("ME")` groups all rows by calendar month. `.last()` takes the last value of each month. So instead of 252 rows per year, you now have 12.

**What is `resample("ME").sum()`?** For returns, you sum them across the month to get the total monthly return. Adding daily percentage returns is an approximation (strictly you should compound them), but it is close enough for monthly periods.

```python
for i in range(len(monthly_factor.index) - 1):
    factor_values      = monthly_factor.loc[date]
    next_month_returns = monthly_returns.iloc[i + 1]
```

**The loop:** Each iteration represents one month. You look at the factor values at the end of month `i`, then check what the actual returns were in month `i+1`. This is the key — you use information available at the end of this month to predict what happens next month. This is called a "point-in-time" setup. Getting this wrong (using future information to rank stocks) is called "look-ahead bias" and it produces fake results.

```python
n_buckets = min(10, len(factor_values))
deciles   = pd.qcut(factor_values.rank(), n_buckets, labels=False, duplicates="drop")
```

**What is `pd.qcut`?** It divides your stocks into equal-sized buckets by rank. With 10 stocks and 10 buckets, each bucket has 1 stock. With 100 stocks, each bucket has 10. `duplicates="drop"` handles the edge case where multiple stocks have identical factor scores.

**What is `.rank()`?** It converts raw scores to ordinal ranks. If your momentum scores are `[0.05, 0.30, -0.10]`, the ranks are `[2, 3, 1]`. This makes the portfolio robust to outliers — an extreme score like `10.0` does not distort things, because it just becomes rank 3.

```python
long_mask  = (deciles == n_buckets - 1)   # top bucket
short_mask = (deciles == 0)               # bottom bucket
weights[long_mask]  =  1 / long_mask.sum()
weights[short_mask] = -1 / short_mask.sum()
```

**The long-short construction:** You go long (buy) the top decile and short (sell short) the bottom decile. Long weights are positive. Short weights are negative. Dividing by `.sum()` makes the weights equal within each leg — if there are 5 stocks in the top bucket, each gets weight `1/5 = 0.2`.

**What is short selling?** Borrowing a stock you do not own, selling it, then hoping to buy it back cheaper later. If the stock falls, you profit. In this backtest, short selling is simulated: you take the negative of the short stocks' returns.

```python
long  = next_month_returns[long_mask].mean()
short = next_month_returns[short_mask].mean()
portfolio_returns.append(long - short)
```

**The return for this month:** Average return of the long stocks minus average return of the short stocks. If top-ranked stocks returned +5% and bottom-ranked stocks returned -3%, the strategy made `5% - (-3%) = 8%` that month.

---

## Performance metrics

### `sharpe_ratio`

```python
def sharpe_ratio(returns):
    return np.sqrt(12) * returns.mean() / returns.std()
```

**What it measures:** Return per unit of risk. The higher the better. `mean() / std()` gives the monthly Sharpe. Multiplying by `sqrt(12)` annualises it (because returns compound over time, and risk scales with the square root of time — this is a property of statistics called the square-root-of-time rule).

**What to look for:**
- Below 0.5: the strategy barely earns enough to justify its risk
- 0.5–1.0: decent, might be publishable in a weaker journal
- 1.0–1.5: good, institutional quality
- Above 1.5: very strong, worth investigating further

---

### `sortino_ratio`

```python
def sortino_ratio(returns):
    downside_returns = returns[returns < 0]
    downside_std     = downside_returns.std()
    return np.sqrt(12) * returns.mean() / downside_std
```

**What it measures:** Like Sharpe, but only counts downside volatility as "bad." The Sharpe penalises you equally for being too volatile upwards or downwards. The Sortino says: big upside swings are fine, only big losses should reduce your score.

**When it matters more than Sharpe:** Momentum strategies often have a "positive skew" — they have frequent small gains and occasional large losses. Sortino captures this better.

---

### `calmar_ratio`

```python
def calmar_ratio(returns):
    ann_return = returns.mean() * 12
    dd         = max_drawdown(returns)
    return ann_return / abs(dd)
```

**What it measures:** Annualised return divided by worst drawdown. If you earned 20% per year but the worst loss was 40%, your Calmar is `0.20 / 0.40 = 0.5`. It asks: "how much return did you earn per unit of worst-case pain?"

**Why it matters:** Two strategies with the same Sharpe ratio can have very different drawdown profiles. The Calmar distinguishes them.

---

### `max_drawdown`

```python
def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak       = cumulative.cummax()
    drawdown   = (cumulative - peak) / peak
    return drawdown.min()
```

**What it measures:** The largest peak-to-trough decline in the strategy's history.

**What is `.cumprod()`?** "Cumulative product." Starting from 1, it multiplies each month's return factor together. So if you earn +10%, -5%, +20% in three months: `1.10 × 0.95 × 1.20 = 1.254`. That is your cumulative return — 25.4% total.

**What is `.cummax()`?** "Cumulative maximum." At each point in time, it records the highest value the portfolio has ever reached up to that date. The drawdown is how far you have fallen from that peak.

**Result:** A negative number. `-0.74` means at its worst point, the strategy was down 74% from its previous high.

---

### `rolling_sharpe`

```python
def rolling_sharpe(returns, window=12):
    rolling_mean = returns.rolling(window).mean()
    rolling_std  = returns.rolling(window).std()
    return np.sqrt(12) * rolling_mean / rolling_std
```

**What it measures:** The Sharpe ratio computed over a sliding 12-month window. Instead of one number for the whole history, you get a time series showing whether the strategy was good in early years and bad in later years (or vice versa).

**Why it matters:** A Sharpe of 1.3 overall might hide the fact that the strategy stopped working in the last 3 years. Rolling Sharpe reveals this. If you plot it and it trends toward zero, the factor's edge is decaying — likely because other investors have discovered the same signal and traded it away.

---

### `hit_rate`

```python
def hit_rate(returns):
    return (returns > 0).mean()
```

**What it measures:** Fraction of months where the strategy made money. `(returns > 0)` produces True/False for each month. `.mean()` on True/False gives the fraction that are True.

**What to look for:** 0.50 is random. A consistent factor should hit 0.55 or higher. Important: a high hit rate with small wins and large losses is worse than a lower hit rate with large wins.

---

### `average_win_loss`

```python
def average_win_loss(returns):
    wins   = returns[returns > 0].mean()
    losses = returns[returns < 0].mean()
    return wins, losses
```

**What it measures:** The average size of winning months versus the average size of losing months. Read alongside hit rate to get the full picture. A strategy with a 45% hit rate but wins that are 3× larger than losses is perfectly viable.

---

### `calculate_turnover`

```python
def calculate_turnover(weights):
    return weights.diff().abs().sum(axis=1)
```

**What it measures:** How much the portfolio changes each month.

**What is `.diff()`?** Difference between this month's weights and last month's weights. If you had 20% in AAPL last month and 0% this month, the diff is -0.20. `.abs()` ignores the sign. `.sum(axis=1)` adds across all stocks to get total trading activity.

**Why it matters:** Every time you buy or sell a stock, you pay transaction costs — the bid-ask spread, broker commissions, and market impact. A turnover of 0.70 means 70% of the portfolio is replaced each month, which is expensive in practice. High theoretical Sharpe ratios often collapse when realistic costs are applied.

---

### `transaction_cost_drag`

```python
def transaction_cost_drag(returns, turnover, cost_per_trade=0.001):
    cost = turnover * cost_per_trade
    net  = returns.values - cost.reindex(returns.index).fillna(0).values
    return pd.Series(net, index=returns.index)
```

**What it does:** Subtracts estimated transaction costs from gross returns to give net returns.

**What is `cost_per_trade=0.001`?** 0.001 = 10 basis points (10 bps). One basis point is 0.01%. A round-trip trade (buy + sell) on a liquid large-cap stock typically costs around 5–15 bps in total. 10 bps is a reasonable middle estimate.

**How to change it:** If you think your stocks are less liquid (smaller companies), you might use 0.003 (30 bps). If you have very good execution, you might use 0.0005. The number matters a lot for strategies with high turnover.

---

---

# main.py — The entry point that runs everything

`main.py` is the conductor. It calls all the other files and assembles their outputs into results. Here is what each numbered section does.

---

**[1] Define the ticker universe**
```python
tickers = ["BEEM", "QBTS", "PLTR", ...]
```
The list of stocks you want to study. You can edit this to any Yahoo Finance tickers. More stocks = better statistical power but longer download time. Fewer stocks = fast but results are noisier and less reliable (you need at least 20–30 stocks for decile portfolios to be meaningful).

---

**[2] Get and clean data**
```python
prices, returns = get_price_data(tickers, "2015-01-01")
prices  = clean_price_data(prices)
returns = prices.pct_change()
```
Download prices since 2015, remove stocks with too many gaps, recompute returns from the cleaned price table.

---

**[3] Compute all factors**
```python
momentum  = momentum_factor(prices)
volatility = volatility_factor(returns)
value      = value_factor(prices)
quality    = quality_factor(returns)
composite  = composite_factor(prices, returns)
```
Each function takes the price or returns table and produces a table of the same shape, with factor scores replacing prices.

---

**[4] Backtest all factors**
```python
strategy_returns, dispersion, weights = form_long_short_portfolio(momentum, returns)
```
For each factor, run the monthly long-short simulation. The `_` on lines like `value_returns, _, val_weights` means "I am ignoring this return value" — the dispersion series is not needed for value and quality in this run.

---

**[5] Get the benchmark**
```python
spy_prices, spy_returns = get_benchmark_data("SPY", "2015-01-01")
regime = market_regime_filter(returns.mean(axis=1), spy_returns)
```
Download SPY to determine whether each month was a "bull" month (S&P 500 above its 200-day moving average) or "bear" month (below it). This lets you compare how the strategy performed in different market environments.

---

**[7] Transaction cost drag**
```python
turnover    = calculate_turnover(weights)
net_returns = transaction_cost_drag(strategy_returns_monthly, turnover)
```
Compute how much of the gross return survives after realistic trading costs.

---

**[8] Print metrics**
All the performance numbers are printed here. The two most important are:
- `Sharpe (net of costs)` — whether the strategy survives after you account for trading friction
- `Max Drawdown` — the worst pain you would have endured holding this strategy

---

**[9] Factor comparison**
```python
for name, ret in factor_map.items():
    print(f"{name:<12}  Sharpe: ...   MaxDD: ...   HitRate: ...")
```
A single table comparing all five factors side by side. This is the most useful output for deciding which signal to focus on.

---

**[10–12] Charts**
Three sets of charts:
1. Momentum cumulative returns over time
2. Rolling 12-month Sharpe — is the edge decaying?
3. All five factors on one chart — which dominated which period?

---

**[14] Top/Bottom ranking table**
For the last 12 months, shows which stocks were in the top and bottom of the momentum ranking, and what they actually returned the following month. This is the "did the signal work?" sanity check at the individual stock level.

---

**[15] Forecast**
```python
top_forecast    = latest_momentum.nlargest(TOP_N).index.tolist()
bottom_forecast = latest_momentum.nsmallest(TOP_N).index.tolist()
```
Takes the most recent momentum scores (not yet realised) and outputs the current top and bottom ranked stocks. This is the strategy's live signal — what it would tell you to do today.

**Important:** This is not investment advice. It is a model output. Real implementation requires transaction cost management, position sizing, and risk controls not present here.

---

---

## How to evaluate results — the full picture

Running the code gives you a wall of numbers. Here is how to read them in order.

**Step 1: Is the gross Sharpe above 1.0?**
If not, the factor is not producing strong enough returns to be interesting, even before costs.

**Step 2: Does the net Sharpe still look reasonable?**
Subtract the cost drag. If the Sharpe drops from 1.3 to 0.4 after costs, the strategy is not implementable in practice. High turnover kills momentum strategies.

**Step 3: Is the Max Drawdown survivable?**
A strategy with -74% max drawdown is theoretically profitable but psychologically impossible to hold. You would almost certainly have stopped the strategy at the bottom and locked in losses.

**Step 4: Is the rolling Sharpe stable?**
Open the rolling Sharpe chart. If the line is flat or slightly rising, the strategy is consistent. If it peaks in 2018 and trends to zero by 2024, the edge has been arbitraged away.

**Step 5: Compare factors**
Look at the factor comparison table. Momentum often has the highest Sharpe but the worst drawdown. Quality often has lower Sharpe but much smaller drawdowns. Composite usually sits in between — more balanced.

**Step 6: Does the ranking table make sense?**
Look at the last 12 months of top/bottom picks. Did the top-ranked stocks actually tend to go up the following month? If your top picks are consistently losing money, the factor is inverted on this particular universe.

---

## Common things to change and experiment with

**Make the momentum lookback longer or shorter:**
In `factors.py`, change `252` to `126` (6-month momentum) or `378` (18-month). Shorter lookbacks are noisier but react faster. Longer lookbacks are smoother but lag.

**Use a different universe:**
Change the `tickers` list. Try all FAANG stocks. Try your ten favourite companies. Try 30 random S&P 500 names. The factor's behaviour changes significantly with the universe.

**Make the portfolio equal-weight top/bottom 30% instead of 10%:**
In `backtest.py`, in `form_long_short_portfolio`, the long bucket is `deciles == n_buckets - 1` (top 10%). Change it to select the top 3 deciles: `deciles >= n_buckets - 3`. This reduces concentration but dilutes the signal.

**Lower the transaction cost assumption:**
Change `cost_per_trade=0.001` to `cost_per_trade=0.0005` (5 bps) to see the upper bound of net performance. Raise it to `0.003` (30 bps) for illiquid stocks.

**Change the start date:**
Try `"2010-01-01"` to include the GFC recovery. Try `"2020-01-01"` to study only post-COVID markets. Different periods tell very different stories about the same factor.

---

## What to say about this project in an interview

You built a cross-sectional equity factor model. You implemented five factors: 12-minus-1 momentum, 3-month rolling volatility, a 2-year price-based value proxy, a 6-month rolling Sharpe quality signal, and a z-scored composite. You ran monthly long-short backtests using decile portfolios, computed Sharpe, Sortino, Calmar, max drawdown, hit rate, and win-loss ratios, and estimated net-of-cost performance using a 10 basis point round-trip cost assumption. You also analysed regime sensitivity by splitting bull and bear markets using a 200-day moving average filter on SPY, and computed rolling Sharpe to check for alpha decay over time.

That is a solid, honest answer that demonstrates you understand what you built.

---

*Pranshu Mundada*