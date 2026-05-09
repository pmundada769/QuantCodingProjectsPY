# Financial Statement Analyser

Pulls 10-K annual filing data directly from SEC EDGAR (free, no key needed). Computes Piotroski F-Score, Altman Z-Score, revenue growth, and margin trends. Optional: fetches the MD&A section and uses Claude to extract key risks, growth drivers, and assess management tone.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"Financial Statements"
source venv/bin/activate
pip install -r requirements.txt

# Dashboard
streamlit run app.py

# CLI
python cli.py --ticker AAPL
python cli.py --tickers AAPL MSFT NVDA META GOOGL
python cli.py --ticker AAPL --mda --key sk-ant-YOUR_KEY_HERE
```

No API key required for financial data. SEC EDGAR is a free official US government API.

For LLM MD&A analysis: free Anthropic key at `console.anthropic.com`.

---

## Files

| File | What it does |
|---|---|
| `financials.py` | SEC EDGAR fetcher, Piotroski, Altman, margin trends, MD&A fetch, LLM analysis |
| `app.py` | Streamlit dashboard — 6 tabs |
| `cli.py` | Terminal interface |
| `requirements.txt` | Dependencies |

---

## Piotroski F-Score (2000)

9-point fundamental quality screen. One point for each:

**Profitability (4 points)**
- ROA positive: earning positive return on assets
- OCF positive: operating cash flow is positive
- ROA improved: more profitable than last year
- Low accruals: cash earnings exceed accounting earnings (quality check)

**Leverage / Liquidity (3 points)**
- Debt ratio fell: becoming less leveraged
- Current ratio improved: better short-term liquidity
- No dilution: no new shares issued

**Operating Efficiency (2 points)**
- Gross margin improved: pricing power increasing
- Asset turnover improved: using assets more productively

**Interpretation:** 8–9 = strong long, 0–2 = potential short, 3–7 = hold and monitor

---

## Altman Z-Score (1968)

`Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5`

| Variable | Formula | Meaning |
|---|---|---|
| X1 | Working capital / Total assets | Short-term liquidity |
| X2 | Retained earnings / Total assets | Accumulated profitability |
| X3 | EBIT / Total assets | Operating efficiency |
| X4 | Market cap / Total liabilities | Leverage buffer |
| X5 | Revenue / Total assets | Asset productivity |

**Zones:** Z > 2.99 = Safe · 1.81–2.99 = Grey · < 1.81 = Distress

---

## MD&A LLM Analysis

With an Anthropic API key, Claude reads the Management Discussion & Analysis section and returns:
- **Summary** of overall financial health
- **Key risks** mentioned by management
- **Growth drivers** and opportunities
- **Tone assessment** (positive / cautious / negative)

Without a key, a regex-based extraction finds risk and growth language automatically.

---

## What to say in an interview

I built a fundamental analysis pipeline pulling 10-K data directly from SEC EDGAR's XBRL API — no paid data vendor. I implemented both the Piotroski F-Score (all 9 signals across profitability, leverage, and efficiency) and the Altman Z-Score from scratch. The peer comparison runs both scores across any user-defined universe, which is how a long/short equity analyst would screen for quality vs distress. I also integrated an LLM layer that fetches the MD&A section from the actual SEC filing and uses Claude to extract structured risk factors and growth drivers — this is the kind of pipeline that fundamental quant teams at long/short equity HFs are starting to build.

---

## Dependencies

```bash
pip install numpy pandas yfinance plotly streamlit requests
pip install anthropic   # optional — for LLM MD&A analysis
```

---

*Pranshu Mundada*