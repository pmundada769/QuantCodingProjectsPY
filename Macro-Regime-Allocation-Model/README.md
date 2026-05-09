# Macro Regime Allocation Model

Classifies the economy into four growth/inflation quadrants using ISM PMI and CPI data from FRED (free, no key needed). Allocates to equities, bonds, gold, and commodities based on the current regime. Backtests the allocation strategy and shows the yield curve as a recession indicator.

Inspired by Bridgewater's All Weather framework and Ray Dalio's work on economic machines.

---

## How to run

```bash
cd Desktop/CODING/"Quant Projects"/"Macro Regime"
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

No API key required. All data from FRED (Federal Reserve Economic Data).

---

## The four quadrants

| Regime | Growth | Inflation | Target Assets |
|---|---|---|---|
| Growth↑ Inflation↑ | Rising | Rising | Commodities, TIPS, EM equities, Gold |
| Growth↑ Inflation↓ | Rising | Falling | Equities, Growth stocks, Credit, Real estate |
| Growth↓ Inflation↓ | Falling | Falling | Long bonds, Gold, Defensive equities |
| Growth↓ Inflation↑ | Falling | Rising | Gold, Commodities, TIPS (stagflation) |

PMI > 50 = expansion (growth rising). CPI YoY > 2.5% = high inflation.

---

## Data sources

| Series | FRED ID | What it measures |
|---|---|---|
| ISM Manufacturing PMI | NAPM | Factory activity — leading growth indicator |
| CPI All Items | CPIAUCSL | Inflation, year-over-year % |
| 10Y-2Y Yield Spread | T10Y2Y | Yield curve — negative = recession warning |
| Unemployment | UNRATE | Labour market health |

---

## What to say in an interview

I built a macro regime classifier using FRED data — ISM PMI for growth and CPI for inflation — mapping the economy into Bridgewater's four All Weather quadrants. The model allocates to a different set of ETFs depending on the current regime: equities and credit when growth is rising and inflation is low, gold and commodities during stagflation, long bonds during deflation. I added the yield curve as a separate recession indicator — inversions have preceded every US recession since 1955, typically by 12-18 months. The backtest runs monthly rebalancing one period lagged to avoid look-ahead bias.

---

## Dependencies

```bash
pip install numpy pandas yfinance plotly streamlit requests
```

---

*Pranshu Mundada*