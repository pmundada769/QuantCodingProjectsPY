# Quant Projects

A collection of quantitative finance projects I built to develop practical skills in systematic trading, risk modelling, and portfolio construction. I worked through these independently, using AI assistance to help debug code, learn new libraries, and refine implementations — the ideas, structure, and research behind each project are my own.

Built in Python mostly. Each project has a Streamlit dashboard, CLI, and test suite.

---

## Projects

| # | Project | What it does | Key concepts |
|---|---|---|---|
| 1 | [Equity Factor Model](./Equity%20Factor/) | Cross-sectional momentum, value, quality factors with long-short backtesting | Factor construction, decile portfolios, Sharpe/Sortino/Calmar |
| 2 | [Options Pricer](./Options%20Pricer/) | Black-Scholes pricing engine with all Greeks and IV solver | BS-Merton, Delta/Gamma/Vega/Theta, Newton-Raphson IV |
| 3 | [Monte Carlo Simulator](./Monte%20Carlo/) | 10,000 portfolio paths via GBM, VaR, CVaR, probability of ruin | Geometric Brownian Motion, Ito correction, Cholesky correlation |
| 4 | [Efficient Frontier](./Efficient%20Frontier/) | Markowitz mean-variance optimisation, Max Sharpe, Min Vol portfolios | SLSQP optimisation, Capital Market Line, random feasible set |
| 5 | [CAPM / Factor Regression](./CAPM%20Factor/) | CAPM and Fama-French 3-factor regressions with rolling estimates | OLS, matrix algebra, alpha/beta, return decomposition |
| 6 | [Portfolio Tracker](./Portfolio%20Tracker/) | Live P&L, sector exposure, and performance metrics from a holdings CSV | yfinance, Sharpe, drawdown, benchmark comparison |
| 7 | [Correlation Dashboard](./Correlation%20Dashboard/) | Rolling correlation matrix with regime shift detection | Rolling correlation, crisis detection, animated heatmap |
| 8 | [TSMOM](./TSMOM/) | Time-series momentum with GARCH volatility scaling | Moskowitz-Ooi-Pedersen (2012), GARCH, volatility targeting |
| 9 | [Volatility Targeting](./Vol%20Targeting/) | Scales position size so portfolio vol hits a target | Realised vol, vol scaling, trend following |
| 10 | [Pairs Trading](./Pairs%20Trading/) | Statistical arbitrage via cointegration and z-score spreads | Engle-Granger, Johansen, mean reversion, spread z-score |
| 11 | [Macro Regime Model](./Macro%20Regime/) | Classifies economy into growth/inflation quadrants and allocates assets | Bridgewater All Weather, FRED API, PMI/CPI regime detection |
| 12 | [PCA Factor Model](./PCA%20Factor/) | Extracts latent risk factors from a stock universe via PCA | Principal Component Analysis, factor loadings, rolling PCA |
| 13 | [vectorbt Backtest](./VBT%20Backtest/) | Professional momentum backtest with transaction costs and slippage | vectorbt, position sizing, underwater curve |
| 14 | [alphalens Factor Analysis](./Alphalens%20Factor/) | Industry-standard factor evaluation: IC, turnover, quantile returns | Information Coefficient, factor decay, Quantopian methodology |
| 15 | [NLP News Sentiment](./NLP_Sentiment/) | Scrapes financial headlines, classifies sentiment with FinBERT, maps scores to forward returns, trains a signal model | FinBERT, HuggingFace Transformers, NewsAPI, sentiment → alpha pipeline |
| 16 | [Financial Statement Analyser](./Financial_Statements/) | Pulls 10-K/10-Q data from SEC EDGAR, computes Piotroski F-Score, Altman Z-Score, margin trends, LLM MD&A summary | SEC EDGAR API, fundamental scoring, Piotroski, Altman, LLM integration |
| 17 | [Unified Trading Signal Bot](./Signal_Bot/) | Ensemble of all strategy signals → vol-targeted position sizing → paper trade execution via Alpaca API | Signal aggregation, risk overlay, live paper trading, systematic execution |

---

## Stack

```
Python 3.11+
numpy · pandas · scipy · statsmodels · sklearn
yfinance · pandas-datareader · arch
streamlit · plotly
vectorbt · alphalens-reloaded
```

---

## How to run any project

```bash
git clone https://github.com/pmundada769/py-quant
cd "Quant Projects"

# create and activate virtual environment (one time)
python -m venv venv
source venv/bin/activate

# install dependencies for the project you want
cd "Options Pricer"
pip install -r requirements.txt
streamlit run app.py
```

Each project folder contains its own `requirements.txt`, `README.md`, and instructions.

---

## Background

I studied CS50 (Harvard's intro CS course) and built these projects to bridge the gap between programming fundamentals and quantitative finance. The projects progress from pricing theory (Black-Scholes) through portfolio construction (Markowitz), risk modelling (Monte Carlo, VaR), factor research (Fama-French, PCA), and systematic strategy development (TSMOM, pairs trading, macro regimes).

I used AI assistance to help debug implementations, learn new libraries, and refine code quality — the financial logic, research reading, and project structure are my own work.

---

*Pranshu Mundada*
*[github.com/pmundada769](https://github.com/pmundada769)*
