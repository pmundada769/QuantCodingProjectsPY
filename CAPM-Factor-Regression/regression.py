#regression.py

# CAPM and Fama-French 3-Factor regression engine.
# Downloads factor data from Ken French's data library,
# regresses stock returns on market (CAPM), then on market + SMB + HML (FF3).
# Outputs alpha, beta, R², t-stats, and rolling estimates.

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats # type: ignore
from dataclasses import dataclass
from typing import Optional
import io
import zipfile
import requests


@dataclass
class RegressionResult:
    # one row of regression output per model
    ticker:     str
    model:      str             # "CAPM" or "FF3"

    alpha:      float           # intercept — return unexplained by the factors
    alpha_tstat: float          # t-statistic on alpha (> 2 is statistically significant)
    alpha_pval: float           # p-value on alpha

    beta_market: float          # sensitivity to market excess return
    beta_smb:   Optional[float] # sensitivity to Small-Minus-Big (size factor), FF3 only
    beta_hml:   Optional[float] # sensitivity to High-Minus-Low (value factor), FF3 only

    r_squared:  float           # fraction of return variance explained by factors
    adj_r_squared: float        # R² penalised for number of factors
    residual_std: float         # standard deviation of unexplained returns (idiosyncratic vol)

    n_obs:      int             # number of monthly observations used
    start_date: str
    end_date:   str


def fetch_ff3_factors(start: str = "2010-01-01") -> pd.DataFrame:
    # Download Fama-French 3-factor monthly data from Ken French's website.
    # Returns DataFrame with columns: Mkt-RF, SMB, HML, RF (all as decimals)
    #
    # Mkt-RF = market excess return (market return minus risk-free rate)
    # SMB    = Small Minus Big: return of small-cap stocks minus large-cap stocks
    # HML    = High Minus Low: return of high book-to-market (value) minus low (growth)
    # RF     = risk-free rate (1-month T-bill)

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        z    = zipfile.ZipFile(io.BytesIO(resp.content))
        name = [n for n in z.namelist() if n.endswith(".CSV")][0]
        raw  = z.read(name).decode("utf-8")

        # the file has a header section and an annual section — parse monthly only
        lines = raw.split("\n")

        # find start of monthly data (first line that starts with a 6-digit date)
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and len(stripped.split(",")[0].strip()) == 6:
                data_start = i
                break

        # find end of monthly data (blank line or annual section marker)
        data_end = data_start
        for i in range(data_start, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                data_end = i
                break
            first_col = stripped.split(",")[0].strip()
            if first_col and not first_col[0].isdigit():
                data_end = i
                break

        monthly_lines = [lines[0]] + lines[data_start:data_end]  # keep header
        csv_str = "\n".join(monthly_lines)

        df = pd.read_csv(io.StringIO(csv_str), index_col=0)
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m")
        df.index = df.index + pd.offsets.MonthEnd(0)   # snap to month-end
        df.columns = [c.strip() for c in df.columns]
        df = df / 100   # convert from percentage to decimal

        return df[df.index >= pd.Timestamp(start)]

    except Exception as e:
        # if download fails, build a synthetic factor series for offline use
        print(f"[regression] Could not download FF3 factors: {e}")
        print("[regression] Using synthetic factors for demonstration.")
        return _synthetic_factors(start)


def _synthetic_factors(start: str) -> pd.DataFrame:
    # fallback synthetic factors if French's website is unreachable
    np.random.seed(42)
    idx  = pd.date_range(start, periods=120, freq="ME")
    data = {
        "Mkt-RF": np.random.normal(0.007, 0.045, 120),
        "SMB":    np.random.normal(0.002, 0.030, 120),
        "HML":    np.random.normal(0.002, 0.030, 120),
        "RF":     np.full(120, 0.0003),
    }
    return pd.DataFrame(data, index=idx)


def fetch_stock_returns(ticker: str, start: str = "2010-01-01") -> pd.Series:
    # download monthly adjusted close prices and compute monthly returns
    prices  = yf.download(ticker, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    returns.name = ticker
    return returns


def run_capm(
    ticker:      str,
    start:       str = "2010-01-01",
    factors_df:  pd.DataFrame = None,
) -> RegressionResult:
    # CAPM: excess_return = alpha + beta * (Mkt-RF) + epsilon
    # excess return = stock return minus risk-free rate

    factors = factors_df if factors_df is not None else fetch_ff3_factors(start)
    stock   = fetch_stock_returns(ticker, start)

    # align on shared dates
    df = pd.concat([stock, factors], axis=1).dropna()
    if len(df) < 24:
        raise ValueError(f"Not enough data for {ticker} — need at least 24 months")

    excess_return = df[ticker] - df["RF"]
    mkt_rf        = df["Mkt-RF"]

    # OLS regression: y = alpha + beta*x
    slope, intercept, r, p, se = stats.linregress(mkt_rf, excess_return)

    # t-statistic on alpha manually (intercept std error)
    n      = len(df)
    x      = mkt_rf.values
    y      = excess_return.values
    y_hat  = intercept + slope * x
    resid  = y - y_hat
    s2     = np.sum(resid**2) / (n - 2)
    se_int = np.sqrt(s2 * (1/n + np.mean(x)**2 / np.sum((x - np.mean(x))**2)))
    alpha_t = intercept / se_int
    alpha_p = 2 * stats.t.sf(abs(alpha_t), df=n-2)

    r_sq     = r**2
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - 2)

    return RegressionResult(
        ticker        = ticker,
        model         = "CAPM",
        alpha         = intercept,
        alpha_tstat   = alpha_t,
        alpha_pval    = alpha_p,
        beta_market   = slope,
        beta_smb      = None,
        beta_hml      = None,
        r_squared     = r_sq,
        adj_r_squared = adj_r_sq,
        residual_std  = np.std(resid),
        n_obs         = n,
        start_date    = str(df.index[0].date()),
        end_date      = str(df.index[-1].date()),
    )


def run_ff3(
    ticker:      str,
    start:       str = "2010-01-01",
    factors_df:  pd.DataFrame = None,
) -> RegressionResult:
    # Fama-French 3-Factor: excess_return = alpha + b1*(Mkt-RF) + b2*SMB + b3*HML + epsilon
    #
    # SMB (Small Minus Big): captures the size premium — small stocks historically outperform large
    # HML (High Minus Low): captures the value premium — value stocks historically outperform growth
    # A positive beta_smb means the stock behaves like a small-cap
    # A positive beta_hml means the stock behaves like a value stock

    factors = factors_df if factors_df is not None else fetch_ff3_factors(start)
    stock   = fetch_stock_returns(ticker, start)

    df = pd.concat([stock, factors], axis=1).dropna()
    if len(df) < 24:
        raise ValueError(f"Not enough data for {ticker} — need at least 24 months")

    y = (df[ticker] - df["RF"]).values
    X = np.column_stack([
        np.ones(len(df)),       # alpha (intercept)
        df["Mkt-RF"].values,
        df["SMB"].values,
        df["HML"].values,
    ])

    # OLS in matrix form: beta = (X'X)^-1 X'y
    # This is the same as numpy's lstsq but gives us the coefficient vector directly
    coeffs, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    alpha, b_mkt, b_smb, b_hml = coeffs

    y_hat = X @ coeffs
    resid = y - y_hat
    n, k  = len(y), 4   # k = number of parameters including intercept

    # R² and adjusted R²
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_sq   = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k)

    # standard errors via OLS formula: se = sqrt(diag(s² * (X'X)^-1))
    s2  = ss_res / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se  = np.sqrt(np.diag(cov))

    alpha_t = alpha / se[0]
    alpha_p = 2 * stats.t.sf(abs(alpha_t), df=n-k)

    return RegressionResult(
        ticker        = ticker,
        model         = "FF3",
        alpha         = alpha,
        alpha_tstat   = alpha_t,
        alpha_pval    = alpha_p,
        beta_market   = b_mkt,
        beta_smb      = b_smb,
        beta_hml      = b_hml,
        r_squared     = r_sq,
        adj_r_squared = adj_r_sq,
        residual_std  = np.std(resid),
        n_obs         = n,
        start_date    = str(df.index[0].date()),
        end_date      = str(df.index[-1].date()),
    )


def rolling_beta(
    ticker:      str,
    start:       str = "2010-01-01",
    window:      int = 24,          # rolling window in months
    factors_df:  pd.DataFrame = None,
) -> pd.DataFrame:
    # compute rolling CAPM beta and alpha over a sliding window
    # shows whether the stock's market sensitivity is stable or changing over time

    factors = factors_df if factors_df is not None else fetch_ff3_factors(start)
    stock   = fetch_stock_returns(ticker, start)
    df      = pd.concat([stock, factors], axis=1).dropna()

    excess  = df[ticker] - df["RF"]
    mkt     = df["Mkt-RF"]

    results = []
    for i in range(window, len(df) + 1):
        y = excess.iloc[i-window:i].values
        x = mkt.iloc[i-window:i].values
        slope, intercept, r, _, _ = stats.linregress(x, y)
        results.append({
            "Date":  df.index[i-1],
            "Beta":  slope,
            "Alpha": intercept * 12,   # annualise alpha
            "R2":    r**2,
        })

    return pd.DataFrame(results).set_index("Date")


def factor_decomposition(result: RegressionResult, factors_df: pd.DataFrame) -> pd.DataFrame:
    # break down average monthly return into:
    # alpha contribution + market contribution + SMB contribution + HML contribution
    # useful for attribution analysis — what drove the return?

    mkt_mean = factors_df["Mkt-RF"].mean()
    smb_mean = factors_df["SMB"].mean()   if "SMB" in factors_df.columns else 0
    hml_mean = factors_df["HML"].mean()   if "HML" in factors_df.columns else 0

    rows = [
        {"Component": "Alpha (monthly)",          "Contribution": result.alpha},
        {"Component": "Market (β × E[Mkt-RF])",  "Contribution": result.beta_market * mkt_mean},
    ]
    if result.model == "FF3":
        rows += [
            {"Component": "SMB (β_SMB × E[SMB])", "Contribution": (result.beta_smb or 0) * smb_mean},
            {"Component": "HML (β_HML × E[HML])", "Contribution": (result.beta_hml or 0) * hml_mean},
        ]

    df = pd.DataFrame(rows)
    df["Contribution %"] = df["Contribution"] * 100
    return df