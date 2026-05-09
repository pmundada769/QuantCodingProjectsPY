#pca_factors.py

# PCA Risk Factor Model
#
# Applies Principal Component Analysis to a universe of stock returns
# to extract latent risk factors — the underlying sources of variation
# driving returns across the whole universe.
#
# PC1 almost always = market factor (explains 30-60% of variance)
# PC2 often = growth vs value / tech vs defensive split
# PC3+ = sector rotation, style factors, idiosyncratic themes
#
# Rolling PCA shows when the factor structure changes — i.e. when
# correlations shift and the old factor loadings no longer apply.
# This is a regime change detector.

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from dataclasses import dataclass
from typing import Optional

def fetch_prices(tickers: list, start: str = "2018-01-01") -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.ffill().dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


@dataclass
class PCAResult:
    # static PCA on full sample
    explained_variance:     np.ndarray   # fraction explained by each PC
    cumulative_variance:    np.ndarray
    loadings:               pd.DataFrame  # (n_stocks × n_pcs)
    scores:                 pd.DataFrame  # (n_dates × n_pcs) — factor realisations
    n_components:           int
    tickers:                list
    # interpretation helpers
    top_contributors:       dict          # {PC1: [(ticker, loading),...], ...}


@dataclass
class RollingPCAResult:
    # rolling PCA — variance explained by PC1 over time
    dates:               pd.DatetimeIndex
    pc1_variance_explained: pd.Series    # how dominant is PC1 over time
    pc1_loadings_history: pd.DataFrame   # how PC1 loadings evolve
    rolling_correlation:  pd.Series      # avg pairwise corr — crisis indicator


def fetch_prices(tickers: list, start: str = "2018-01-01") -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, threads=False, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.ffill().dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def run_pca(
    returns:      pd.DataFrame,
    n_components: int = 5,
) -> PCAResult:

    # standardise returns before PCA
    # each stock has mean 0 and std 1 so high-vol stocks don't dominate
    scaler   = StandardScaler()
    scaled   = scaler.fit_transform(returns.fillna(0))

    n_comp   = min(n_components, len(returns.columns), len(returns))
    pca      = PCA(n_components=n_comp)
    scores   = pca.fit_transform(scaled)

    # loadings: how much each stock contributes to each factor
    loadings = pd.DataFrame(
        pca.components_.T,
        index   = returns.columns,
        columns = [f"PC{i+1}" for i in range(n_comp)],
    )

    scores_df = pd.DataFrame(
        scores,
        index   = returns.index,
        columns = [f"PC{i+1}" for i in range(n_comp)],
    )

    explained      = pca.explained_variance_ratio_
    cumulative     = np.cumsum(explained)

    # top 5 contributors per PC (by absolute loading)
    top_contributors = {}
    for pc in loadings.columns:
        top = loadings[pc].abs().nlargest(5)
        top_contributors[pc] = [(t, round(loadings.loc[t, pc], 4)) for t in top.index]

    return PCAResult(
        explained_variance  = explained,
        cumulative_variance = cumulative,
        loadings            = loadings,
        scores              = scores_df,
        n_components        = n_comp,
        tickers             = returns.columns.tolist(),
        top_contributors    = top_contributors,
    )


def rolling_pca(
    returns: pd.DataFrame,
    window:  int = 126,   # rolling window in trading days (6 months)
) -> RollingPCAResult:

    dates              = []
    pc1_var_explained  = []
    pc1_loadings_list  = []
    avg_corr_list      = []

    tickers = returns.columns.tolist()

    for i in range(window, len(returns) + 1):
        window_ret = returns.iloc[i-window:i]

        # average pairwise correlation
        corr_mat  = window_ret.corr()
        n         = len(tickers)
        upper_tri = corr_mat.values[np.triu_indices(n, k=1)]
        avg_corr  = float(np.nanmean(upper_tri))

        # PCA on this window
        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(window_ret.fillna(0))
            pca    = PCA(n_components=1)
            pca.fit(scaled)

            pc1_var = float(pca.explained_variance_ratio_[0])
            pc1_load = pca.components_[0]
        except Exception:
            pc1_var  = np.nan
            pc1_load = np.zeros(len(tickers))

        dates.append(returns.index[i-1])
        pc1_var_explained.append(pc1_var)
        pc1_loadings_list.append(pc1_load)
        avg_corr_list.append(avg_corr)

    dates_idx    = pd.DatetimeIndex(dates)
    pc1_series   = pd.Series(pc1_var_explained, index=dates_idx, name="PC1 Var Explained")
    avg_corr_ser = pd.Series(avg_corr_list,      index=dates_idx, name="Avg Pairwise Corr")

    loadings_hist = pd.DataFrame(
        pc1_loadings_list,
        index   = dates_idx,
        columns = tickers,
    )

    return RollingPCAResult(
        dates                  = dates_idx,
        pc1_variance_explained = pc1_series,
        pc1_loadings_history   = loadings_hist,
        rolling_correlation    = avg_corr_ser,
    )


def factor_returns(pca_result: PCAResult, returns: pd.DataFrame) -> pd.DataFrame:
    # long-short portfolio for each PC based on factor loadings
    # long stocks with highest loadings, short those with lowest
    factor_ports = {}
    for pc in pca_result.loadings.columns:
        loads  = pca_result.loadings[pc]
        long   = loads[loads > loads.quantile(0.75)]
        short  = loads[loads < loads.quantile(0.25)]

        long_ret  = returns[long.index].mean(axis=1)
        short_ret = returns[short.index].mean(axis=1)
        factor_ports[pc] = long_ret - short_ret

    return pd.DataFrame(factor_ports).dropna()


def variance_attribution(pca_result: PCAResult) -> pd.DataFrame:
    # how much of each stock's variance is explained by the first N PCs
    # R² of regressing each stock's returns on the PC scores
    rows = []
    for ticker in pca_result.tickers:
        loads    = pca_result.loadings.loc[ticker]
        total_r2 = float((loads**2).sum())   # proportion of variance explained (PCA property)
        rows.append({
            "Ticker":       ticker,
            "PC1 exposure": round(abs(loads["PC1"]), 4),
            "Total R² (all PCs)": round(min(total_r2, 1.0), 4),
            "Idiosyncratic": round(max(1 - total_r2, 0), 4),
        })
    return pd.DataFrame(rows).sort_values("PC1 exposure", ascending=False)