#charts.py

# All Plotly visualisations for the Efficient Frontier optimizer.
# Returns go.Figure objects consumed by app.py.

import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from optimizer import FrontierResult, random_portfolios

# colour palette: cool slate-blue on near-black
# deliberately different from Options Pricer (teal) and Monte Carlo (amber)
ELECTRIC = "#4F8EF7"   # electric blue — frontier line
GOLD     = "#FFD166"   # gold star — max Sharpe
MINT     = "#06D6A0"   # mint — min vol
CORAL    = "#EF476F"   # coral — individual assets
MUTED    = "#5A6A80"
BG       = "#090F1A"
CARD     = "#0E1826"
GRID     = "#162030"
TEXT     = "#C8D8E8"

_base = dict(
    plot_bgcolor  = BG,
    paper_bgcolor = BG,
    font          = dict(family="Syne Mono, monospace", color=TEXT, size=11),
    xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    margin        = dict(l=60, r=30, t=60, b=60),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, font=dict(size=11)),
)

def _lay(fig, **kw):
    fig.update_layout(**{**_base, **kw})


# ─── Main frontier chart ──────────────────────────────────────────────────

def frontier_chart(result: FrontierResult, show_random: bool = True) -> go.Figure:
    # The signature Markowitz chart:
    # x-axis = portfolio volatility (risk)
    # y-axis = portfolio expected return
    # The frontier is the upper-left edge of all possible portfolios

    fig = go.Figure()

    # random portfolio cloud — the "feasible set"
    if show_random:
        rand_df = random_portfolios(
            result.mean_returns,
            result.cov_matrix,
            result.tickers,
            n=2500,
            risk_free_rate=result.risk_free_rate,
        )
        fig.add_trace(go.Scatter(
            x    = rand_df["Volatility"] * 100,
            y    = rand_df["Return"]     * 100,
            mode = "markers",
            marker = dict(
                color       = rand_df["Sharpe"],
                colorscale  = "Blues",
                size        = 3,
                opacity     = 0.35,
                showscale   = True,
                colorbar    = dict(
                    title      = "Sharpe",
                    thickness  = 12,
                    len        = 0.6,
                    tickfont   = dict(color=TEXT),
                    title_font  = dict(color=TEXT),
                ),
            ),
            name      = "Random Portfolios",
            hovertemplate = "Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))

    # efficient frontier line
    vols = [p.volatility * 100        for p in result.frontier_portfolios]
    rets = [p.expected_return * 100   for p in result.frontier_portfolios]

    fig.add_trace(go.Scatter(
        x    = vols, y = rets,
        mode = "lines",
        name = "Efficient Frontier",
        line = dict(color=ELECTRIC, width=3),
        hovertemplate = "Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra>Frontier</extra>",
    ))

    # individual assets — shows where each stock sits on its own
    for i, ticker in enumerate(result.tickers):
        asset_ret = result.mean_returns[i] * 100
        asset_vol = np.sqrt(result.cov_matrix[i, i]) * 100
        fig.add_trace(go.Scatter(
            x    = [asset_vol], y = [asset_ret],
            mode = "markers+text",
            name = ticker,
            marker = dict(color=CORAL, size=10, symbol="diamond", opacity=0.9),
            text = [ticker],
            textposition = "top center",
            textfont     = dict(color=CORAL, size=10),
            hovertemplate = f"{ticker}<br>Vol: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<extra></extra>",
        ))

    # max Sharpe star
    ms = result.max_sharpe
    fig.add_trace(go.Scatter(
        x    = [ms.volatility * 100], y = [ms.expected_return * 100],
        mode = "markers+text",
        name = f"Max Sharpe ({ms.sharpe:.2f})",
        marker = dict(color=GOLD, size=18, symbol="star"),
        text = ["★ Max Sharpe"],
        textposition = "top right",
        textfont     = dict(color=GOLD, size=11, family="Syne Mono, monospace"),
        hovertemplate = f"Max Sharpe: {ms.sharpe:.3f}<br>Vol: {ms.volatility*100:.2f}%<br>Return: {ms.expected_return*100:.2f}%<extra></extra>",
    ))

    # min vol diamond
    mv = result.min_vol
    fig.add_trace(go.Scatter(
        x    = [mv.volatility * 100], y = [mv.expected_return * 100],
        mode = "markers+text",
        name = f"Min Vol",
        marker = dict(color=MINT, size=14, symbol="diamond"),
        text = ["◆ Min Vol"],
        textposition = "top right",
        textfont     = dict(color=MINT, size=11, family="Syne Mono, monospace"),
        hovertemplate = f"Min Vol<br>Vol: {mv.volatility*100:.2f}%<br>Return: {mv.expected_return*100:.2f}%<extra></extra>",
    ))

    # Capital Market Line: from risk-free rate through Max Sharpe, extended right
    rf  = result.risk_free_rate * 100
    x0, x1 = 0, ms.volatility * 100 * 2.0
    y1 = rf + (ms.expected_return * 100 - rf) / (ms.volatility * 100) * x1
    fig.add_trace(go.Scatter(
        x    = [x0, x1], y = [rf, y1],
        mode = "lines",
        name = "Capital Market Line",
        line = dict(color=GOLD, width=1.5, dash="dot"),
        hoverinfo = "skip",
    ))

    _lay(fig, title="Efficient Frontier — Markowitz Mean-Variance Optimization", height=580)
    fig.update_xaxes(title_text="Annualised Volatility (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Annualised Expected Return (%)", ticksuffix="%")
    return fig


# ─── Weight bar charts ────────────────────────────────────────────────────

def weights_chart(result: FrontierResult) -> go.Figure:
    # Side-by-side bar chart comparing Max Sharpe and Min Vol allocations
    tickers   = result.tickers
    ms_weights = [result.max_sharpe.weights.get(t, 0) * 100 for t in tickers]
    mv_weights = [result.min_vol.weights.get(t, 0)    * 100 for t in tickers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name   = "Max Sharpe",
        x      = tickers,
        y      = ms_weights,
        marker = dict(color=GOLD, opacity=0.85),
        text   = [f"{w:.1f}%" for w in ms_weights],
        textposition = "outside",
        textfont     = dict(color=TEXT),
    ))
    fig.add_trace(go.Bar(
        name   = "Min Vol",
        x      = tickers,
        y      = mv_weights,
        marker = dict(color=MINT, opacity=0.85),
        text   = [f"{w:.1f}%" for w in mv_weights],
        textposition = "outside",
        textfont     = dict(color=TEXT),
    ))

    _lay(fig, title="Optimal Portfolio Weights", barmode="group", height=420)
    fig.update_xaxes(title_text="Asset")
    fig.update_yaxes(title_text="Weight (%)", ticksuffix="%")
    return fig


# ─── Correlation heatmap ──────────────────────────────────────────────────

def correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    # Shows how assets move together
    # High positive correlation = assets move in sync (less diversification benefit)
    # Near-zero correlation = assets are independent (good for diversification)
    # Negative correlation = assets move opposite (best for hedging)
    corr = prices.pct_change().dropna().corr()

    fig = go.Figure(go.Heatmap(
        z           = corr.values,
        x           = corr.columns.tolist(),
        y           = corr.index.tolist(),
        colorscale  = "RdBu_r",
        zmid        = 0,
        zmin        = -1, zmax = 1,
        text        = np.round(corr.values, 2),
        texttemplate = "%{text}",
        textfont    = dict(size=11, color="white"),
        colorbar    = dict(
            title     = "ρ",
            thickness = 14,
            tickfont  = dict(color=TEXT),
            titlefont = dict(color=TEXT),
        ),
    ))

    _lay(fig, title="Return Correlation Matrix", height=460)
    return fig


# ─── Rolling Sharpe along the frontier ───────────────────────────────────

def sharpe_curve(result: FrontierResult) -> go.Figure:
    # x = portfolio volatility, y = Sharpe ratio
    # Shows where on the frontier the Sharpe is highest
    # The peak is exactly the Max Sharpe portfolio

    vols   = [p.volatility * 100 for p in result.frontier_portfolios]
    sharpes = [p.sharpe           for p in result.frontier_portfolios]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = vols, y = sharpes,
        mode = "lines",
        name = "Sharpe Ratio",
        line = dict(color=ELECTRIC, width=2.5),
        fill = "tozeroy",
        fillcolor = "rgba(79,142,247,0.07)",
    ))

    # mark the peak
    peak_idx = int(np.argmax(sharpes))
    fig.add_trace(go.Scatter(
        x    = [vols[peak_idx]], y = [sharpes[peak_idx]],
        mode = "markers",
        name = "Peak Sharpe",
        marker = dict(color=GOLD, size=14, symbol="star"),
        hovertemplate = f"Peak Sharpe: {sharpes[peak_idx]:.3f}<br>At vol: {vols[peak_idx]:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))

    _lay(fig, title="Sharpe Ratio Along the Efficient Frontier", height=380)
    fig.update_xaxes(title_text="Annualised Volatility (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Sharpe Ratio")
    return fig


# ─── Pie chart of weights ─────────────────────────────────────────────────

def weights_pie(result: FrontierResult, portfolio: str = "max_sharpe") -> go.Figure:
    p       = result.max_sharpe if portfolio == "max_sharpe" else result.min_vol
    label   = "Max Sharpe" if portfolio == "max_sharpe" else "Min Vol"
    tickers = result.tickers
    weights  = [p.weights.get(t, 0) for t in tickers]

    # filter out near-zero weights for cleaner display
    pairs   = [(t, w) for t, w in zip(tickers, weights) if w > 0.005]
    labels_ = [x[0] for x in pairs]
    vals_   = [x[1] for x in pairs]

    fig = go.Figure(go.Pie(
        labels    = labels_,
        values    = vals_,
        hole      = 0.45,
        marker    = dict(colors=[ELECTRIC, GOLD, MINT, CORAL, "#B388FF", "#80DEEA"]),
        textinfo  = "label+percent",
        textfont  = dict(color=TEXT, size=11),
    ))

    _lay(fig, title=f"{label} Portfolio — Weight Allocation", height=400,
         annotations=[dict(text=label, x=0.5, y=0.5, font_size=13, showarrow=False,
                           font_color=TEXT)])
    return fig


# ─── Return distribution per asset ───────────────────────────────────────

def return_distributions(prices: pd.DataFrame) -> go.Figure:
    # Overlapping histograms of each asset's daily returns
    # Wider = more volatile, fatter tails = more extreme days
    daily = prices.pct_change().dropna()
    colours = [ELECTRIC, GOLD, MINT, CORAL, "#B388FF", "#80DEEA", "#FFB347", "#90EE90"]

    fig = go.Figure()
    for i, col in enumerate(daily.columns):
        fig.add_trace(go.Histogram(
            x          = daily[col] * 100,
            name       = col,
            nbinsx     = 60,
            opacity    = 0.55,
            marker_color = colours[i % len(colours)],
        ))

    _lay(fig, title="Daily Return Distributions", barmode="overlay", height=400)
    fig.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Frequency")
    return fig