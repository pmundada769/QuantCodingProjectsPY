#charts.py

# All Plotly charts for the CAPM / FF3 Factor Regression dashboard.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from regression import RegressionResult

# forest green on near-black — different from all previous projects
FOREST  = "#2ECC71"
LIME    = "#A8FF3E"
GOLD    = "#F1C40F"
CORAL   = "#E74C3C"
MUTED   = "#5D7A6A"
BG      = "#080E0A"
CARD    = "#0D1610"
GRID    = "#142018"
TEXT    = "#C8DDD0"
BLUE    = "#3498DB"

_base = dict(
    plot_bgcolor  = BG,
    paper_bgcolor = BG,
    font          = dict(family="Fira Code, monospace", color=TEXT, size=11),
    xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    margin        = dict(l=60, r=30, t=60, b=60),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

def _lay(fig, **kw):
    fig.update_layout(**{**_base, **kw})


def scatter_regression(
    excess_returns: pd.Series,
    mkt_rf:         pd.Series,
    result:         RegressionResult,
) -> go.Figure:
    # classic CAPM scatter: x = market excess return, y = stock excess return
    # the regression line slope is beta, the intercept is alpha

    x   = mkt_rf.values
    y   = excess_returns.values
    fit = result.alpha + result.beta_market * x

    fig = go.Figure()

    # scatter points
    fig.add_trace(go.Scatter(
        x=x*100, y=y*100,
        mode="markers",
        marker=dict(color=FOREST, size=6, opacity=0.6),
        name="Monthly Returns",
        hovertemplate="Mkt-RF: %{x:.2f}%<br>Stock excess: %{y:.2f}%<extra></extra>",
    ))

    # regression line
    x_sorted = np.sort(x)
    y_fitted  = result.alpha + result.beta_market * x_sorted
    fig.add_trace(go.Scatter(
        x=x_sorted*100, y=y_fitted*100,
        mode="lines",
        line=dict(color=GOLD, width=2.5),
        name=f"Fit  β={result.beta_market:.3f}  α={result.alpha*100:.3f}%/mo",
    ))

    # zero lines
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))

    _lay(fig, title=f"{result.ticker} — CAPM Scatter  (R²={result.r_squared:.3f})")
    fig.update_xaxes(title_text="Market Excess Return (%)", ticksuffix="%")
    fig.update_yaxes(title_text=f"{result.ticker} Excess Return (%)", ticksuffix="%")
    return fig


def rolling_beta_chart(rolling_df: pd.DataFrame, ticker: str) -> go.Figure:
    # rolling 24-month beta and alpha — shows if market sensitivity is stable
    fig = sp.make_subplots(
        rows=2, cols=1,
        subplot_titles=["Rolling Beta (24-month window)", "Rolling Annualised Alpha (24-month window)"],
        vertical_spacing=0.15,
    )

    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["Beta"],
        mode="lines", line=dict(color=FOREST, width=2.5),
        name="Beta", showlegend=False,
    ), row=1, col=1)

    # reference line at beta=1 (moves exactly with market)
    fig.add_hline(y=1, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text="β=1", annotation_font_color=MUTED, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["Alpha"]*100,
        mode="lines", line=dict(color=GOLD, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(241,196,15,0.08)",
        name="Alpha", showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=2, col=1)

    for r in [1, 2]:
        fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)

    fig.update_yaxes(title_text="Beta",     row=1, col=1)
    fig.update_yaxes(title_text="Alpha (%)", ticksuffix="%", row=2, col=1)

    _lay(fig, title=f"{ticker} — Rolling CAPM Estimates", height=520)
    return fig


def factor_betas_bar(capm: RegressionResult, ff3: RegressionResult) -> go.Figure:
    # side-by-side bar chart comparing CAPM beta vs FF3 betas
    # if beta_market changes a lot between models, SMB/HML were correlated with market

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name   = "CAPM",
        x      = ["β Market"],
        y      = [capm.beta_market],
        marker = dict(color=FOREST, opacity=0.85),
        text   = [f"{capm.beta_market:.3f}"],
        textposition = "outside",
        textfont     = dict(color=TEXT),
    ))

    ff3_labels = ["β Market", "β SMB", "β HML"]
    ff3_values = [ff3.beta_market, ff3.beta_smb or 0, ff3.beta_hml or 0]
    fig.add_trace(go.Bar(
        name   = "FF3",
        x      = ff3_labels,
        y      = ff3_values,
        marker = dict(color=GOLD, opacity=0.85),
        text   = [f"{v:.3f}" for v in ff3_values],
        textposition = "outside",
        textfont     = dict(color=TEXT),
    ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))

    _lay(fig, title=f"{capm.ticker} — Factor Betas: CAPM vs FF3", barmode="group", height=420)
    fig.update_xaxes(title_text="Factor")
    fig.update_yaxes(title_text="Beta Coefficient")
    return fig


def decomposition_waterfall(decomp_df: pd.DataFrame, ticker: str) -> go.Figure:
    # waterfall chart showing how each factor contributes to average monthly return
    # makes the attribution story visually obvious

    components  = decomp_df["Component"].tolist()
    values      = decomp_df["Contribution %"].tolist()
    total       = sum(values)
    components.append("Total Expected Return")
    values.append(total)

    colours = []
    for v in values[:-1]:
        colours.append(FOREST if v >= 0 else CORAL)
    colours.append(GOLD)   # total bar always gold

    fig = go.Figure(go.Bar(
        x    = components,
        y    = values,
        marker_color = colours,
        text = [f"{v:.3f}%" for v in values],
        textposition = "outside",
        textfont     = dict(color=TEXT),
    ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))

    _lay(fig, title=f"{ticker} — Return Decomposition (Monthly %)", height=420)
    fig.update_xaxes(title_text="Component")
    fig.update_yaxes(title_text="Contribution (% per month)", ticksuffix="%")
    return fig


def multi_ticker_summary(results: list) -> go.Figure:
    # scatter plot: x = beta, y = alpha, size = R²
    # at a glance shows which stocks have real alpha vs market exposure
    betas  = [r.beta_market for r in results]
    alphas = [r.alpha * 100 * 12 for r in results]  # annualised alpha %
    r2s    = [r.r_squared for r in results]
    labels = [r.ticker for r in results]
    models = [r.model for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = betas,
        y    = alphas,
        mode = "markers+text",
        text = labels,
        textposition = "top center",
        textfont     = dict(color=TEXT, size=10),
        marker = dict(
            size   = [max(8, r*30) for r in r2s],
            color  = r2s,
            colorscale = "Greens",
            showscale  = True,
            colorbar   = dict(
                title    = dict(text="R²", font=dict(color=TEXT)),
                tickfont = dict(color=TEXT),
                thickness = 12,
            ),
            opacity = 0.85,
        ),
        hovertemplate="<b>%{text}</b><br>Beta: %{x:.3f}<br>Ann. Alpha: %{y:.2f}%<extra></extra>",
    ))

    fig.add_vline(x=1, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text="β=1", annotation_font_color=MUTED)
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text="α=0", annotation_font_color=MUTED)

    _lay(fig, title="Multi-Ticker Factor Summary  (bubble size = R²)", height=480)
    fig.update_xaxes(title_text="Market Beta")
    fig.update_yaxes(title_text="Annualised Alpha (%)", ticksuffix="%")
    return fig