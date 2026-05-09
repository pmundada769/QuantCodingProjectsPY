#charts.py

# Plotly charts for the TSMOM / Volatility Targeting dashboard.

import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from tsmom import TSMOMResult, AssetSignal

# deep crimson on near-black
CRIMSON  = "#C0392B"
EMBER    = "#E74C3C"
SAND     = "#F39C12"
TEAL     = "#1ABC9C"
MUTED    = "#7F8C8D"
BG       = "#0C0808"
GRID     = "#1E1010"
TEXT     = "#EAD5D5"

_base = dict(
    plot_bgcolor  = BG,
    paper_bgcolor = BG,
    font          = dict(family="Courier New, monospace", color=TEXT, size=11),
    xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    margin        = dict(l=60, r=30, t=60, b=60),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

def _lay(fig, **kw):
    fig.update_layout(**{**_base, **kw})


def cumulative_return_chart(result: TSMOMResult, benchmark_cum: pd.Series = None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.portfolio_cumret.index,
        y=(result.portfolio_cumret - 1) * 100,
        mode="lines", name="TSMOM Strategy",
        line=dict(color=CRIMSON, width=2.5),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.08)",
    ))

    if benchmark_cum is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_cum.index,
            y=(benchmark_cum - 1) * 100,
            mode="lines", name="Buy & Hold (equal weight)",
            line=dict(color=MUTED, width=1.5, dash="dot"),
        ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _lay(fig, title="TSMOM Strategy — Cumulative Return")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Cumulative Return (%)", ticksuffix="%")
    return fig


def signal_chart(asset: AssetSignal) -> go.Figure:
    fig = sp.make_subplots(rows=3, cols=1,
        subplot_titles=[
            f"{asset.ticker} Price",
            f"{asset.ticker} TSMOM Signal (+1 Long / -1 Short)",
            f"{asset.ticker} Realised Vol (annualised) + Scaled Position",
        ],
        vertical_spacing=0.10,
    )

    # price
    cum_price = (1 + asset.returns).cumprod()
    fig.add_trace(go.Scatter(x=asset.returns.index, y=cum_price,
        mode="lines", line=dict(color=TEAL, width=1.8), name="Price", showlegend=False),
        row=1, col=1)

    # signal: colour background green for long, red for short
    for i in range(1, len(asset.raw_signal)):
        sig = asset.raw_signal.iloc[i]
        if sig == 1:
            fig.add_vrect(
                x0=asset.raw_signal.index[i-1], x1=asset.raw_signal.index[i],
                fillcolor="rgba(26,188,156,0.12)", layer="below", line_width=0,
                row=1, col=1,
            )

    fig.add_trace(go.Scatter(x=asset.raw_signal.index, y=asset.raw_signal,
        mode="lines", line=dict(color=SAND, width=1.5), name="Signal", showlegend=False),
        row=2, col=1)

    # vol + position
    fig.add_trace(go.Scatter(x=asset.realised_vol.index, y=asset.realised_vol * 100,
        mode="lines", line=dict(color=CRIMSON, width=1.8), name="Realised Vol %", showlegend=False),
        row=3, col=1)
    fig.add_trace(go.Scatter(x=asset.scaled_position.index, y=asset.scaled_position,
        mode="lines", line=dict(color=SAND, width=1.5, dash="dot"), name="Scaled Position",
        showlegend=False), row=3, col=1)

    for r in range(1, 4):
        fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)

    _lay(fig, title=f"{asset.ticker} — Signal Decomposition", height=700)
    return fig


def vol_target_comparison(results: dict) -> go.Figure:
    # results is {vol_target: TSMOMResult}
    vol_targets = sorted(results.keys())

    sharpes   = [results[v].sharpe        for v in vol_targets]
    ann_rets  = [results[v].ann_return * 100 for v in vol_targets]
    max_dds   = [results[v].max_drawdown * 100 for v in vol_targets]

    fig = sp.make_subplots(rows=1, cols=3,
        subplot_titles=["Sharpe Ratio", "Ann. Return (%)", "Max Drawdown (%)"])

    for vals, col_idx, col in [(sharpes, 1, CRIMSON), (ann_rets, 2, SAND), (max_dds, 3, TEAL)]:
        fig.add_trace(go.Scatter(
            x=[v*100 for v in vol_targets], y=vals,
            mode="lines+markers",
            line=dict(color=col, width=2.5),
            marker=dict(size=8),
            showlegend=False,
        ), row=1, col=col_idx)
        for r_i in range(1, 4):
            fig.update_xaxes(gridcolor=GRID, ticksuffix="%", title_text="Target Vol %", row=1, col=r_i)
            fig.update_yaxes(gridcolor=GRID, row=1, col=r_i)

    _lay(fig, title="Performance vs Volatility Target", height=380)
    return fig


def drawdown_chart(result: TSMOMResult) -> go.Figure:
    cum  = result.portfolio_cumret
    peak = cum.cummax()
    dd   = (cum - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        mode="lines", line=dict(color=EMBER, width=1.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.12)",
        name="Drawdown",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _lay(fig, title="TSMOM Portfolio Drawdown")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    return fig


def position_heatmap(result: TSMOMResult) -> go.Figure:
    # monthly position for each asset — shows regime of long/short across time
    pos_data = {}
    for asset in result.asset_signals:
        monthly = asset.scaled_position.resample("ME").last()
        pos_data[asset.ticker] = monthly

    pos_df = pd.DataFrame(pos_data).dropna(how="all")

    fig = go.Figure(go.Heatmap(
        z          = pos_df.T.values,
        x          = pos_df.index,
        y          = pos_df.columns.tolist(),
        colorscale = "RdYlGn",
        zmid=0, zmin=-2, zmax=2,
        colorbar   = dict(
            title    = dict(text="Position", font=dict(color=TEXT)),
            tickfont = dict(color=TEXT),
            thickness = 12,
        ),
    ))
    _lay(fig, title="Monthly Position Heatmap (Green = Long, Red = Short)", height=max(350, len(result.tickers)*40))
    fig.update_xaxes(title_text="Date")
    return fig