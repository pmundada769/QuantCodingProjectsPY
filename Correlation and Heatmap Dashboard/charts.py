#charts.py

# Plotly charts for the Correlation & Heatmap Dashboard.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

# warm orange-red on near-black — last distinct palette
ORANGE  = "#FF6B35"
AMBER   = "#FFB347"
CREAM   = "#FFF0D6"
TEAL    = "#2EC4B6"
CORAL   = "#E74C3C"
MUTED   = "#7A5A3A"
BG      = "#0E0A06"
CARD    = "#18110A"
GRID    = "#241A0E"
TEXT    = "#EAD8C0"

_base = dict(
    plot_bgcolor  = BG,
    paper_bgcolor = BG,
    font          = dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
    xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    margin        = dict(l=60, r=30, t=60, b=60),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

def _lay(fig, **kw):
    fig.update_layout(**{**_base, **kw})


def static_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    # full static correlation heatmap with values printed in each cell
    fig = go.Figure(go.Heatmap(
        z          = corr_matrix.values,
        x          = corr_matrix.columns.tolist(),
        y          = corr_matrix.index.tolist(),
        colorscale = "RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text       = np.round(corr_matrix.values, 2),
        texttemplate = "%{text}",
        textfont   = dict(size=10, color="white"),
        colorbar   = dict(
            title    = dict(text="ρ", font=dict(color=TEXT)),
            tickfont = dict(color=TEXT),
            thickness = 14,
        ),
    ))
    _lay(fig, title=title, height=480)
    return fig


def rolling_correlation_lines(rolling_corr: dict, highlight_pairs: list = None) -> go.Figure:
    # line chart of rolling correlations for all (or selected) pairs over time
    colours = [ORANGE, TEAL, AMBER, CORAL, "#9B59B6", "#2ECC71", "#3498DB", "#E67E22"]

    fig = go.Figure()
    pairs_to_show = highlight_pairs or list(rolling_corr.keys())[:8]

    for i, pair in enumerate(pairs_to_show):
        if pair not in rolling_corr:
            continue
        series = rolling_corr[pair]
        a, b   = pair
        fig.add_trace(go.Scatter(
            x    = series.index,
            y    = series.values,
            mode = "lines",
            name = f"{a}–{b}",
            line = dict(color=colours[i % len(colours)], width=1.8),
        ))

    fig.add_hline(y=0,    line=dict(color=MUTED, width=1))
    fig.add_hline(y=0.7,  line=dict(color=CORAL, dash="dot", width=1),
                  annotation_text="High correlation", annotation_font_color=CORAL)
    fig.add_hline(y=-0.3, line=dict(color=TEAL,  dash="dot", width=1),
                  annotation_text="Negative correlation", annotation_font_color=TEAL)

    _lay(fig, title="Rolling Pairwise Correlations")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Correlation (ρ)", range=[-1, 1])
    return fig


def average_correlation_chart(avg_corr: pd.Series, regime_shifts: list = None) -> go.Figure:
    # average pairwise correlation over time — the "fear gauge" for diversification
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x    = avg_corr.index,
        y    = avg_corr.values,
        mode = "lines",
        name = "Avg Pairwise Corr",
        line = dict(color=ORANGE, width=2.5),
        fill = "tozeroy",
        fillcolor = "rgba(255,107,53,0.08)",
    ))

    # shade high-correlation "crisis" zones
    high_corr = avg_corr[avg_corr > 0.6]
    if len(high_corr) > 0:
        for start_date, grp in high_corr.groupby((high_corr.index.to_series().diff() > pd.Timedelta("5d")).cumsum()):
            if len(grp) > 5:
                fig.add_vrect(
                    x0=grp.index[0], x1=grp.index[-1],
                    fillcolor="rgba(231,76,60,0.10)",
                    layer="below", line_width=0,
                    annotation_text="Crisis zone",
                    annotation_font_color=CORAL,
                    annotation_position="top left",
                )

    fig.add_hline(y=0.6, line=dict(color=CORAL, dash="dot", width=1.5),
                  annotation_text="Crisis threshold (0.6)", annotation_font_color=CORAL)
    fig.add_hline(y=0,   line=dict(color=MUTED, width=1))

    _lay(fig, title="Average Pairwise Correlation — Regime Indicator")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Average Correlation", range=[-0.2, 1])
    return fig


def correlation_heatmap_animated(returns: pd.DataFrame, n_periods: int = 6) -> go.Figure:
    # snapshots of correlation matrix at evenly spaced time points
    # shows how correlations evolve over the history

    n     = len(returns)
    step  = n // n_periods
    dates = [returns.index[min(i * step + step - 1, n-1)] for i in range(n_periods)]

    frames = []
    sliders_steps = []

    first_corr = returns.iloc[:step].corr()
    fig = go.Figure(go.Heatmap(
        z          = first_corr.values,
        x          = first_corr.columns.tolist(),
        y          = first_corr.index.tolist(),
        colorscale = "RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        text       = np.round(first_corr.values, 2),
        texttemplate = "%{text}",
        textfont   = dict(size=10, color="white"),
        colorbar   = dict(
            title    = dict(text="ρ", font=dict(color=TEXT)),
            tickfont = dict(color=TEXT),
            thickness= 14,
        ),
    ))

    for i, date in enumerate(dates):
        end_idx = returns.index.get_loc(date) + 1
        subset  = returns.iloc[max(0, end_idx-63):end_idx]
        corr    = subset.corr()
        frames.append(go.Frame(
            data  = [go.Heatmap(z=corr.values, text=np.round(corr.values, 2))],
            name  = str(date.date()),
        ))
        sliders_steps.append(dict(
            args=[[ str(date.date()) ], {"frame": {"duration": 500}, "mode": "immediate"}],
            label=str(date.date())[:7],
            method="animate",
        ))

    fig.frames = frames
    fig.update_layout(
        sliders=[dict(steps=sliders_steps, currentvalue=dict(prefix="Period: "))],
        updatemenus=[dict(type="buttons", showactive=False, buttons=[
            dict(label="▶ Play",  method="animate",
                 args=[None, {"frame": {"duration": 800}, "fromcurrent": True}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None],    {"frame": {"duration": 0},   "mode": "immediate"}]),
        ])],
    )
    _lay(fig, title="Correlation Matrix Over Time (animated)", height=520)
    return fig


def dispersion_chart(disp: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=disp.index, y=disp * 100,
        mode="lines", name="Dispersion",
        line=dict(color=TEAL, width=2),
        fill="tozeroy", fillcolor="rgba(46,196,182,0.07)",
    ))

    _lay(fig, title="Cross-Sectional Dispersion (Rolling 20-day Avg Std Dev)")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Avg Daily Std Dev (%)", ticksuffix="%")
    return fig