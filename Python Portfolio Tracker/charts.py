#charts.py

# Plotly charts for the Portfolio Tracker.

import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore

# deep plum / violet on near-black — distinct from all previous projects
VIOLET  = "#9B59B6"
LAVENDER= "#C39BD3"
GOLD    = "#F39C12"
MINT    = "#1ABC9C"
CORAL   = "#E74C3C"
MUTED   = "#6C5A7C"
BG      = "#0A080E"
CARD    = "#120F18"
GRID    = "#1C1626"
TEXT    = "#D5CCE8"

_base = dict(
    plot_bgcolor  = BG,
    paper_bgcolor = BG,
    font          = dict(family="Space Mono, monospace", color=TEXT, size=11),
    xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    margin        = dict(l=60, r=30, t=60, b=60),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

def _lay(fig, **kw):
    fig.update_layout(**{**_base, **kw})


def cumulative_return_chart(daily_returns: pd.Series, benchmark: pd.Series = None) -> go.Figure:
    cum = (1 + daily_returns).cumprod()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum.index, y=(cum - 1) * 100,
        mode="lines", name="Portfolio",
        line=dict(color=VIOLET, width=2.5),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.07)",
    ))

    if benchmark is not None:
        b_aligned = benchmark.reindex(daily_returns.index).ffill()
        b_cum     = (1 + b_aligned).cumprod()
        fig.add_trace(go.Scatter(
            x=b_cum.index, y=(b_cum - 1) * 100,
            mode="lines", name="SPY (benchmark)",
            line=dict(color=MUTED, width=1.5, dash="dot"),
        ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _lay(fig, title="Portfolio Cumulative Return")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Return (%)", ticksuffix="%")
    return fig


def pnl_bar_chart(holdings_df: pd.DataFrame) -> go.Figure:
    df     = holdings_df.sort_values("P&L ($)")
    colours = [MINT if v >= 0 else CORAL for v in df["P&L ($)"]]

    fig = go.Figure(go.Bar(
        x=df["Ticker"], y=df["P&L ($)"],
        marker_color=colours,
        text=[f"${v:,.0f}" for v in df["P&L ($)"]],
        textposition="outside",
        textfont=dict(color=TEXT),
    ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _lay(fig, title="P&L by Holding", height=400)
    fig.update_xaxes(title_text="Ticker")
    fig.update_yaxes(title_text="P&L ($)", tickprefix="$")
    return fig


def sector_pie(sector_df: pd.DataFrame) -> go.Figure:
    colours = [VIOLET, GOLD, MINT, CORAL, LAVENDER,
               "#3498DB", "#E67E22", "#1ABC9C", "#E74C3C", "#95A5A6"]

    fig = go.Figure(go.Pie(
        labels   = sector_df["Sector"],
        values   = sector_df["Value"],
        hole     = 0.45,
        marker   = dict(colors=colours[:len(sector_df)]),
        textinfo = "label+percent",
        textfont = dict(color=TEXT),
    ))

    _lay(fig, title="Sector Allocation", height=420,
         annotations=[dict(text="Sectors", x=0.5, y=0.5,
                           font_size=12, showarrow=False, font_color=TEXT)])
    return fig


def drawdown_chart(daily_returns: pd.Series) -> go.Figure:
    cum    = (1 + daily_returns).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        mode="lines", name="Drawdown",
        line=dict(color=CORAL, width=1.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
    ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _lay(fig, title="Portfolio Drawdown")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    return fig


def daily_returns_histogram(daily_returns: pd.Series) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=daily_returns * 100,
        nbinsx=60,
        marker=dict(color=VIOLET, opacity=0.7),
        name="Daily Returns",
    ))

    mean_r = daily_returns.mean() * 100
    fig.add_vline(x=mean_r, line=dict(color=GOLD, dash="dash", width=2),
                  annotation_text=f"Mean: {mean_r:.3f}%", annotation_font_color=GOLD)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))

    _lay(fig, title="Daily Return Distribution", height=380)
    fig.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Frequency")
    return fig


def top_movers_bar(holdings_df: pd.DataFrame, n: int = 10) -> go.Figure:
    df      = holdings_df.nlargest(n, "P&L (%)").append(
              holdings_df.nsmallest(n, "P&L (%)")).drop_duplicates()
    colours = [MINT if v >= 0 else CORAL for v in df["P&L (%)"]]

    fig = go.Figure(go.Bar(
        x=df["Ticker"], y=df["P&L (%)"],
        marker_color=colours,
        text=[f"{v:.1f}%" for v in df["P&L (%)"]],
        textposition="outside",
        textfont=dict(color=TEXT),
    ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _lay(fig, title="Best & Worst Performers (%)", height=400)
    fig.update_xaxes(title_text="Ticker")
    fig.update_yaxes(title_text="Return (%)", ticksuffix="%")
    return fig