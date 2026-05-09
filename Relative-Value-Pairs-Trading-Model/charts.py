#charts.py

# Plotly charts for the Pairs Trading dashboard.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from pairs import PairResult

# slate-indigo on near-black
INDIGO  = "#6C5CE7"
VIOLET  = "#A29BFE"
MINT    = "#00B894"
CORAL   = "#E17055"
MUTED   = "#636E72"
BG      = "#080810"
GRID    = "#10101E"
TEXT    = "#D0D0F0"

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


def spread_chart(pair: PairResult) -> go.Figure:
    fig = sp.make_subplots(rows=3, cols=1,
        subplot_titles=[
            f"Spread: {pair.ticker_a} – {pair.hedge_ratio:.3f} × {pair.ticker_b}",
            "Z-Score (entry ±2σ, exit ±0.5σ)",
            "Strategy Signal (+1 Long / -1 Short)",
        ],
        vertical_spacing=0.10,
    )

    # spread
    fig.add_trace(go.Scatter(x=pair.spread.index, y=pair.spread,
        mode="lines", line=dict(color=INDIGO, width=1.8), showlegend=False), row=1, col=1)
    fig.add_hline(y=pair.spread.mean(), line=dict(color=MUTED, dash="dash", width=1), row=1, col=1)

    # z-score with threshold lines
    fig.add_trace(go.Scatter(x=pair.zscore.index, y=pair.zscore,
        mode="lines", line=dict(color=VIOLET, width=1.8), showlegend=False), row=2, col=1)
    for level, col in [(2, CORAL), (-2, MINT), (0.5, MUTED), (-0.5, MUTED)]:
        fig.add_hline(y=level, line=dict(color=col, dash="dot", width=1), row=2, col=1)

    # signal
    fig.add_trace(go.Scatter(x=pair.signals.index, y=pair.signals,
        mode="lines", line=dict(color=MINT, width=1.5), showlegend=False,
        fill="tozeroy", fillcolor="rgba(0,184,148,0.08)"), row=3, col=1)

    for r in range(1, 4):
        fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=r, col=1)

    _lay(fig, title=f"{pair.ticker_a} / {pair.ticker_b} Pairs Trade", height=650)
    return fig


def cumulative_pnl(pair: PairResult, benchmark_a: pd.Series = None) -> go.Figure:
    cum = (1 + pair.strategy_returns).cumprod()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum.index, y=(cum-1)*100,
        mode="lines", name="Pairs Strategy",
        line=dict(color=INDIGO, width=2.5),
        fill="tozeroy", fillcolor="rgba(108,92,231,0.08)",
    ))

    if benchmark_a is not None:
        b_cum = (1 + benchmark_a.pct_change().dropna()).cumprod()
        fig.add_trace(go.Scatter(
            x=b_cum.index, y=(b_cum-1)*100,
            mode="lines", name=f"{pair.ticker_a} buy & hold",
            line=dict(color=MUTED, width=1.5, dash="dot"),
        ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    _lay(fig, title="Pairs Strategy — Cumulative P&L")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Return (%)", ticksuffix="%")
    return fig


def price_comparison(prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> go.Figure:
    # normalised price chart of both assets — shows how they co-move
    p_a = prices[ticker_a] / prices[ticker_a].iloc[0] * 100
    p_b = prices[ticker_b] / prices[ticker_b].iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_a.index, y=p_a, mode="lines",
        name=ticker_a, line=dict(color=INDIGO, width=2)))
    fig.add_trace(go.Scatter(x=p_b.index, y=p_b, mode="lines",
        name=ticker_b, line=dict(color=MINT, width=2)))

    _lay(fig, title=f"{ticker_a} vs {ticker_b} — Normalised Prices (base=100)")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Normalised Price")
    return fig


def scan_summary_chart(scan_result) -> go.Figure:
    # scatter: x = p-value, y = Sharpe, colour = half-life
    # shows which pairs are both statistically cointegrated AND profitable
    pairs = scan_result.all_pairs

    pvals     = [p.eg_pvalue  for p in pairs]
    sharpes   = [p.sharpe     for p in pairs]
    hls       = [min(p.half_life, 200) if not np.isnan(p.half_life) else 100 for p in pairs]
    labels    = [f"{p.ticker_a}/{p.ticker_b}" for p in pairs]
    coint_col = [MINT if p.cointegrated else MUTED for p in pairs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pvals, y=sharpes,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(color=TEXT, size=9),
        marker=dict(
            size=10,
            color=hls,
            colorscale="Plasma_r",
            showscale=True,
            colorbar=dict(
                title=dict(text="Half-Life (d)", font=dict(color=TEXT)),
                tickfont=dict(color=TEXT),
                thickness=12,
            ),
            opacity=0.85,
        ),
        hovertemplate="<b>%{text}</b><br>p-val: %{x:.3f}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))

    fig.add_vline(x=0.05, line=dict(color=CORAL, dash="dash", width=1.5),
                  annotation_text="p=0.05 threshold", annotation_font_color=CORAL)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))

    _lay(fig, title="Pairs Universe Scan — Cointegration p-value vs Sharpe", height=500)
    fig.update_xaxes(title_text="Engle-Granger p-value (lower = more cointegrated)")
    fig.update_yaxes(title_text="Strategy Sharpe Ratio")
    return fig