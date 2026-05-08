"""
Payoff diagram and Greeks surface generators.
Returns Plotly figures for use in Streamlit.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from black_scholes import black_scholes

# ─────────────────────────── Colour palette ───────────────────────────────
CALL_COL  = "#00C4B4"   # teal
PUT_COL   = "#FF6B6B"   # coral-red
NET_COL   = "#F5C542"   # amber
BG        = "#0D1117"
PAPER_BG  = "#0D1117"
GRID_COL  = "#1E2A38"
TEXT_COL  = "#E8EEF4"
AXIS_COL  = "#4A5568"

_layout_defaults = dict(
    plot_bgcolor=BG,
    paper_bgcolor=PAPER_BG,
    font=dict(family="JetBrains Mono, monospace", color=TEXT_COL, size=12),
    xaxis=dict(gridcolor=GRID_COL, zerolinecolor=AXIS_COL),
    yaxis=dict(gridcolor=GRID_COL, zerolinecolor=AXIS_COL),
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COL),
)


def _apply_layout(fig, **extra):
    fig.update_layout(**{**_layout_defaults, **extra})


# ─────────────────────────── Payoff diagram ───────────────────────────────

def payoff_diagram(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    option_type: str, show_breakeven: bool = True
) -> go.Figure:
    """
    Plots P&L at expiry vs current option value across a range of spot prices.
    """
    spot_range = np.linspace(S * 0.5, S * 1.5, 400)
    premium = black_scholes(S, K, T, r, sigma, q, option_type).price

    # Expiry payoff
    if option_type == "call":
        expiry_pnl = np.maximum(spot_range - K, 0) - premium
    else:
        expiry_pnl = np.maximum(K - spot_range, 0) - premium

    # Current theoretical value vs cost
    current_pnl = np.array([
        black_scholes(s, K, T, r, sigma, q, option_type).price - premium
        for s in spot_range
    ])

    col = CALL_COL if option_type == "call" else PUT_COL
    fig = go.Figure()

    # Current value curve
    fig.add_trace(go.Scatter(
        x=spot_range, y=current_pnl,
        name="Current value (P&L)",
        line=dict(color=col, width=2.5, dash="dot"),
        fill="tozeroy",
        fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:],16)},0.08)",
    ))

    # Expiry payoff
    fig.add_trace(go.Scatter(
        x=spot_range, y=expiry_pnl,
        name="Payoff at expiry",
        line=dict(color=col, width=3),
    ))

    # Strike line
    fig.add_vline(x=K, line=dict(color=AXIS_COL, dash="dash", width=1.5),
                  annotation_text=f"K = {K:.2f}", annotation_font_color=TEXT_COL)

    # Current spot
    fig.add_vline(x=S, line=dict(color=NET_COL, dash="dash", width=1.5),
                  annotation_text=f"S = {S:.2f}", annotation_font_color=NET_COL)

    # Break-even
    if show_breakeven:
        be = K + premium if option_type == "call" else K - premium
        if spot_range[0] <= be <= spot_range[-1]:
            fig.add_vline(x=be, line=dict(color="#888", dash="dot", width=1),
                          annotation_text=f"BE = {be:.2f}", annotation_font_color="#aaa")

    # Zero P&L line
    fig.add_hline(y=0, line=dict(color=AXIS_COL, width=1))

    _apply_layout(fig, title=f"{option_type.capitalize()} Option — Payoff Diagram")
    fig.update_xaxes(title_text="Spot Price")
    fig.update_yaxes(title_text="P&L (per option)")
    return fig


# ─────────────────────────── Greeks vs Spot ───────────────────────────────

def greeks_vs_spot(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    option_type: str,
) -> go.Figure:
    """
    2×2 subplot: Delta, Gamma, Theta, Vega vs spot price.
    """
    spot_range = np.linspace(S * 0.5, S * 1.5, 300)
    deltas, gammas, thetas, vegas = [], [], [], []

    for s in spot_range:
        res = black_scholes(s, K, T, r, sigma, q, option_type)
        deltas.append(res.delta)
        gammas.append(res.gamma)
        thetas.append(res.theta)
        vegas.append(res.vega)

    col = CALL_COL if option_type == "call" else PUT_COL
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=["Δ Delta", "Γ Gamma", "Θ Theta (per day)", "ν Vega (per 1% vol)"],
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    data = [
        (deltas, 1, 1), (gammas, 1, 2),
        (thetas, 2, 1), (vegas,  2, 2),
    ]

    for vals, row, col_idx in data:
        fig.add_trace(go.Scatter(
            x=spot_range, y=vals,
            line=dict(color=col, width=2.5),
            showlegend=False,
        ), row=row, col=col_idx)

        fig.add_vline(x=S, line=dict(color=NET_COL, dash="dash", width=1),
                      row=row, col=col_idx)   # type: ignore

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=AXIS_COL, row=i, col=j)
            fig.update_yaxes(gridcolor=GRID_COL, zerolinecolor=AXIS_COL, row=i, col=j)

    _apply_layout(fig, title="Greeks vs Spot Price", height=550)
    return fig


# ─────────────────────────── Greeks vs Time ───────────────────────────────

def greeks_vs_time(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    option_type: str,
) -> go.Figure:
    """Delta and Theta decay as time-to-expiry decreases."""
    times = np.linspace(max(T, 1/365), 0.01, 300)[::-1]  # from far to near
    days  = times * 365

    deltas, thetas = [], []
    for t in times:
        res = black_scholes(S, K, t, r, sigma, q, option_type)
        deltas.append(res.delta)
        thetas.append(res.theta)

    col = CALL_COL if option_type == "call" else PUT_COL
    fig = sp.make_subplots(rows=1, cols=2,
                           subplot_titles=["Δ Delta decay", "Θ Theta decay (per day)"])

    for vals, col_idx in [(deltas, 1), (thetas, 2)]:
        fig.add_trace(go.Scatter(
            x=days, y=vals,
            line=dict(color=col, width=2.5),
            showlegend=False,
        ), row=1, col=col_idx)

    for j in range(1, 3):
        fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=AXIS_COL, title_text="Days to expiry", row=1, col=j)
        fig.update_yaxes(gridcolor=GRID_COL, zerolinecolor=AXIS_COL, row=1, col=j)

    _apply_layout(fig, title="Greeks vs Time to Expiry", height=380)
    return fig


# ─────────────────────────── Vol surface (smile) ──────────────────────────

def vol_smile(
    S: float, K: float, T: float, r: float, sigma: float, q: float,
    option_type: str,
) -> go.Figure:
    """
    Simulated vol smile: flat vol shifted by a small skew + curvature,
    illustrating that real markets deviate from constant-vol BS.
    """
    strikes  = np.linspace(S * 0.70, S * 1.30, 60)
    moneyness = np.log(strikes / S)

    # Synthetic smile: vol = sigma + skew*m + curvature*m^2
    skew      = -0.10
    curvature =  0.30
    smile_vol = sigma + skew * moneyness + curvature * moneyness**2

    # Flat BS vol for reference
    flat_vol = np.full_like(strikes, sigma)

    col = CALL_COL if option_type == "call" else PUT_COL
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strikes, y=smile_vol * 100,
        name="Market vol smile (synthetic)",
        line=dict(color=col, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=strikes, y=flat_vol * 100,
        name="Flat (BS) vol",
        line=dict(color="#888", width=2, dash="dot"),
    ))
    fig.add_vline(x=S, line=dict(color=NET_COL, dash="dash", width=1.5),
                  annotation_text="ATM", annotation_font_color=NET_COL)

    _apply_layout(fig, title="Implied Volatility Smile (illustrative)")
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="Implied Vol (%)")
    return fig


# ─────────────────────────── Strategy builder ─────────────────────────────

STRATEGIES = {
    "Bull Call Spread": [
        {"type": "call", "sign": +1, "offset": -0.05},   # long lower call
        {"type": "call", "sign": -1, "offset": +0.05},   # short upper call
    ],
    "Bear Put Spread": [
        {"type": "put",  "sign": +1, "offset": +0.05},
        {"type": "put",  "sign": -1, "offset": -0.05},
    ],
    "Long Straddle": [
        {"type": "call", "sign": +1, "offset": 0},
        {"type": "put",  "sign": +1, "offset": 0},
    ],
    "Long Strangle": [
        {"type": "call", "sign": +1, "offset": +0.05},
        {"type": "put",  "sign": +1, "offset": -0.05},
    ],
    "Covered Call": [
        {"type": "stock","sign": +1, "offset": 0},
        {"type": "call", "sign": -1, "offset": +0.05},
    ],
    "Protective Put": [
        {"type": "stock","sign": +1, "offset": 0},
        {"type": "put",  "sign": +1, "offset": -0.05},
    ],
    "Long Butterfly": [
        {"type": "call", "sign": +1, "offset": -0.05},
        {"type": "call", "sign": -2, "offset": 0},
        {"type": "call", "sign": +1, "offset": +0.05},
    ],
    "Iron Condor": [
        {"type": "put",  "sign": +1, "offset": -0.10},
        {"type": "put",  "sign": -1, "offset": -0.05},
        {"type": "call", "sign": -1, "offset": +0.05},
        {"type": "call", "sign": +1, "offset": +0.10},
    ],
}


def strategy_payoff(
    S: float, T: float, r: float, sigma: float, q: float,
    strategy_name: str,
) -> go.Figure:
    """Payoff diagram for common multi-leg strategies."""
    legs = STRATEGIES[strategy_name]
    spot_range = np.linspace(S * 0.60, S * 1.40, 400)
    net_pnl = np.zeros(len(spot_range))

    leg_traces = []
    for leg in legs:
        otype  = leg["type"]
        sign   = leg["sign"]
        K_leg  = S * (1 + leg["offset"])

        if otype == "stock":
            cost_leg = S
            expiry_pnl = (spot_range - S) * sign
        else:
            res = black_scholes(S, K_leg, T, r, sigma, q, otype)
            cost_leg = res.price
            if otype == "call":
                expiry_pnl = np.maximum(spot_range - K_leg, 0) * sign - cost_leg * sign
            else:
                expiry_pnl = np.maximum(K_leg - spot_range, 0) * sign - cost_leg * sign

        net_pnl += expiry_pnl

        lbl = f"{'Long' if sign > 0 else 'Short'} {otype.capitalize()} K={K_leg:.1f}"
        col = CALL_COL if sign > 0 else PUT_COL
        leg_traces.append((spot_range, expiry_pnl, lbl, col))

    fig = go.Figure()
    for x, y, name, col in leg_traces:
        fig.add_trace(go.Scatter(x=x, y=y, name=name,
                                 line=dict(color=col, width=1.5, dash="dot"),
                                 opacity=0.6))

    fig.add_trace(go.Scatter(
        x=spot_range, y=net_pnl,
        name="Net P&L",
        line=dict(color=NET_COL, width=3.5),
        fill="tozeroy",
        fillcolor="rgba(245,197,66,0.08)",
    ))

    fig.add_vline(x=S, line=dict(color="#aaa", dash="dash", width=1),
                  annotation_text=f"S={S:.0f}", annotation_font_color="#aaa")
    fig.add_hline(y=0, line=dict(color=AXIS_COL, width=1))

    _apply_layout(fig, title=f"{strategy_name} — Payoff at Expiry")
    fig.update_xaxes(title_text="Spot at Expiry")
    fig.update_yaxes(title_text="P&L")
    return fig