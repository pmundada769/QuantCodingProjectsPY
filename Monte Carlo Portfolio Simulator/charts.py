#charts.py

'''
All Plotly visualisations for the Monte Carlo simulator.
Returns go.Figure objects for use in Streamlit.
'''
import numpy as np
import plotly.graph_objects as go # type: ignore
import plotly.subplots as sp # type: ignore
from simulator import SimulationResult

'''colour palette - amber/rust on near-black, different from Options Pricer'''
AMBER    = "#F5A623"
RUST     = "#E8472A"
TEAL     = "#2ABFB0"
MUTED    = "#8A9BB0"
BG       = "#0B0E14"
CARD_BG  = "#111520"
GRID     = "#1A2030"
TEXT     = "#DDE4EE"
GREEN    = "#4CAF7D"
RED      = "#E8472A"

_base_layout = dict(
    plot_bgcolor  = BG,
    paper_bgcolor = BG,
    font          = dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
    xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    margin        = dict(l=50, r=20, t=50, b=50),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

def _layout(fig, **extra):
    fig.update_layout(**{**_base_layout, **extra})


'''path fan chart - the signature Monte Carlo visual'''
def paths_chart(result: SimulationResult, n_display: int = 200) -> go.Figure:
    '''
    Plots a sample of simulated portfolio paths as a fan.
    Shows median, 5th/95th percentile bands, and individual paths.
    n_display: how many individual paths to draw (all 10k would be too slow)
    '''
    n_days     = result.n_days
    days       = np.arange(n_days + 1)
    paths      = result.paths

    '''percentile bands across all simulations at each time step'''
    p05 = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()

    '''individual paths - thin, semi-transparent, subsample for performance'''
    idx = np.random.choice(len(paths), size=min(n_display, len(paths)), replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=days, y=paths[i],
            mode="lines",
            line=dict(color=f"rgba(245,166,35,0.04)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    '''shaded band: 5th to 95th percentile'''
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p95, p05[::-1]]),
        fill="toself",
        fillcolor="rgba(245,166,35,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th percentile",
        hoverinfo="skip",
    ))

    '''shaded band: 25th to 75th percentile (interquartile range)'''
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(245,166,35,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th–75th percentile",
        hoverinfo="skip",
    ))

    '''median path'''
    fig.add_trace(go.Scatter(
        x=days, y=p50,
        name="Median",
        line=dict(color=AMBER, width=2.5),
    ))

    '''starting value horizontal line'''
    fig.add_hline(
        y=result.initial_value,
        line=dict(color=MUTED, dash="dash", width=1),
        annotation_text="Start",
        annotation_font_color=MUTED,
    )

    _layout(fig, title=f"Monte Carlo Simulation — {result.n_simulations:,} Paths")
    fig.update_xaxes(title_text="Trading Days")
    fig.update_yaxes(title_text="Portfolio Value ($)", tickprefix="$", tickformat=",.0f")
    return fig


'''histogram of terminal values with VaR / CVaR markers'''
def terminal_distribution(result: SimulationResult) -> go.Figure:
    '''
    Distribution of all final portfolio values.
    Shades the tail to visually represent VaR and CVaR regions.
    '''
    final = result.final_values
    p5    = np.percentile(final, 5)
    p1    = np.percentile(final, 1)

    fig = go.Figure()

    '''full histogram'''
    fig.add_trace(go.Histogram(
        x=final,
        nbinsx=80,
        name="Final Value",
        marker=dict(color=AMBER, opacity=0.6, line=dict(color=BG, width=0.3)),
    ))

    '''shade the 5% tail (VaR 95 region)'''
    tail_vals = final[final <= p5]
    if len(tail_vals) > 0:
        fig.add_trace(go.Histogram(
            x=tail_vals,
            nbinsx=20,
            name="5% Tail (VaR 95)",
            marker=dict(color=RUST, opacity=0.85),
        ))

    '''VaR 95 vertical line'''
    fig.add_vline(
        x=p5,
        line=dict(color=RUST, dash="dash", width=2),
        annotation_text=f"VaR 95%: ${result.var_95:,.0f} loss",
        annotation_font_color=RUST,
        annotation_position="top left",
    )

    '''CVaR 95 vertical line'''
    cvar_level = result.initial_value - result.cvar_95
    fig.add_vline(
        x=cvar_level,
        line=dict(color="#FF8C00", dash="dot", width=1.5),
        annotation_text=f"CVaR 95%: ${result.cvar_95:,.0f} loss",
        annotation_font_color="#FF8C00",
        annotation_position="top left",
    )

    '''starting value'''
    fig.add_vline(
        x=result.initial_value,
        line=dict(color=TEAL, dash="dash", width=1.5),
        annotation_text="Break-even",
        annotation_font_color=TEAL,
    )

    _layout(fig, title="Distribution of Terminal Portfolio Values", barmode="overlay")
    fig.update_xaxes(title_text="Final Portfolio Value ($)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title_text="Number of Simulations")
    return fig


'''drawdown distribution'''
def drawdown_chart(result: SimulationResult) -> go.Figure:
    '''
    For each path, computes the maximum drawdown (worst peak-to-trough loss).
    Plots the distribution of max drawdowns across all simulations.
    '''
    paths = result.paths

    '''compute max drawdown for each path'''
    max_drawdowns = []
    for path in paths:
        peak     = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        max_drawdowns.append(drawdown.min())

    max_drawdowns = np.array(max_drawdowns)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=max_drawdowns * 100,
        nbinsx=60,
        name="Max Drawdown",
        marker=dict(color=RUST, opacity=0.7),
    ))

    '''median drawdown line'''
    median_dd = np.median(max_drawdowns) * 100
    fig.add_vline(
        x=median_dd,
        line=dict(color=AMBER, dash="dash", width=2),
        annotation_text=f"Median: {median_dd:.1f}%",
        annotation_font_color=AMBER,
    )

    _layout(fig, title="Distribution of Maximum Drawdowns")
    fig.update_xaxes(title_text="Max Drawdown (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Number of Simulations")
    return fig


'''risk metric comparison: VaR vs CVaR at different confidence levels'''
def var_cvar_bar(result: SimulationResult) -> go.Figure:
    '''
    Side-by-side bar chart comparing VaR and CVaR at 95% and 99%.
    Makes the gap between VaR and CVaR visually obvious -
    CVaR is always worse because it averages the tail, not just its boundary.
    '''
    labels  = ["VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"]
    values  = [result.var_95, result.cvar_95, result.var_99, result.cvar_99]
    colours = [AMBER, RUST, "#FFD700", "#C0392B"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colours, opacity=0.85),
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT),
    ))

    _layout(fig, title="VaR vs CVaR Comparison")
    fig.update_xaxes(title_text="Risk Measure")
    fig.update_yaxes(title_text="Loss ($)", tickprefix="$", tickformat=",.0f")
    return fig


'''cumulative probability chart (empirical CDF of terminal values)'''
def cdf_chart(result: SimulationResult) -> go.Figure:
    '''
    Empirical CDF of terminal portfolio values.
    Reads as: "probability that the final value is below X"
    The 5% and 1% quantiles are where VaR is read off.
    '''
    sorted_vals = np.sort(result.final_values)
    cdf         = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_vals, y=cdf * 100,
        name="CDF",
        line=dict(color=AMBER, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(245,166,35,0.07)",
    ))

    '''mark VaR 95 and 99 on the CDF'''
    for pct, color, label in [(5, RUST, "5%"), (1, "#C0392B", "1%")]:
        val = np.percentile(result.final_values, pct)
        fig.add_vline(x=val, line=dict(color=color, dash="dash", width=1.5),
                      annotation_text=f"p{pct}=${val:,.0f}", annotation_font_color=color)

    fig.add_vline(x=result.initial_value, line=dict(color=TEAL, dash="dash", width=1.5),
                  annotation_text="Start", annotation_font_color=TEAL)

    _layout(fig, title="Cumulative Distribution of Terminal Values")
    fig.update_xaxes(title_text="Final Portfolio Value ($)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title_text="Probability (%)", ticksuffix="%")
    return fig


'''vol sensitivity: how VaR changes as volatility increases'''
def vol_sensitivity_chart(
    initial_value: float,
    annual_return: float,
    n_simulations: int,
    n_days: int,
    ruin_threshold: float,
) -> go.Figure:
    '''
    Runs multiple simulations across a range of volatilities
    and plots how VaR 95, CVaR 95, and Prob of Ruin respond.
    Shows the non-linear relationship between vol and tail risk.
    '''
    from simulator import run_simulation

    vol_range = np.linspace(0.05, 0.60, 20)
    var95s, cvar95s, ruins = [], [], []

    for v in vol_range:
        r = run_simulation(
            initial_value  = initial_value,
            annual_return  = annual_return,
            annual_vol     = v,
            n_simulations  = 2000,         # fewer paths for speed
            n_days         = n_days,
            ruin_threshold = ruin_threshold,
            seed           = 42,
        )
        var95s.append(r.var_95)
        cvar95s.append(r.cvar_95)
        ruins.append(r.prob_ruin * 100)

    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=["VaR 95% and CVaR 95% vs Volatility", "Prob of Ruin vs Volatility"],
    )

    fig.add_trace(go.Scatter(x=vol_range*100, y=var95s,  name="VaR 95%",
                             line=dict(color=AMBER, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=vol_range*100, y=cvar95s, name="CVaR 95%",
                             line=dict(color=RUST, width=2.5)),  row=1, col=1)
    fig.add_trace(go.Scatter(x=vol_range*100, y=ruins,   name="Prob Ruin (%)",
                             line=dict(color="#FF8C00", width=2.5), fill="tozeroy",
                             fillcolor="rgba(232,71,42,0.1)"),    row=1, col=2)

    for i in range(1, 3):
        fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, title_text="Annual Vol (%)",
                         ticksuffix="%", row=1, col=i)
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=1, col=i)

    _layout(fig, title="Sensitivity to Volatility", height=400)
    return fig