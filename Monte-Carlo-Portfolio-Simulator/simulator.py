#simulator.py

'''
Core Monte Carlo simulation engine.
Generates N portfolio paths using Geometric Brownian Motion (GBM),
then computes risk metrics: VaR, CVaR, probability of ruin, and more.
'''
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SimulationResult:
    '''container for all outputs from one simulation run'''

    '''raw simulation data'''
    paths:            np.ndarray   # shape: (n_simulations, n_days+1)
    final_values:     np.ndarray   # shape: (n_simulations,) - terminal portfolio values
    daily_returns:    np.ndarray   # shape: (n_simulations, n_days)

    '''risk metrics'''
    var_95:           float        # Value at Risk at 95% confidence
    var_99:           float        # Value at Risk at 99% confidence
    cvar_95:          float        # Conditional VaR (Expected Shortfall) at 95%
    cvar_99:          float        # Conditional VaR at 99%
    prob_ruin:        float        # probability of losing more than ruin_threshold
    prob_profit:      float        # probability of ending above starting value
    expected_return:  float        # mean final portfolio value
    median_return:    float        # median final portfolio value
    best_case:        float        # 95th percentile final value
    worst_case:       float        # 5th percentile final value
    sharpe:           float        # annualised Sharpe of simulated paths

    '''input parameters stored for reference'''
    initial_value:    float
    n_simulations:    int
    n_days:           int
    annual_return:    float
    annual_vol:       float
    ruin_threshold:   float


def run_simulation(
    initial_value:   float = 100_000,
    annual_return:   float = 0.08,
    annual_vol:      float = 0.15,
    n_simulations:   int   = 10_000,
    n_days:          int   = 252,
    ruin_threshold:  float = 0.20,
    seed:            int   = None,
) -> SimulationResult:
    '''
    Simulates portfolio paths using Geometric Brownian Motion (GBM).

    GBM models a price that:
    - Drifts upward at the expected return rate
    - Gets shocked each day by a random normal draw scaled by volatility
    - Cannot go below zero (log-normal distribution of returns)

    This is the same model underlying Black-Scholes.
    '''

    if seed is not None:
        np.random.seed(seed)

    '''convert annual parameters to daily - sqrt(252) rule for vol, /252 for return'''
    daily_return = annual_return / 252
    daily_vol    = annual_vol    / np.sqrt(252)

    '''
    GBM formula for one day:
    S(t+1) = S(t) * exp( (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z )
    where Z ~ N(0,1) is a standard normal random draw

    The (mu - sigma^2/2) term is the Ito correction - it adjusts the drift
    so the expected value of S(T) is S(0)*exp(mu*T).
    Without it, the average of many paths would drift lower due to
    Jensen's inequality. This detail impresses interviewers.
    '''
    drift       = (daily_return - 0.5 * daily_vol**2)
    shocks      = np.random.standard_normal((n_simulations, n_days))
    log_returns = drift + daily_vol * shocks

    '''
    - cumsum gives cumulative log returns over time
    - exp converts back from log space to price space
    - multiply by initial_value to get dollar values
    - prepend column of ones so all paths start at initial_value
    '''
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    price_relatives        = np.exp(cumulative_log_returns)
    paths_relative         = np.hstack([np.ones((n_simulations, 1)), price_relatives])
    paths                  = paths_relative * initial_value

    '''terminal values - the final portfolio value of each path'''
    final_values = paths[:, -1]

    '''
    Value at Risk (VaR):
    - VaR 95% = the loss you will NOT exceed in 95% of scenarios
    - In the worst 5% of cases, losses are at least this bad
    - Reported as a positive dollar loss figure
    '''
    var_95 = initial_value - np.percentile(final_values, 5)
    var_99 = initial_value - np.percentile(final_values, 1)

    '''
    Conditional VaR (CVaR) / Expected Shortfall:
    - Average loss in the worst 5% (or 1%) of scenarios
    - Answers: when things go really bad, how bad is the average outcome?
    - CVaR is always >= VaR and is the superior risk measure
    - Basel III / bank regulation moved from VaR to CVaR for exactly this reason
    '''
    tail_95 = final_values[final_values <= np.percentile(final_values, 5)]
    tail_99 = final_values[final_values <= np.percentile(final_values, 1)]
    cvar_95 = initial_value - tail_95.mean() if len(tail_95) > 0 else var_95
    cvar_99 = initial_value - tail_99.mean() if len(tail_99) > 0 else var_99

    '''
    Probability of ruin:
    - fraction of paths that lost more than ruin_threshold of initial value
    - ruin_threshold=0.20 means losing more than 20% counts as ruin
    '''
    ruin_level  = initial_value * (1 - ruin_threshold)
    prob_ruin   = (final_values < ruin_level).mean()
    prob_profit = (final_values > initial_value).mean()

    '''summary statistics across all terminal values'''
    expected_return = final_values.mean()
    median_return   = np.median(final_values)
    best_case       = np.percentile(final_values, 95)
    worst_case      = np.percentile(final_values, 5)

    '''annualised Sharpe across all simulated daily returns'''
    mean_daily = log_returns.mean()
    std_daily  = log_returns.std()
    sharpe     = (mean_daily * 252) / (std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

    return SimulationResult(
        paths           = paths,
        final_values    = final_values,
        daily_returns   = log_returns,
        var_95          = var_95,
        var_99          = var_99,
        cvar_95         = cvar_95,
        cvar_99         = cvar_99,
        prob_ruin       = prob_ruin,
        prob_profit     = prob_profit,
        expected_return = expected_return,
        median_return   = median_return,
        best_case       = best_case,
        worst_case      = worst_case,
        sharpe          = sharpe,
        initial_value   = initial_value,
        n_simulations   = n_simulations,
        n_days          = n_days,
        annual_return   = annual_return,
        annual_vol      = annual_vol,
        ruin_threshold  = ruin_threshold,
    )


def run_multi_asset_simulation(
    weights:        list,
    annual_returns: list,
    annual_vols:    list,
    correlations:   np.ndarray,
    initial_value:  float = 100_000,
    n_simulations:  int   = 10_000,
    n_days:         int   = 252,
    ruin_threshold: float = 0.20,
    seed:           int   = None,
) -> SimulationResult:
    '''
    Multi-asset portfolio simulation with correlated returns.

    Uses Cholesky decomposition to generate correlated random shocks.
    Cholesky finds matrix L such that L @ L.T = covariance matrix.
    Multiplying uncorrelated normals by L injects the correct correlations.
    Standard method for simulating correlated GBM paths.
    '''
    if seed is not None:
        np.random.seed(seed)

    n_assets          = len(weights)
    weights           = np.array(weights)
    daily_returns_arr = np.array(annual_returns) / 252
    daily_vols_arr    = np.array(annual_vols)    / np.sqrt(252)

    '''build daily covariance matrix from vols and correlations'''
    cov_matrix = np.outer(daily_vols_arr, daily_vols_arr) * correlations

    '''Cholesky decomposition - lower triangular L such that L @ L.T = cov'''
    L = np.linalg.cholesky(cov_matrix)

    '''Ito-corrected drift per asset'''
    drifts = daily_returns_arr - 0.5 * daily_vols_arr**2

    '''uncorrelated standard normals, then correlate via Cholesky'''
    Z                 = np.random.standard_normal((n_simulations, n_days, n_assets))
    correlated_shocks = Z @ L.T

    '''log returns per asset, then weighted into portfolio log return'''
    log_returns_assets    = drifts + correlated_shocks
    portfolio_log_returns = (log_returns_assets * weights).sum(axis=2)

    '''build portfolio paths'''
    cumulative   = np.cumsum(portfolio_log_returns, axis=1)
    price_rel    = np.exp(cumulative)
    paths_rel    = np.hstack([np.ones((n_simulations, 1)), price_rel])
    paths        = paths_rel * initial_value
    final_values = paths[:, -1]

    '''risk metrics - same calculations as single-asset'''
    var_95  = initial_value - np.percentile(final_values, 5)
    var_99  = initial_value - np.percentile(final_values, 1)
    tail_95 = final_values[final_values <= np.percentile(final_values, 5)]
    tail_99 = final_values[final_values <= np.percentile(final_values, 1)]
    cvar_95 = initial_value - tail_95.mean() if len(tail_95) > 0 else var_95
    cvar_99 = initial_value - tail_99.mean() if len(tail_99) > 0 else var_99

    ruin_level  = initial_value * (1 - ruin_threshold)
    prob_ruin   = (final_values < ruin_level).mean()
    prob_profit = (final_values > initial_value).mean()

    mean_daily = portfolio_log_returns.mean()
    std_daily  = portfolio_log_returns.std()
    sharpe     = (mean_daily * 252) / (std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

    '''portfolio-level annual vol from covariance matrix'''
    annual_cov      = cov_matrix * 252
    port_annual_vol = float(np.sqrt(weights @ annual_cov @ weights))
    port_annual_ret = float(np.dot(weights, annual_returns))

    return SimulationResult(
        paths           = paths,
        final_values    = final_values,
        daily_returns   = portfolio_log_returns,
        var_95          = var_95,
        var_99          = var_99,
        cvar_95         = cvar_95,
        cvar_99         = cvar_99,
        prob_ruin       = prob_ruin,
        prob_profit     = prob_profit,
        expected_return = final_values.mean(),
        median_return   = np.median(final_values),
        best_case       = np.percentile(final_values, 95),
        worst_case      = np.percentile(final_values, 5),
        sharpe          = sharpe,
        initial_value   = initial_value,
        n_simulations   = n_simulations,
        n_days          = n_days,
        annual_return   = port_annual_ret,
        annual_vol      = port_annual_vol,
        ruin_threshold  = ruin_threshold,
    )