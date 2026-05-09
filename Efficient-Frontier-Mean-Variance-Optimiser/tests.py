#tests.py

# Unit tests for the Efficient Frontier optimizer.
# Run with: python tests.py
# Tests use synthetic price data so no internet connection is required.

import numpy as np
import pandas as pd
import sys
from optimizer import compute_frontier, random_portfolios, correlation_matrix

tests_passed = 0
tests_failed = 0

def check(name, condition, detail=""):
    global tests_passed, tests_failed
    if condition:
        print(f"  ✓  {name}")
        tests_passed += 1
    else:
        print(f"  ✗  {name}  {detail}")
        tests_failed += 1


# --- build synthetic price data (no Yahoo Finance needed for tests) ---

np.random.seed(0)
n_days   = 504    # 2 years of daily data
n_assets = 4

# four synthetic assets with different return/vol profiles
tickers = ["A", "B", "C", "D"]
daily_rets = np.array([
    np.random.normal(0.0004, 0.012, n_days),   # steady, low vol
    np.random.normal(0.0006, 0.020, n_days),   # higher return, higher vol
    np.random.normal(0.0002, 0.008, n_days),   # defensive, very low vol
    np.random.normal(0.0008, 0.025, n_days),   # aggressive
])

# build prices from returns: P(t) = P(t-1) * (1 + r(t))
prices_array = np.ones((n_assets, n_days + 1))
for i in range(n_assets):
    for t in range(n_days):
        prices_array[i, t+1] = prices_array[i, t] * (1 + daily_rets[i, t])

prices = pd.DataFrame(
    prices_array.T,
    columns=tickers,
)

RESULT = compute_frontier(prices, risk_free_rate=0.04, n_frontier=40)

print()
print("  EFFICIENT FRONTIER — TEST SUITE")
print("  " + "─" * 44)

# [1] frontier computed with correct number of portfolios (may be fewer if some fail)
check("frontier has portfolios",
      len(RESULT.frontier_portfolios) > 10)

# [2] weights sum to 1 for max sharpe
ws = sum(RESULT.max_sharpe.weights.values())
check("max sharpe weights sum to 1",
      abs(ws - 1.0) < 1e-6, f"sum={ws:.8f}")

# [3] weights sum to 1 for min vol
wv = sum(RESULT.min_vol.weights.values())
check("min vol weights sum to 1",
      abs(wv - 1.0) < 1e-6, f"sum={wv:.8f}")

# [4] all weights non-negative (long-only constraint)
check("max sharpe all weights >= 0",
      all(w >= -1e-8 for w in RESULT.max_sharpe.weights.values()))
check("min vol all weights >= 0",
      all(w >= -1e-8 for w in RESULT.min_vol.weights.values()))

# [5] min vol portfolio has lowest volatility on frontier
frontier_vols = [p.volatility for p in RESULT.frontier_portfolios]
check("min vol portfolio is actually minimum",
      RESULT.min_vol.volatility <= min(frontier_vols) + 1e-4)

# [6] max sharpe has highest Sharpe on frontier
frontier_sharpes = [p.sharpe for p in RESULT.frontier_portfolios]
check("max sharpe portfolio has highest Sharpe",
      RESULT.max_sharpe.sharpe >= max(frontier_sharpes) - 0.05)

# [7] frontier is upward sloping (higher vol = higher return)
rets = [p.expected_return for p in RESULT.frontier_portfolios]
vols = [p.volatility       for p in RESULT.frontier_portfolios]
check("frontier is broadly upward sloping",
      rets[-1] > rets[0])

# [8] expected return and vol are positive
check("max sharpe return > 0",
      RESULT.max_sharpe.expected_return > 0)
check("max sharpe vol > 0",
      RESULT.max_sharpe.volatility > 0)

# [9] random portfolios output shape and validity
rand = random_portfolios(
    RESULT.mean_returns, RESULT.cov_matrix, tickers, n=500, risk_free_rate=0.04
)
check("random portfolios returns correct shape",
      rand.shape == (500, 3))
check("random portfolios all positive vol",
      (rand["Volatility"] > 0).all())

# [10] correlation matrix is symmetric and diagonal is 1
corr = correlation_matrix(prices)
check("correlation matrix is symmetric",
      np.allclose(corr.values, corr.values.T, atol=1e-10))
check("correlation diagonal is 1",
      np.allclose(np.diag(corr.values), 1.0, atol=1e-10))
check("correlation values in [-1, 1]",
      (corr.values >= -1 - 1e-8).all() and (corr.values <= 1 + 1e-8).all())

# [11] three-asset frontier still works
prices_3 = prices[["A", "B", "C"]]
result_3  = compute_frontier(prices_3, risk_free_rate=0.04, n_frontier=20)
check("three-asset frontier runs",
      len(result_3.frontier_portfolios) > 5)

print()
print(f"  Results: {tests_passed} passed, {tests_failed} failed")
print()

if tests_failed > 0:
    sys.exit(1)