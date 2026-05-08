#tests.py

'''
Unit tests for the Monte Carlo simulation engine.
Run with: python tests.py
'''
import numpy as np
import sys
from simulator import run_simulation, run_multi_asset_simulation

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


'''run a base simulation for most tests'''
BASE = run_simulation(
    initial_value  = 100_000,
    annual_return  = 0.08,
    annual_vol     = 0.15,
    n_simulations  = 5_000,
    n_days         = 252,
    ruin_threshold = 0.20,
    seed           = 42,
)

print()
print("  MONTE CARLO ENGINE — TEST SUITE")
print("  " + "─" * 42)

'''[1] output shape checks'''
check("paths shape correct",
      BASE.paths.shape == (5_000, 253))
check("final_values length correct",
      len(BASE.final_values) == 5_000)
check("all paths start at initial value",
      np.allclose(BASE.paths[:, 0], 100_000))

'''[2] all paths positive (GBM cannot go below zero)'''
check("all path values positive",
      (BASE.paths > 0).all())

'''[3] VaR ordering: VaR 99 >= VaR 95'''
check("VaR 99 >= VaR 95",
      BASE.var_99 >= BASE.var_95)

'''[4] CVaR >= VaR at same confidence level'''
check("CVaR 95 >= VaR 95",
      BASE.cvar_95 >= BASE.var_95)
check("CVaR 99 >= VaR 99",
      BASE.cvar_99 >= BASE.var_99)

'''[5] probabilities are valid fractions'''
check("prob_ruin in [0,1]",
      0 <= BASE.prob_ruin <= 1)
check("prob_profit in [0,1]",
      0 <= BASE.prob_profit <= 1)

'''[6] higher vol = higher VaR'''
low_vol  = run_simulation(100_000, 0.08, 0.05,  2_000, 252, seed=42)
high_vol = run_simulation(100_000, 0.08, 0.40,  2_000, 252, seed=42)
check("higher vol → higher VaR 95",
      high_vol.var_95 > low_vol.var_95)
check("higher vol → higher prob of ruin",
      high_vol.prob_ruin > low_vol.prob_ruin)

'''[7] zero vol = deterministic path (all paths identical)'''
zero_vol = run_simulation(100_000, 0.08, 0.001, 1_000, 252, seed=42)
check("near-zero vol → near-zero VaR (deterministic drift)",
      zero_vol.var_95 < 5_000)

'''[8] ruin threshold respected'''
tight_ruin = run_simulation(100_000, 0.08, 0.15, 2_000, 252, ruin_threshold=0.01, seed=42)
wide_ruin  = run_simulation(100_000, 0.08, 0.15, 2_000, 252, ruin_threshold=0.50, seed=42)
check("tighter ruin threshold → higher prob of ruin",
      tight_ruin.prob_ruin >= wide_ruin.prob_ruin)

'''[9] seed reproducibility'''
r1 = run_simulation(100_000, 0.08, 0.15, 1_000, 252, seed=99)
r2 = run_simulation(100_000, 0.08, 0.15, 1_000, 252, seed=99)
check("same seed → identical results",
      np.allclose(r1.final_values, r2.final_values))

'''[10] multi-asset simulation runs without error'''
try:
    multi = run_multi_asset_simulation(
        weights        = [0.6, 0.4],
        annual_returns = [0.10, 0.04],
        annual_vols    = [0.18, 0.05],
        correlations   = np.array([[1.0, 0.2], [0.2, 1.0]]),
        initial_value  = 100_000,
        n_simulations  = 1_000,
        n_days         = 252,
        seed           = 42,
    )
    check("multi-asset simulation runs",       True)
    check("multi-asset paths shape correct",   multi.paths.shape == (1_000, 253))
    check("multi-asset all paths positive",    (multi.paths > 0).all())
except Exception as e:
    check("multi-asset simulation runs",       False, str(e))

print()
print(f"  Results: {tests_passed} passed, {tests_failed} failed")
print()

if tests_failed > 0:
    sys.exit(1)