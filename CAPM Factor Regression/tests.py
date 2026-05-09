#tests.py

# Unit tests for CAPM / FF3 regression engine using synthetic data.
# python tests.py

import numpy as np
import pandas as pd
import sys
from regression import run_capm, run_ff3, rolling_beta, factor_decomposition

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:  print(f"  ✓  {name}"); passed += 1
    else:     print(f"  ✗  {name}  {detail}"); failed += 1


# build synthetic data with known properties
np.random.seed(7)
n = 120
idx = pd.date_range("2010-01-31", periods=n, freq="ME")

mkt_rf = pd.Series(np.random.normal(0.007, 0.045, n), index=idx)
smb    = pd.Series(np.random.normal(0.002, 0.030, n), index=idx)
hml    = pd.Series(np.random.normal(0.002, 0.030, n), index=idx)
rf     = pd.Series(np.full(n, 0.0003),                index=idx)

TRUE_ALPHA = 0.003
TRUE_BETA  = 1.2

noise  = pd.Series(np.random.normal(0, 0.02, n), index=idx)
excess = TRUE_ALPHA + TRUE_BETA * mkt_rf + noise
stock  = excess + rf

factors_df = pd.DataFrame({"Mkt-RF": mkt_rf, "SMB": smb, "HML": hml, "RF": rf})
ticker = "TEST"

# monkey-patch fetch functions to use synthetic data
import regression as reg_mod
_orig_fetch_stock  = reg_mod.fetch_stock_returns
_orig_fetch_ff3    = reg_mod.fetch_ff3_factors
reg_mod.fetch_stock_returns = lambda t, s="2010-01-01": stock
reg_mod.fetch_ff3_factors   = lambda s="2010-01-01": factors_df

capm = run_capm(ticker, factors_df=factors_df)
ff3  = run_ff3( ticker, factors_df=factors_df)

print()
print("  CAPM / FF3 TEST SUITE")
print("  " + "─" * 40)

check("CAPM recovers beta close to 1.2",     abs(capm.beta_market - TRUE_BETA) < 0.15)
check("CAPM recovers alpha close to 0.003",  abs(capm.alpha - TRUE_ALPHA) < 0.003)
check("CAPM R² > 0.8 (clean synthetic data)",capm.r_squared > 0.80)
check("CAPM R² <= 1.0",                      capm.r_squared <= 1.0)
check("CAPM alpha_tstat is float",           isinstance(capm.alpha_tstat, float))
check("CAPM n_obs == 119",                   capm.n_obs == n - 1)

check("FF3 beta_market exists",              ff3.beta_market is not None)
check("FF3 beta_smb exists",                 ff3.beta_smb is not None)
check("FF3 beta_hml exists",                 ff3.beta_hml is not None)
check("FF3 R² >= CAPM R²",                   ff3.r_squared >= capm.r_squared - 0.05)
check("FF3 adj R² <= R²",                    ff3.adj_r_squared <= ff3.r_squared + 1e-8)

roll = rolling_beta(ticker, window=24, factors_df=factors_df)
check("Rolling beta returns DataFrame",      isinstance(roll, pd.DataFrame))
check("Rolling has Beta column",             "Beta" in roll.columns)
check("Rolling has Alpha column",            "Alpha" in roll.columns)
check("Rolling length correct",              len(roll) == n - 1 - 24 + 1)

decomp = factor_decomposition(ff3, factors_df)
check("Decomposition returns DataFrame",     isinstance(decomp, pd.DataFrame))
check("Decomposition has Contribution col",  "Contribution %" in decomp.columns)

reg_mod.fetch_stock_returns = _orig_fetch_stock
reg_mod.fetch_ff3_factors   = _orig_fetch_ff3

print()
print(f"  Results: {passed} passed, {failed} failed\n")
if failed: sys.exit(1)