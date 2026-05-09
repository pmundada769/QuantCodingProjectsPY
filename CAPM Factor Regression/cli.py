#cli.py

# CAPM / FF3 Factor Regression — CLI
# python cli.py --ticker AAPL
# python cli.py --ticker NVDA --start 2015-01-01
# python cli.py --tickers AAPL MSFT NVDA META JPM

import argparse
import numpy as np
from regression import fetch_ff3_factors, run_capm, run_ff3

RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
GOLD  = "\033[93m"; RED  = "\033[91m"; GREY  = "\033[90m"; WHITE = "\033[97m"

def hr(w=58): print(f"{GREY}{'─'*w}{RESET}")
def section(t): hr(); print(f"{BOLD}{WHITE}  {t}{RESET}"); hr()
def row(l, v, note="", col=WHITE):
    print(f"  {GREY}{l:<22}{RESET}{col}{v:<20}{RESET}{GREY}{note}{RESET}")

def print_result(r, label):
    section(f"{label}  —  {r.ticker}")
    sig = "★ SIGNIFICANT" if abs(r.alpha_tstat) > 2 else "not significant"
    col = GREEN if r.alpha > 0 else RED
    row("Alpha (monthly)",  f"{r.alpha*100:.4f}%",           color=col)
    row("Alpha (annual)",   f"{r.alpha*100*12:.2f}%",         color=col)
    row("Alpha t-stat",     f"{r.alpha_tstat:.3f}",          f"  {sig}")
    row("Alpha p-value",    f"{r.alpha_pval:.4f}")
    row("Beta (Market)",    f"{r.beta_market:.4f}")
    if r.model == "FF3":
        row("Beta SMB",     f"{r.beta_smb:.4f}",  "  +ve = small-cap tilt")
        row("Beta HML",     f"{r.beta_hml:.4f}",  "  +ve = value tilt")
    row("R²",               f"{r.r_squared:.4f}")
    row("Adj R²",           f"{r.adj_r_squared:.4f}")
    row("Residual Std",     f"{r.residual_std*100:.4f}%")
    row("Observations",     f"{r.n_obs}")
    row("Period",           f"{r.start_date} → {r.end_date}")

parser = argparse.ArgumentParser()
parser.add_argument("--ticker",  default="AAPL")
parser.add_argument("--tickers", nargs="+")
parser.add_argument("--start",   default="2010-01-01")
args = parser.parse_args()

print(f"\n  {BOLD}{GREEN}CAPM / FAMA-FRENCH 3-FACTOR REGRESSION{RESET}\n")
factors = fetch_ff3_factors(args.start)

tickers = args.tickers if args.tickers else [args.ticker.upper()]

for t in tickers:
    try:
        capm = run_capm(t, args.start, factors)
        ff3  = run_ff3(t,  args.start, factors)
        print_result(capm, "CAPM")
        print_result(ff3,  "Fama-French 3")
    except Exception as e:
        print(f"\n  {RED}Error for {t}: {e}{RESET}\n")

hr(); print()