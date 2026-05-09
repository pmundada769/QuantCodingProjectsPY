#cli.py

# Pairs Trading CLI
# python cli.py --pair XOM CVX
# python cli.py --pair GS MS --start 2015-01-01

import argparse
import numpy as np
from pairs import fetch_prices, analyse_pair

RESET="\033[0m"; BOLD="\033[1m"; INDIGO="\033[94m"
MINT="\033[92m"; CORAL="\033[91m"; GREY="\033[90m"; WHITE="\033[97m"

def hr(): print(f"{GREY}{'─'*58}{RESET}")
def section(t): hr(); print(f"{BOLD}{WHITE}  {t}{RESET}"); hr()
def row(l, v, note="", col=WHITE):
    print(f"  {GREY}{l:<24}{RESET}{col}{v:<20}{RESET}{GREY}{note}{RESET}")

parser = argparse.ArgumentParser()
parser.add_argument("--pair",  nargs=2, default=["XOM","CVX"])
parser.add_argument("--start", default="2015-01-01")
parser.add_argument("--entry", type=float, default=2.0)
parser.add_argument("--exit",  type=float, default=0.5)
args = parser.parse_args()

a, b = [t.upper() for t in args.pair]
print(f"\n  {BOLD}{INDIGO}PAIRS TRADING  |  Statistical Arbitrage{RESET}")
print(f"  {GREY}Engle-Granger Cointegration + Z-Score Mean Reversion{RESET}\n")

prices = fetch_prices([a, b], start=args.start)
r      = analyse_pair(a, b, prices, entry=args.entry, exit=args.exit)

section(f"PAIR: {a} / {b}")
coint_col = MINT if r.cointegrated else CORAL
coint_str = "COINTEGRATED ✓" if r.cointegrated else "NOT COINTEGRATED ✗"
row("Cointegration",      coint_str,                color=coint_col + BOLD)
row("EG p-value",         f"{r.eg_pvalue:.6f}",     "(< 0.05 = cointegrated)")
row("Hedge Ratio",        f"{r.hedge_ratio:.6f}",   f"long 1 {a}, short {r.hedge_ratio:.4f} {b}")
row("Half-Life",          f"{r.half_life:.1f} days" if not np.isnan(r.half_life) else "N/A",
                          "(mean reversion speed)")

section("PERFORMANCE")
row("Sharpe Ratio",       f"{r.sharpe:.4f}",        color=MINT if r.sharpe > 0.8 else CORAL)
row("Max Drawdown",       f"{r.max_drawdown*100:.2f}%", color=CORAL)
row("Round-Trip Trades",  str(r.n_trades))

section("CURRENT SPREAD")
cur_z = r.zscore.dropna().iloc[-1]
cur_s = r.signals.iloc[-1]
z_col = MINT if cur_z < -2 else (CORAL if cur_z > 2 else WHITE)
row("Current Z-Score",    f"{cur_z:.4f}",           color=z_col)
pos_str = "LONG spread" if cur_s == 1 else ("SHORT spread" if cur_s == -1 else "FLAT")
row("Current Signal",     pos_str,                  color=MINT if cur_s == 1 else (CORAL if cur_s == -1 else WHITE))

hr(); print()