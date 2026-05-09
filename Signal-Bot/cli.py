#cli.py

# Signal Bot CLI
# python cli.py --tickers SPY QQQ TLT GLD EEM --target-vol 15

import argparse
import numpy as np
from signal_bot import run_bot, generate_orders, ALPACA_AVAILABLE

RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; RED="\033[91m"
GOLD="\033[93m"; GREY="\033[90m"; WHITE="\033[97m"

def hr(): print(f"{GREY}{'─'*60}{RESET}")
def section(t): hr(); print(f"{BOLD}{WHITE}  {t}{RESET}"); hr()
def row(l, v, col=WHITE): print(f"  {GREY}{l:<24}{RESET}{col}{v}{RESET}")

parser = argparse.ArgumentParser()
parser.add_argument("--tickers",    nargs="+", default=["SPY","QQQ","TLT","GLD","EEM","VNQ"])
parser.add_argument("--target-vol", type=float, default=15.0)
parser.add_argument("--start",      default="2015-01-01")
parser.add_argument("--dd-stop",    type=float, default=20.0)
args = parser.parse_args()

print(f"\n  {BOLD}{GREEN}UNIFIED TRADING SIGNAL BOT{RESET}")
print(f"  {GREY}TSMOM + Cross-Sect + Vol Regime + Trend + Sentiment{RESET}\n")

result = run_bot(
    [t.upper() for t in args.tickers],
    start        = args.start,
    target_vol   = args.target_vol / 100,
    dd_threshold = args.dd_stop / 100,
)

section("BACKTEST PERFORMANCE")
row("Ann. Return",  f"{result.ann_return*100:.2f}%",    GREEN if result.ann_return > 0 else RED)
row("Ann. Vol",     f"{result.ann_vol*100:.2f}%")
row("Sharpe",       f"{result.sharpe:.4f}",             GREEN if result.sharpe > 0.8 else GOLD)
row("Sortino",      f"{result.sortino:.4f}")
row("Max Drawdown", f"{result.max_drawdown*100:.2f}%",  RED)
row("Calmar",       f"{result.calmar:.4f}")
row("Hit Rate",     f"{result.hit_rate*100:.1f}%")

section("CURRENT SIGNALS")
for t, s in result.signals.items():
    direction = "▲ LONG " if s.final_position > 0.05 else ("▼ SHORT" if s.final_position < -0.05 else "— FLAT")
    if s.dd_stop_active:
        direction = "🛑 STOPPED"
    col = GREEN if s.final_position > 0.05 else (RED if s.final_position < -0.05 else WHITE)
    row(t, f"{direction}  pos={s.final_position:+.3f}  σ={s.realised_vol*100:.1f}%  agree={s.signal_agreement*100:.0f}%", col)

section("PROPOSED ORDERS  (dry run)")
orders = generate_orders(result.signals, portfolio_value=100_000)
for o in orders:
    if o["side"] == "flat":
        continue
    col = GREEN if o["side"] == "buy" else RED
    row(o["ticker"], f"{o['side'].upper():5}  {o['shares']:4} shares  alloc=${o['alloc']:,.0f}", col)

hr()
if not ALPACA_AVAILABLE:
    print(f"  {GREY}pip install alpaca-py  to submit live paper orders{RESET}")
print()