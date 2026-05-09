#cli.py

# TSMOM CLI
# python cli.py --tickers SPY QQQ TLT GLD EEM --target-vol 15 --start 2010-01-01

import argparse
from tsmom import run_tsmom, ARCH_AVAILABLE

RESET="\033[0m"; BOLD="\033[1m"; RED="\033[91m"; GOLD="\033[93m"
TEAL="\033[96m"; GREY="\033[90m"; WHITE="\033[97m"

def hr(): print(f"{GREY}{'─'*58}{RESET}")
def section(t): hr(); print(f"{BOLD}{WHITE}  {t}{RESET}"); hr()
def row(l, v, note="", col=WHITE):
    print(f"  {GREY}{l:<22}{RESET}{col}{v:<20}{RESET}{GREY}{note}{RESET}")

parser = argparse.ArgumentParser()
parser.add_argument("--tickers",    nargs="+", default=["SPY","QQQ","TLT","GLD","EEM","VNQ"])
parser.add_argument("--target-vol", type=float, default=15.0)
parser.add_argument("--start",      default="2010-01-01")
parser.add_argument("--no-garch",   action="store_true")
args = parser.parse_args()

use_garch = ARCH_AVAILABLE and not args.no_garch

print(f"\n  {BOLD}{RED}TIME-SERIES MOMENTUM (TSMOM){RESET}")
print(f"  {GREY}Moskowitz, Ooi & Pedersen (2012){RESET}\n")

result = run_tsmom(
    [t.upper() for t in args.tickers],
    start=args.start,
    target_vol=args.target_vol / 100,
    use_garch=use_garch,
)

section("INPUTS")
row("Tickers",         ", ".join(result.tickers))
row("Target Vol",      f"{args.target_vol:.1f}%")
row("Vol estimation",  "GARCH(1,1)" if use_garch else "Rolling 60d std")
row("Start",           args.start)

section("PERFORMANCE")
row("Ann. Return",     f"{result.ann_return*100:.2f}%",   color=GOLD)
row("Ann. Volatility", f"{result.ann_vol*100:.2f}%")
row("Sharpe Ratio",    f"{result.sharpe:.4f}",             color=GOLD if result.sharpe > 0.8 else RED)
row("Sortino Ratio",   f"{result.sortino:.4f}")
row("Max Drawdown",    f"{result.max_drawdown*100:.2f}%",  color=RED)
row("Hit Rate",        f"{result.hit_rate*100:.1f}%")
row("Calmar Ratio",    f"{result.calmar:.4f}")

section("ASSET SIGNALS (latest)")
for asset in result.asset_signals:
    sig  = asset.raw_signal.dropna().iloc[-1] if len(asset.raw_signal.dropna()) > 0 else 0
    pos  = asset.scaled_position.dropna().iloc[-1] if len(asset.scaled_position.dropna()) > 0 else 0
    vol  = asset.realised_vol.dropna().iloc[-1] if len(asset.realised_vol.dropna()) > 0 else 0
    direction = "LONG " if sig > 0 else "SHORT"
    col = TEAL if sig > 0 else RED
    row(asset.ticker, f"{direction}  pos={pos:+.3f}", f"vol={vol*100:.1f}%", col=col)

hr(); print()