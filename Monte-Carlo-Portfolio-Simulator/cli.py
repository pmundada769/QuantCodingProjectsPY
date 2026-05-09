#cli.py

'''
Monte Carlo Portfolio Simulator — Command Line Interface
Usage examples:

  python cli.py --value 100000 --return 8 --vol 15 --days 252
  python cli.py --value 500000 --return 10 --vol 20 --days 504 --sims 20000
  python cli.py --value 100000 --return 8 --vol 15 --days 252 --ruin 30
'''
import argparse
import numpy as np
from simulator import run_simulation

'''terminal colour codes'''
RESET  = "\033[0m"
BOLD   = "\033[1m"
AMBER  = "\033[93m"
RED    = "\033[91m"
GREEN  = "\033[92m"
TEAL   = "\033[96m"
GREY   = "\033[90m"
WHITE  = "\033[97m"


def hr(width=58, color=GREY):
    print(f"{color}{'─' * width}{RESET}")

def section(title):
    hr()
    print(f"{BOLD}{WHITE}  {title}{RESET}")
    hr()

def row(label, value, note="", color=WHITE):
    note_str = f"  {GREY}{note}{RESET}" if note else ""
    print(f"  {GREY}{label:<22}{RESET}{color}{value:<20}{RESET}{note_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Portfolio Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--value",  type=float, default=100_000, help="Initial portfolio value ($)")
    parser.add_argument("--return", type=float, default=8.0,     help="Expected annual return (%)", dest="ret")
    parser.add_argument("--vol",    type=float, default=15.0,    help="Annual volatility (%)")
    parser.add_argument("--days",   type=int,   default=252,     help="Simulation horizon in trading days")
    parser.add_argument("--sims",   type=int,   default=10_000,  help="Number of simulated paths")
    parser.add_argument("--ruin",   type=float, default=20.0,    help="Ruin threshold: loss %% that counts as ruin")
    parser.add_argument("--seed",   type=int,   default=None,    help="Random seed for reproducibility")
    args = parser.parse_args()

    result = run_simulation(
        initial_value  = args.value,
        annual_return  = args.ret  / 100,
        annual_vol     = args.vol  / 100,
        n_simulations  = args.sims,
        n_days         = args.days,
        ruin_threshold = args.ruin / 100,
        seed           = args.seed,
    )

    horizon_label = f"{args.days} days" if args.days < 252 else f"{args.days // 252} year(s)"

    print()
    print(f"  {BOLD}{AMBER}MONTE CARLO PORTFOLIO SIMULATOR{RESET}")
    print()

    section("INPUTS")
    row("Initial Value",    f"${result.initial_value:,.2f}")
    row("Annual Return",    f"{result.annual_return*100:.2f}%")
    row("Annual Vol",       f"{result.annual_vol*100:.2f}%")
    row("Horizon",          horizon_label)
    row("Simulations",      f"{result.n_simulations:,}")
    row("Ruin Threshold",   f">{result.ruin_threshold*100:.0f}% loss")

    section("TERMINAL VALUES")
    row("Expected Value",   f"${result.expected_return:,.2f}", color=AMBER)
    row("Median Value",     f"${result.median_return:,.2f}")
    row("Best Case (p95)",  f"${result.best_case:,.2f}",  color=GREEN)
    row("Worst Case (p5)",  f"${result.worst_case:,.2f}",  color=RED)
    gain = result.expected_return - result.initial_value
    row("Expected Gain",    f"${gain:+,.2f}", color=GREEN if gain >= 0 else RED)

    section("RISK METRICS")
    row("VaR 95%",          f"${result.var_95:,.2f}",
        f"({result.var_95/result.initial_value*100:.1f}% of portfolio)", color=AMBER)
    row("CVaR 95%",         f"${result.cvar_95:,.2f}",
        f"({result.cvar_95/result.initial_value*100:.1f}% of portfolio)", color=AMBER)
    row("VaR 99%",          f"${result.var_99:,.2f}",
        f"({result.var_99/result.initial_value*100:.1f}% of portfolio)", color=RED)
    row("CVaR 99%",         f"${result.cvar_99:,.2f}",
        f"({result.cvar_99/result.initial_value*100:.1f}% of portfolio)", color=RED)

    section("PROBABILITY")
    ruin_col   = RED   if result.prob_ruin   > 0.15 else (AMBER if result.prob_ruin   > 0.05 else GREEN)
    profit_col = GREEN if result.prob_profit > 0.60 else (AMBER if result.prob_profit > 0.45 else RED)
    row("Prob of Profit",   f"{result.prob_profit*100:.2f}%", color=profit_col)
    row("Prob of Ruin",     f"{result.prob_ruin*100:.2f}%",   color=ruin_col)
    row("Sharpe (sim)",     f"{result.sharpe:.4f}")

    hr()
    print()


if __name__ == "__main__":
    main()