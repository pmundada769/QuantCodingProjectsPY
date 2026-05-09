#cli.py

# Efficient Frontier — Command Line Interface
# Usage:
#   python cli.py --tickers AAPL MSFT GOOGL NVDA JPM
#   python cli.py --tickers SPY TLT GLD --start 2019-01-01 --rf 4.5

import argparse
import numpy as np
from optimizer import get_price_data, compute_frontier

RESET  = "\033[0m"
BOLD   = "\033[1m"
BLUE   = "\033[94m"
GOLD   = "\033[93m"
MINT   = "\033[92m"
CORAL  = "\033[91m"
GREY   = "\033[90m"
WHITE  = "\033[97m"


def hr(width=60, color=GREY):
    print(f"{color}{'─' * width}{RESET}")

def section(title):
    hr()
    print(f"{BOLD}{WHITE}  {title}{RESET}")
    hr()

def row(label, value, note="", color=WHITE):
    note_str = f"  {GREY}{note}{RESET}" if note else ""
    print(f"  {GREY}{label:<22}{RESET}{color}{value:<22}{RESET}{note_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Efficient Frontier / Mean-Variance Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
    parser.add_argument("--start",   default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--rf",      type=float, default=4.0, help="Risk-free rate %%")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    rf      = args.rf / 100

    print()
    print(f"  {BOLD}{BLUE}EFFICIENT FRONTIER  |  Markowitz Mean-Variance{RESET}")
    print()

    print(f"  Downloading {len(tickers)} assets from {args.start}...")
    try:
        prices = get_price_data(tickers, start=args.start)
        result = compute_frontier(prices, risk_free_rate=rf)
    except Exception as e:
        print(f"\n  {CORAL}Error: {e}{RESET}\n")
        return

    ms = result.max_sharpe
    mv = result.min_vol

    section("UNIVERSE")
    for i, t in enumerate(result.tickers):
        ret = result.mean_returns[i] * 100
        vol = np.sqrt(result.cov_matrix[i, i]) * 100
        sr  = (result.mean_returns[i] - rf) / np.sqrt(result.cov_matrix[i, i])
        row(t, f"ret={ret:.1f}%  vol={vol:.1f}%", f"SR={sr:.2f}")

    section(f"★  MAX SHARPE PORTFOLIO  (Sharpe = {ms.sharpe:.3f})")
    row("Expected Return",  f"{ms.expected_return*100:.2f}%", color=GOLD)
    row("Volatility",       f"{ms.volatility*100:.2f}%")
    row("Sharpe Ratio",     f"{ms.sharpe:.4f}", color=GOLD + BOLD)
    print()
    print(f"  {GREY}Weights:{RESET}")
    for t, w in sorted(ms.weights.items(), key=lambda x: -x[1]):
        bar  = "█" * int(w * 40)
        print(f"    {GOLD}{t:<8}{RESET}  {bar:<40}  {w*100:.1f}%")

    section(f"◆  MIN VOLATILITY PORTFOLIO  (Vol = {mv.volatility*100:.2f}%)")
    row("Expected Return",  f"{mv.expected_return*100:.2f}%", color=MINT)
    row("Volatility",       f"{mv.volatility*100:.2f}%",      color=MINT + BOLD)
    row("Sharpe Ratio",     f"{mv.sharpe:.4f}")
    print()
    print(f"  {GREY}Weights:{RESET}")
    for t, w in sorted(mv.weights.items(), key=lambda x: -x[1]):
        bar  = "█" * int(w * 40)
        print(f"    {MINT}{t:<8}{RESET}  {bar:<40}  {w*100:.1f}%")

    hr()
    print()


if __name__ == "__main__":
    main()