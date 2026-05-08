#!/usr/bin/env python3
"""
Options Pricer — Command-Line Interface
Usage examples:

  python cli.py --spot 100 --strike 100 --days 30 --vol 20 --rate 5
  python cli.py --spot 150 --strike 145 --days 60 --vol 25 --rate 4.5 --type put
  python cli.py --spot 100 --strike 100 --days 30 --vol 20 --rate 5 --market-price 3.50
"""

import argparse
import sys
import numpy as np
from black_scholes import black_scholes, implied_volatility

# ───────── terminal colours ─────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
TEAL   = "\033[96m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREY   = "\033[90m"
WHITE  = "\033[97m"
GREEN  = "\033[92m"


def hr(char="─", width=56, color=GREY):
    print(f"{color}{char * width}{RESET}")


def section(title: str):
    hr()
    print(f"{BOLD}{WHITE}  {title}{RESET}")
    hr()


def row(label: str, value: str, note: str = "", color: str = WHITE):
    note_str = f"  {GREY}{note}{RESET}" if note else ""
    print(f"  {GREY}{label:<16}{RESET}{color}{value:<18}{RESET}{note_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Black-Scholes Options Pricer with Greeks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--spot",         type=float, required=True,  help="Current underlying price")
    parser.add_argument("--strike",       type=float, required=True,  help="Option strike price")
    parser.add_argument("--days",         type=float, required=True,  help="Days to expiry")
    parser.add_argument("--vol",          type=float, required=True,  help="Implied vol in %% (e.g. 20 for 20%%)")
    parser.add_argument("--rate",         type=float, required=True,  help="Risk-free rate in %% (e.g. 5 for 5%%)")
    parser.add_argument("--div",          type=float, default=0.0,    help="Dividend yield in %% (default: 0)")
    parser.add_argument("--type",         type=str,   default="call", choices=["call", "put"])
    parser.add_argument("--market-price", type=float, default=None,   help="Market price → compute IV")

    args = parser.parse_args()

    S     = args.spot
    K     = args.strike
    T     = args.days / 365.0
    sigma = args.vol  / 100.0
    r     = args.rate / 100.0
    q     = args.div  / 100.0
    otype = args.type

    res = black_scholes(S, K, T, r, sigma, q, otype)

    # moneyness
    if abs(S - K) / K < 0.01:
        money = "ATM"
    elif otype == "call":
        money = "ITM" if S > K else "OTM"
    else:
        money = "ITM" if S < K else "OTM"

    col_price = TEAL if otype == "call" else RED

    print()
    print(f"  {BOLD}{WHITE}OPTIONS PRICER  |  Black-Scholes (Generalised Merton){RESET}")
    print()

    section("CONTRACT")
    row("Type",        f"{otype.upper()} [{money}]",  color=col_price)
    row("Spot (S)",    f"${S:.4f}")
    row("Strike (K)",  f"${K:.4f}")
    row("Expiry",      f"{args.days:.0f} days  ({T:.4f} yr)")
    row("Vol (σ)",     f"{sigma*100:.2f}%")
    row("Rate (r)",    f"{r*100:.2f}%")
    row("Div yield",   f"{q*100:.2f}%")

    section("PRICE")
    row("Option Price", f"${res.price:.6f}",    color=col_price + BOLD)
    row("Intrinsic",    f"${res.intrinsic:.6f}")
    row("Time Value",   f"${res.time_value:.6f}")
    row("d₁",           f"{res.d1:.6f}")
    row("d₂",           f"{res.d2:.6f}")
    row("N(d₁)",        f"{res.nd1:.6f}")
    row("N(d₂)",        f"{res.nd2:.6f}")

    section("FIRST-ORDER GREEKS")
    row("Δ  Delta",  f"{res.delta:+.6f}",  "per $1 move in S")
    row("ν  Vega",   f"{res.vega:+.6f}",   "per 1% move in σ")
    row("Θ  Theta",  f"{res.theta:+.6f}",  "per calendar day")
    row("ρ  Rho",    f"{res.rho:+.6f}",    "per 1% move in r")

    section("SECOND-ORDER GREEKS")
    row("Γ  Gamma",  f"{res.gamma:+.6f}",  "ΔΔ per $1 move in S")
    row("   Vanna",  f"{res.vanna:+.6f}",  "dDelta/dVol")
    row("   Charm",  f"{res.charm:+.6f}",  "dDelta/dTime (per day)")
    row("   Volga",  f"{res.volga:+.6f}",  "dVega/dVol")

    # Break-even
    be = K + res.price if otype == "call" else K - res.price
    hr()
    row("Break-even",    f"${be:.4f}")

    # IV if market price provided
    if args.market_price is not None:
        hr()
        iv = implied_volatility(args.market_price, S, K, T, r, q, otype)
        if np.isnan(iv):
            print(f"  {RED}IV solver did not converge for market price ${args.market_price:.4f}{RESET}")
        else:
            print(f"  {GREEN}Implied Volatility for ${args.market_price:.4f}:  {iv*100:.4f}%{RESET}")

    hr()
    print()


if __name__ == "__main__":
    main()