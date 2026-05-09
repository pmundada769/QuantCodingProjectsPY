#cli.py

# Financial Statement Analyser — CLI
# python cli.py --ticker AAPL
# python cli.py --tickers AAPL MSFT NVDA META GOOGL
# python cli.py --ticker AAPL --mda --key sk-ant-...

import argparse
from financials import (
    fetch_financials, piotroski_fscore, altman_zscore, get_market_cap,
    revenue_growth, fetch_mda_text, analyse_mda_with_llm,
)

RESET="\033[0m"; BOLD="\033[1m"; GOLD="\033[93m"; TEAL="\033[92m"
CORAL="\033[91m"; GREY="\033[90m"; WHITE="\033[97m"

def hr(): print(f"{GREY}{'─'*58}{RESET}")
def section(t): hr(); print(f"{BOLD}{WHITE}  {t}{RESET}"); hr()
def row(l, v, col=WHITE): print(f"  {GREY}{l:<24}{RESET}{col}{v}{RESET}")

parser = argparse.ArgumentParser()
parser.add_argument("--ticker",  default="AAPL")
parser.add_argument("--tickers", nargs="+")
parser.add_argument("--mda",     action="store_true", help="Fetch and analyse MD&A section")
parser.add_argument("--key",     default="", help="Anthropic API key for LLM analysis")
args = parser.parse_args()

tickers = args.tickers if args.tickers else [args.ticker.upper()]

print(f"\n  {BOLD}{GOLD}FINANCIAL STATEMENT ANALYSER  |  SEC EDGAR{RESET}\n")

for ticker in tickers:
    print(f"\n  {BOLD}{WHITE}▶ {ticker}{RESET}")
    fd = fetch_financials(ticker)
    if fd is None:
        print(f"  {CORAL}Could not fetch data for {ticker}{RESET}")
        continue

    mcap   = get_market_cap(ticker)
    fscore = piotroski_fscore(fd)
    zscore = altman_zscore(fd, market_cap=mcap)

    section(f"{fd.company_name}  —  {ticker}")

    if fscore:
        f_col = TEAL if fscore.total_score >= 7 else (CORAL if fscore.total_score <= 2 else GOLD)
        row("Piotroski F-Score",  f"{fscore.total_score}/9  —  {fscore.interpretation}", f_col)

    if zscore:
        z_col = TEAL if zscore.zone == "Safe" else (CORAL if zscore.zone == "Distress" else GOLD)
        row("Altman Z-Score",     f"{zscore.z_score:.4f}  —  {zscore.zone}", z_col)
        row("  X1 (WC/Assets)",   f"{zscore.x1:.4f}")
        row("  X3 (EBIT/Assets)", f"{zscore.x3:.4f}")
        row("  X4 (Mkt/Liab)",    f"{zscore.x4:.4f}")

    rg = revenue_growth(fd)
    if rg is not None and len(rg) > 0:
        latest = float(rg.iloc[-1])
        r_col  = TEAL if latest > 0 else CORAL
        row("Revenue Growth (YoY)", f"{latest*100:+.1f}%", r_col)

    if args.mda:
        print(f"\n  Fetching MD&A from SEC...")
        mda_text, date = fetch_mda_text(fd.cik)
        if mda_text:
            mda = analyse_mda_with_llm(mda_text, ticker, fd.company_name, args.key)
            if mda:
                section(f"MD&A ANALYSIS  ({date or 'latest 10-K'})")
                tone_col = TEAL if mda.tone == "positive" else (CORAL if mda.tone == "negative" else GOLD)
                row("Tone", mda.tone.upper(), tone_col)
                print(f"\n  {GREY}Key Risks:{RESET}")
                for r in mda.key_risks:
                    print(f"    {CORAL}▸{RESET} {r[:80]}")
                print(f"\n  {GREY}Growth Drivers:{RESET}")
                for d in mda.growth_drivers:
                    print(f"    {TEAL}▸{RESET} {d[:80]}")
        else:
            print(f"  {CORAL}MD&A text not found for {ticker}{RESET}")

hr()
print()