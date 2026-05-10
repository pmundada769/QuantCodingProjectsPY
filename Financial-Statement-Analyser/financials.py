#financials.py

# Financial Statement Analyser
#
# Data source: SEC EDGAR API (free, no key, official US government data)
#   https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
#
# Scores:
#   Piotroski F-Score (2000) — 9-point fundamental quality screen
#   Altman Z-Score (1968)    — bankruptcy prediction model
#
# LLM MD&A Summary:
#   Fetches the MD&A section from the most recent 10-K filing
#   Sends it to Claude claude-sonnet-4-20250514 via the Anthropic API
#   Extracts: key risks, growth drivers, management tone

import numpy as np
import pandas as pd
import requests
import time
import re
from dataclasses import dataclass, field
from typing import Optional
import yfinance as yf

SEC_HEADERS = {
    "User-Agent": "FinancialAnalyser/1.0 prashu@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

# hardcoded CIK fallback for the most common tickers — avoids rate limits
KNOWN_CIKS = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "GOOG":  "0001652044",
    "AMZN":  "0001018724",
    "NVDA":  "0001045810",
    "META":  "0001326801",
    "TSLA":  "0001318605",
    "JPM":   "0000019617",
    "BAC":   "0000070858",
    "JNJ":   "0000200406",
    "XOM":   "0000034088",
    "KO":    "0000021344",
    "PG":    "0000080424",
    "HD":    "0000354950",
    "V":     "0001403161",
    "WMT":   "0000104169",
    "DIS":   "0001001039",
    "NFLX":  "0001065280",
    "INTC":  "0000050863",
    "AMD":   "0000002488",
    "PYPL":  "0001633917",
    "SBUX":  "0000829224",
    "BA":    "0000012927",
    "GS":    "0000886982",
    "MS":    "0000895421",
}


# ─── data classes ─────────────────────────────────────────────────────────────

@dataclass
class FinancialData:
    ticker:           str
    cik:              str
    company_name:     str
    revenue:          Optional[pd.Series] = None
    net_income:       Optional[pd.Series] = None
    operating_income: Optional[pd.Series] = None
    gross_profit:     Optional[pd.Series] = None
    total_assets:     Optional[pd.Series] = None
    current_assets:   Optional[pd.Series] = None
    current_liab:     Optional[pd.Series] = None
    total_liab:       Optional[pd.Series] = None
    long_term_debt:   Optional[pd.Series] = None
    shares_outstanding: Optional[pd.Series] = None
    operating_cf:     Optional[pd.Series] = None
    retained_earnings: Optional[pd.Series] = None


@dataclass
class PiotroskiResult:
    ticker:      str
    year:        str
    total_score: int           # 0–9
    # profitability
    roa_positive:  int
    ocf_positive:  int
    roa_increase:  int
    accruals_low:  int
    # leverage/liquidity
    leverage_dec:  int
    liquidity_inc: int
    no_dilution:   int
    # efficiency
    margin_inc:    int
    turnover_inc:  int
    interpretation: str = ""


@dataclass
class AltmanResult:
    ticker:   str
    year:     str
    z_score:  float
    x1: float; x2: float; x3: float; x4: float; x5: float
    zone:           str    # "Safe" / "Grey" / "Distress"
    interpretation: str = ""


@dataclass
class MDAAnalysis:
    ticker:       str
    filing_date:  str
    mda_excerpt:  str      # raw text excerpt
    llm_summary:  str      # Claude's analysis
    key_risks:    list     # extracted risk factors
    growth_drivers: list   # extracted opportunities
    tone:         str      # "positive" / "cautious" / "negative"


# ─── SEC EDGAR helpers ────────────────────────────────────────────────────────

def get_cik(ticker: str) -> Optional[str]:
    t = ticker.upper()
    # check hardcoded table first — avoids SEC rate limits for common tickers
    if t in KNOWN_CIKS:
        return KNOWN_CIKS[t]
    # fallback: live SEC lookup
    try:
        resp = requests.get(
            TICKER_CIK_URL,
            headers={"User-Agent": "FinancialAnalyser/1.0 prashu@gmail.com",
                     "Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for val in data.values():
            if val.get("ticker", "").upper() == t:
                return str(val["cik_str"]).zfill(10)
        return None
    except Exception as e:
        print(f"[financials] CIK lookup failed for {ticker}: {e}")
        return None


def fetch_edgar_facts(cik: str) -> Optional[dict]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[financials] EDGAR fetch failed: {e}")
        return None


def extract_series(facts: dict, concept: str, unit: str = "USD",
                   include_quarterly: bool = False) -> Optional[pd.Series]:
    try:
        gaap    = facts.get("facts", {}).get("us-gaap", {})
        entries = gaap.get(concept, {}).get("units", {}).get(unit, [])
        forms   = ("10-K", "10-Q") if include_quarterly else ("10-K",)
        annual  = [e for e in entries if e.get("form") in forms and "end" in e and "val" in e]
        if not annual:
            return None
        df = pd.DataFrame(annual)[["end","val"]].drop_duplicates("end")
        df["end"] = pd.to_datetime(df["end"])
        return df.sort_values("end").set_index("end")["val"]
    except Exception:
        return None


def try_concepts(facts: dict, concepts: list, unit: str = "USD") -> Optional[pd.Series]:
    for c in concepts:
        s = extract_series(facts, c, unit)
        if s is not None and len(s) >= 2:
            return s
    return None


def fetch_financials(ticker: str) -> Optional[FinancialData]:
    cik = get_cik(ticker)
    if not cik:
        return None
    time.sleep(0.1)
    facts = fetch_edgar_facts(cik)
    if not facts:
        return None

    return FinancialData(
        ticker           = ticker.upper(),
        cik              = cik,
        company_name     = facts.get("entityName", ticker),
        revenue          = try_concepts(facts, ["Revenues",
                           "RevenueFromContractWithCustomerExcludingAssessedTax",
                           "SalesRevenueNet"]),
        net_income       = try_concepts(facts, ["NetIncomeLoss", "NetIncome"]),
        operating_income = try_concepts(facts, ["OperatingIncomeLoss"]),
        gross_profit     = try_concepts(facts, ["GrossProfit"]),
        total_assets     = try_concepts(facts, ["Assets"]),
        current_assets   = try_concepts(facts, ["AssetsCurrent"]),
        current_liab     = try_concepts(facts, ["LiabilitiesCurrent"]),
        total_liab       = try_concepts(facts, ["Liabilities"]),
        long_term_debt   = try_concepts(facts, ["LongTermDebt",
                           "LongTermDebtNoncurrent"]),
        shares_outstanding = try_concepts(facts, ["CommonStockSharesOutstanding"],
                             unit="shares"),
        operating_cf     = try_concepts(facts, [
                           "NetCashProvidedByUsedInOperatingActivities"]),
        retained_earnings = try_concepts(facts, ["RetainedEarningsAccumulatedDeficit"]),
    )


# ─── Piotroski F-Score ────────────────────────────────────────────────────────

def piotroski_fscore(fd: FinancialData) -> Optional[PiotroskiResult]:
    if fd.total_assets is None or fd.net_income is None:
        return None
    try:
        ta  = fd.total_assets.dropna()
        if len(ta) < 2:
            return None
        t, t1  = ta.index[-1], ta.index[-2]
        ta_t   = float(ta.loc[t])
        ta_t1  = float(ta.loc[t1])

        def get(series, date, default=0.0):
            if series is None:
                return default
            s = series.reindex(ta.index)
            return float(s.loc[date]) if date in s.index and not pd.isna(s.loc[date]) else default

        ni_t   = get(fd.net_income, t)
        ni_t1  = get(fd.net_income, t1)
        cf_t   = get(fd.operating_cf, t)
        ld_t   = get(fd.long_term_debt, t)
        ld_t1  = get(fd.long_term_debt, t1)
        ca_t   = get(fd.current_assets, t)
        ca_t1  = get(fd.current_assets, t1)
        cl_t   = get(fd.current_liab, t)
        cl_t1  = get(fd.current_liab, t1)
        sh_t   = get(fd.shares_outstanding, t)
        sh_t1  = get(fd.shares_outstanding, t1)
        gp_t   = get(fd.gross_profit, t)
        gp_t1  = get(fd.gross_profit, t1)
        rv_t   = get(fd.revenue, t)
        rv_t1  = get(fd.revenue, t1)

        roa_t  = ni_t / ta_t  if ta_t  > 0 else 0
        roa_t1 = ni_t1 / ta_t1 if ta_t1 > 0 else 0

        f1 = int(roa_t > 0)
        f2 = int(cf_t > 0)
        f3 = int(roa_t > roa_t1)
        f4 = int((cf_t / ta_t > roa_t) if ta_t > 0 else False)
        f5 = int((ld_t / ta_t < ld_t1 / ta_t1) if ta_t > 0 and ta_t1 > 0 else False)
        f6 = int(((ca_t / cl_t) > (ca_t1 / cl_t1)) if cl_t > 0 and cl_t1 > 0 else False)
        f7 = int(sh_t <= sh_t1)
        f8 = int(((gp_t / rv_t) > (gp_t1 / rv_t1)) if rv_t > 0 and rv_t1 > 0 else False)
        f9 = int(((rv_t / ta_t) > (rv_t1 / ta_t1)) if ta_t > 0 and ta_t1 > 0 else False)

        total = f1+f2+f3+f4+f5+f6+f7+f8+f9
        interp = ("Strong — high quality, likely outperformer" if total >= 7 else
                  "Weak — fundamental deterioration, potential short" if total <= 2 else
                  "Average — monitor closely")

        return PiotroskiResult(
            ticker=fd.ticker, year=str(t.year), total_score=total,
            roa_positive=f1, ocf_positive=f2, roa_increase=f3, accruals_low=f4,
            leverage_dec=f5, liquidity_inc=f6, no_dilution=f7,
            margin_inc=f8, turnover_inc=f9, interpretation=interp,
        )
    except Exception as e:
        print(f"[financials] Piotroski error: {e}")
        return None


# ─── Altman Z-Score ───────────────────────────────────────────────────────────

def altman_zscore(fd: FinancialData, market_cap: float = None) -> Optional[AltmanResult]:
    if fd.total_assets is None:
        return None
    try:
        ta = fd.total_assets.dropna()
        t  = ta.index[-1]
        ta_val = float(ta.loc[t])
        if ta_val <= 0:
            return None

        def get(series, default=0.0):
            if series is None:
                return default
            s = series.reindex(ta.index)
            return float(s.loc[t]) if t in s.index and not pd.isna(s.loc[t]) else default

        ca = get(fd.current_assets)
        cl = get(fd.current_liab)
        re = get(fd.retained_earnings)
        oi = get(fd.operating_income)
        tl = get(fd.total_liab)
        rv = get(fd.revenue)
        mc = market_cap or 0.0

        x1 = (ca - cl) / ta_val
        x2 = re / ta_val
        x3 = oi / ta_val
        x4 = mc / tl if tl > 0 else 0.0
        x5 = rv / ta_val

        z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

        zone   = "Safe" if z > 2.99 else ("Grey" if z > 1.81 else "Distress")
        interp = {"Safe":    "Low bankruptcy risk — strong financial position",
                  "Grey":    "Moderate risk — monitor for deterioration",
                  "Distress":"HIGH bankruptcy risk — significant financial stress"}[zone]

        return AltmanResult(
            ticker=fd.ticker, year=str(t.year), z_score=round(z, 4),
            x1=round(x1,4), x2=round(x2,4), x3=round(x3,4),
            x4=round(x4,4), x5=round(x5,4),
            zone=zone, interpretation=interp,
        )
    except Exception as e:
        print(f"[financials] Altman error: {e}")
        return None


def get_market_cap(ticker: str) -> Optional[float]:
    try:
        return yf.Ticker(ticker).info.get("marketCap")
    except Exception:
        return None


def revenue_growth(fd: FinancialData) -> Optional[pd.Series]:
    if fd.revenue is None or len(fd.revenue) < 2:
        return None
    return fd.revenue.pct_change().dropna()


def margin_trends(fd: FinancialData) -> Optional[pd.DataFrame]:
    rows = {}
    if fd.gross_profit is not None and fd.revenue is not None:
        rows["Gross Margin"]     = (fd.gross_profit / fd.revenue.replace(0, np.nan)).dropna()
    if fd.operating_income is not None and fd.revenue is not None:
        rows["Operating Margin"] = (fd.operating_income / fd.revenue.replace(0, np.nan)).dropna()
    if fd.net_income is not None and fd.revenue is not None:
        rows["Net Margin"]       = (fd.net_income / fd.revenue.replace(0, np.nan)).dropna()
    return pd.DataFrame(rows) if rows else None


# ─── MD&A fetcher ─────────────────────────────────────────────────────────────

def fetch_mda_text(cik: str, max_chars: int = 8000) -> tuple:
    # fetches the most recent 10-K filing index, finds the document,
    # extracts the MD&A section via regex
    try:
        # get filing list
        url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
        accnos  = filings.get("accessionNumber", [])

        # find most recent 10-K
        for form, date, accno in zip(forms, dates, accnos):
            if form == "10-K":
                acc_clean = accno.replace("-","")
                idx_url   = (f"https://www.sec.gov/Archives/edgar/full-index/"
                             f"{date[:4]}/QTR{(int(date[5:7])-1)//3+1}/"
                             f"company.idx")
                # use the filing viewer instead
                doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                           f"{int(cik)}/{acc_clean}/{accno}-index.htm")
                return _extract_mda_from_filing(acc_clean, int(cik), date)

        return "", ""
    except Exception as e:
        print(f"[financials] MD&A fetch failed: {e}")
        return "", ""


def _extract_mda_from_filing(acc_clean: str, cik_int: int, date: str) -> tuple:
    try:
        # get the filing index to find the 10-K document filename
        idx_url = (f"https://www.sec.gov/cgi-bin/browse-edgar?"
                   f"action=getcompany&CIK={cik_int}&type=10-K&dateb=&owner=include&count=1&search_text=")
        # simpler: use EDGAR full-text search for the HTM file
        doc_base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/"
        idx_resp = requests.get(doc_base, headers=SEC_HEADERS, timeout=10)
        if idx_resp.status_code != 200:
            return "", date

        # find the main 10-K htm file
        htm_files = re.findall(r'href="([^"]*\.htm)"', idx_resp.text, re.IGNORECASE)
        if not htm_files:
            return "", date

        # pick the largest htm file (usually the 10-K body)
        best = None
        for f in htm_files:
            if "R" not in f.split("/")[-1]:  # skip XBRL viewer files
                best = f
                break

        if best is None:
            return "", date

        full_url = f"https://www.sec.gov{best}" if best.startswith("/") else doc_base + best
        doc_resp = requests.get(full_url, headers=SEC_HEADERS, timeout=20)
        if doc_resp.status_code != 200:
            return "", date

        # strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', doc_resp.text)
        text = re.sub(r'\s+', ' ', text)

        # find MD&A section
        mda_pattern = re.compile(
            r"(?:ITEM\s*7[.\s]*MANAGEMENT[\'S]*\s*DISCUSSION|"
            r"MANAGEMENT[\'S]*\s*DISCUSSION\s*AND\s*ANALYSIS)",
            re.IGNORECASE
        )
        match = mda_pattern.search(text)
        if match:
            start   = match.start()
            excerpt = text[start:start + 8000].strip()
            return excerpt, date

        return "", date

    except Exception as e:
        print(f"[financials] Filing parse error: {e}")
        return "", date


# ─── LLM MD&A analysis ───────────────────────────────────────────────────────

def analyse_mda_with_llm(
    mda_text:    str,
    ticker:      str,
    company_name: str,
    api_key:     str = "",
) -> Optional[MDAAnalysis]:
    # calls Claude claude-sonnet-4-20250514 to summarise the MD&A section
    # api_key: Anthropic API key — get one free at console.anthropic.com
    # if no key, falls back to a regex-based extraction

    if not mda_text or len(mda_text) < 200:
        return None

    excerpt = mda_text[:6000]   # stay well within token limit

    if api_key:
        summary, risks, drivers, tone = _llm_anthropic(excerpt, ticker, company_name, api_key)
    else:
        summary, risks, drivers, tone = _regex_fallback(excerpt)

    return MDAAnalysis(
        ticker        = ticker,
        filing_date   = "",
        mda_excerpt   = excerpt[:800] + "..." if len(excerpt) > 800 else excerpt,
        llm_summary   = summary,
        key_risks     = risks,
        growth_drivers = drivers,
        tone          = tone,
    )


def _llm_anthropic(text: str, ticker: str, company: str, api_key: str) -> tuple:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are a senior equity analyst. Analyse this MD&A section from {company} ({ticker})'s most recent 10-K filing.

MD&A TEXT:
{text}

Provide a structured analysis with exactly these sections:

SUMMARY (2-3 sentences): Overall financial health and performance direction.

KEY RISKS (bullet list of 3-5 specific risks mentioned):
- 
- 

GROWTH DRIVERS (bullet list of 3-5 specific opportunities mentioned):
- 
- 

MANAGEMENT TONE: One word — positive, cautious, or negative — based on language used.

Be specific and reference actual numbers or events from the text where possible."""

        msg = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 800,
            messages   = [{"role": "user", "content": prompt}]
        )
        response = msg.content[0].text

        # parse sections
        summary = _extract_section(response, "SUMMARY", "KEY RISKS")
        risks   = _extract_bullets(response, "KEY RISKS", "GROWTH DRIVERS")
        drivers = _extract_bullets(response, "GROWTH DRIVERS", "MANAGEMENT TONE")
        tone_raw = _extract_section(response, "MANAGEMENT TONE", None).lower().strip()
        tone    = ("positive" if "positive" in tone_raw else
                   "negative" if "negative" in tone_raw else "cautious")

        return summary, risks, drivers, tone

    except ImportError:
        print("[financials] anthropic package not installed — pip install anthropic")
        return _regex_fallback(text)
    except Exception as e:
        print(f"[financials] LLM call failed: {e}")
        return _regex_fallback(text)


def _extract_section(text: str, start_label: str, end_label: str) -> str:
    pattern = rf"{start_label}.*?:(.*?)"
    end_pat = rf"(?={end_label})" if end_label else r"$"
    match   = re.search(pattern + end_pat, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_bullets(text: str, start_label: str, end_label: str) -> list:
    section = _extract_section(text, start_label, end_label)
    bullets = re.findall(r'[-•]\s*(.+)', section)
    return [b.strip() for b in bullets if b.strip()]


def _regex_fallback(text: str) -> tuple:
    # offline fallback: extract risk keywords and growth language without LLM
    risk_keywords    = ["risk", "uncertain", "challenge", "competition", "regulation",
                        "litigation", "inflation", "interest rate", "cybersecurity",
                        "supply chain", "macroeconomic", "geopolit"]
    growth_keywords  = ["growth", "opportunit", "expand", "invest", "innovate",
                        "market share", "launch", "develop", "increase", "improve"]

    sentences = re.split(r'[.!?]', text)
    risks     = [s.strip() for s in sentences
                 if any(k in s.lower() for k in risk_keywords) and len(s.strip()) > 30][:4]
    drivers   = [s.strip() for s in sentences
                 if any(k in s.lower() for k in growth_keywords) and len(s.strip()) > 30][:4]

    neg_words = sum(text.lower().count(k) for k in ["decline","decrease","loss","risk","challenge"])
    pos_words = sum(text.lower().count(k) for k in ["growth","increase","strong","improved","record"])
    tone      = "positive" if pos_words > neg_words * 1.5 else (
                "negative" if neg_words > pos_words * 1.5 else "cautious")

    summary = (f"Regex-based extraction (no LLM key provided). "
               f"Found {len(risks)} risk factors and {len(drivers)} growth drivers. "
               f"Overall tone: {tone}.")

    return summary, risks, drivers, tone