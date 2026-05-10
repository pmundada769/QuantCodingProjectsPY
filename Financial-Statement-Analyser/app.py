#app.py — Financial Statement Analyser
# SEC EDGAR 10-K/10-Q + yfinance valuation + DCF + peer comparison
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import yfinance as yf
from financials import (
    fetch_financials, piotroski_fscore, altman_zscore,
    get_market_cap, revenue_growth, margin_trends,
    fetch_mda_text, analyse_mda_with_llm,
)

st.set_page_config(page_title="Financial Analyser", page_icon="📑", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#080C10;color:#C8D8E0;}
h1{font-family:'IBM Plex Mono',monospace!important;color:#E8C468!important;}
h2,h3{color:#2A3840!important;font-weight:400!important;}
[data-testid="metric-container"]{background:#0E1418;border:1px solid #141E24;border-radius:4px;padding:12px 16px;}
[data-testid="metric-container"] label{font-family:'IBM Plex Mono',monospace!important;font-size:.58rem!important;color:#1A2830!important;text-transform:uppercase;letter-spacing:.1em;}
[data-testid="metric-container"] [data-testid="metric-value"]{font-family:'IBM Plex Mono',monospace!important;font-size:1.0rem!important;color:#E8C468!important;}
[data-testid="stSidebar"]{background:#050810;border-right:1px solid #141E24;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.68rem;background:#0E1418;border:1px solid #141E24;color:#1A2830;border-radius:3px;}
.stTabs [aria-selected="true"]{background:#141E24!important;border-color:#E8C468!important;color:#E8C468!important;}
hr{border-color:#141E24!important;}
</style>
""", unsafe_allow_html=True)

GOLD="#E8C468"; TEAL="#2ECC9A"; CORAL="#E74C3C"; BLUE="#3498DB"; MUTED="#2A3840"
BG="#080C10"; GRID="#141E24"; TEXT="#C8D8E0"; PURPLE="#9B59B6"

def _base(**kw):
    d = dict(plot_bgcolor=BG, paper_bgcolor=BG,
             font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
             xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
             yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
             margin=dict(l=60,r=20,t=50,b=40),
             legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID))
    d.update(kw)
    return d

def fmt_M(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    try:
        v = float(v)
        if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
        if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
        return f"${v:,.0f}"
    except: return "N/A"

def safe_last(series, default=None):
    if series is None: return default
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) > 0 else default

def safe_pct_change(series):
    if series is None: return None
    s = series.dropna()
    if len(s) < 2: return None
    return s.pct_change().dropna()

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📑 Financial Analyser")
    st.markdown("---")
    tickers_input = st.text_area("Tickers (one per line)", height=140,
                                  value="AAPL\nMSFT\nNVDA\nMETA\nGOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]
    primary = st.selectbox("Primary ticker", tickers, index=0)
    include_quarterly = st.checkbox("Include quarterly (10-Q) data", value=True)

    st.markdown("---")
    st.markdown("#### LLM MD&A (optional)")
    anthropic_key = st.text_input("Anthropic API key", type="password",
        help="Free at console.anthropic.com — blank = regex fallback")
    run_mda = st.checkbox("Fetch & analyse MD&A", value=False)

    st.markdown("---")
    st.markdown("#### DCF Assumptions")
    dcf_growth1 = st.slider("Revenue growth yr 1-5 (%)", -10, 50, 15)
    dcf_growth2 = st.slider("Revenue growth yr 6-10 (%)", -10, 30, 8)
    dcf_terminal = st.slider("Terminal growth (%)", 0, 5, 3)
    dcf_wacc    = st.slider("WACC (%)", 5, 20, 10)
    dcf_margin  = st.slider("Target FCF margin (%)", 5, 40, 20)

    st.caption("Data: SEC EDGAR + Yahoo Finance\nAll free, no API key needed for financials")


# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_edgar(ticker, quarterly):
    fd = fetch_financials(ticker)
    if fd is None: return None, None, None
    mcap = get_market_cap(ticker)
    fs = piotroski_fscore(fd)
    zs = altman_zscore(fd, market_cap=mcap)
    return fd, fs, zs

@st.cache_data(ttl=300)
def load_yf(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        # quarterly financials
        q_income = t.quarterly_income_stmt
        q_balance= t.quarterly_balance_sheet
        q_cash   = t.quarterly_cashflow
        a_income = t.income_stmt
        a_balance= t.balance_sheet
        a_cash   = t.cashflow
        return info, q_income, q_balance, q_cash, a_income, a_balance, a_cash
    except Exception as e:
        return {}, None, None, None, None, None, None

@st.cache_data(ttl=3600)
def load_mda(cik, ticker, company, api_key):
    text, date = fetch_mda_text(cik)
    if not text: return None
    result = analyse_mda_with_llm(text, ticker, company, api_key)
    if result: result.filing_date = date
    return result

with st.spinner(f"Loading {primary}..."):
    fd, fscore, zscore = load_edgar(primary, include_quarterly)
    info, q_inc, q_bal, q_cf, a_inc, a_bal, a_cf = load_yf(primary)

if fd is None and (q_inc is None or (hasattr(q_inc,'empty') and q_inc.empty)):
    st.error(f"Could not load data for {primary}.")
    st.stop()

company_name = (fd.company_name if fd else info.get("longName", primary))

mda_result = None
if run_mda and fd and fd.cik:
    with st.spinner("Fetching MD&A..."):
        mda_result = load_mda(fd.cik, primary, company_name, anthropic_key)

# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
st.markdown("# 📑 Financial Statement Analyser")
st.markdown(f"`{company_name}` · `{primary}` · `SEC EDGAR + Yahoo Finance`")
st.markdown("---")

# key metrics row
mcap  = info.get("marketCap")
pe    = info.get("trailingPE")
fpe   = info.get("forwardPE")
ps    = info.get("priceToSalesTrailing12Months")
pb    = info.get("priceToBook")
ev_eb = info.get("enterpriseToEbitda")
roe   = info.get("returnOnEquity")
debt_eq = info.get("debtToEquity")
curr_price = info.get("currentPrice") or info.get("regularMarketPrice")

c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("Market Cap",    fmt_M(mcap))
c2.metric("Price",         f"${curr_price:.2f}" if curr_price else "N/A")
c3.metric("P/E (TTM)",     f"{pe:.1f}x" if pe else "N/A")
c4.metric("Fwd P/E",       f"{fpe:.1f}x" if fpe else "N/A")
c5.metric("P/S",           f"{ps:.1f}x" if ps else "N/A")
c6.metric("P/B",           f"{pb:.1f}x" if pb else "N/A")
c7.metric("EV/EBITDA",     f"{ev_eb:.1f}x" if ev_eb else "N/A")
c8.metric("ROE",           f"{roe*100:.1f}%" if roe else "N/A")

f_val = fscore.total_score if fscore else "N/A"
z_val = f"{zscore.z_score:.2f}" if zscore else "N/A"
z_zone= zscore.zone if zscore else "N/A"
z_icon= {"Safe":"🟢","Grey":"🟡","Distress":"🔴"}.get(z_zone,"⚪")

d1,d2,d3,d4,d5,d6 = st.columns(6)
d1.metric("Piotroski F",   f"{f_val}/9")
d2.metric("Altman Z",      f"{z_val}  {z_icon}")
d3.metric("Debt/Equity",   f"{debt_eq:.1f}%" if debt_eq else "N/A")
d4.metric("Sector",        info.get("sector","N/A"))
d5.metric("Industry",      info.get("industry","N/A")[:22] if info.get("industry") else "N/A")
d6.metric("52W Range",
    f"${info.get('fiftyTwoWeekLow',0):.2f}–${info.get('fiftyTwoWeekHigh',0):.2f}"
    if info.get("fiftyTwoWeekLow") else "N/A")

st.markdown("---")

# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tabs = st.tabs([
    "📈  Income Statement",
    "🏦  Balance Sheet",
    "💵  Cash Flow",
    "📊  Valuation Multiples",
    "🔮  DCF Model",
    "⚖️  Peer Comparison",
    "🏆  Piotroski + Altman",
    "🤖  MD&A Analysis",
])

# ── global key counter to guarantee unique widget keys ──
import itertools as _itertools
_key_counter = _itertools.count()
def _uk(prefix="k"):
    return f"{prefix}_{next(_key_counter)}"

# ── helper: render a yfinance statement ───────────────
def render_statement(annual_df, quarterly_df, title, key_rows=None):
    view = st.radio("View", ["Annual","Quarterly"], horizontal=True, key=_uk("view"))
    df   = annual_df if view == "Annual" else quarterly_df
    if df is None or (hasattr(df,'empty') and df.empty):
        st.warning(f"No {view.lower()} data available for {primary}.")
        return
    df = df.copy()
    try:
        display = df.copy()
        for col in display.columns:
            display[col] = display[col].apply(
                lambda v: fmt_M(v) if pd.notna(v) else "N/A")
        st.dataframe(display, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render table: {e}")

def chart_metric_over_time(annual_df, quarterly_df, row_name, title, color=TEAL):
    view = st.radio("View", ["Annual","Quarterly"], horizontal=True, key=_uk("chart"))
    df   = annual_df if view == "Annual" else quarterly_df
    if df is None or (hasattr(df,'empty') and df.empty): return
    if row_name not in df.index: return
    series = df.loc[row_name].dropna()
    if len(series) == 0: return
    vals   = [float(v)/1e9 for v in series.values]
    dates  = [str(c)[:10] for c in series.index]
    fig = go.Figure(go.Bar(x=dates, y=vals,
        marker_color=[color if v>=0 else CORAL for v in vals],
        text=[f"${v:.2f}B" for v in vals], textposition="outside",
        textfont=dict(color=TEXT)))
    fig.update_layout(**_base(height=300, title=f"{title} ($B)"))
    fig.update_yaxes(ticksuffix="B")
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 1: Income Statement ────────────────────────────
with tabs[0]:
    st.markdown("### Income Statement")

    # chart key metrics
    if a_inc is not None and not a_inc.empty:
        col1, col2 = st.columns(2)
        with col1:
            for row in ["Total Revenue","Gross Profit"]:
                if row in a_inc.index:
                    chart_metric_over_time(a_inc, q_inc, row, row, TEAL)
        with col2:
            for row in ["Net Income","Operating Income"]:
                if row in a_inc.index:
                    chart_metric_over_time(a_inc, q_inc, row, row, GOLD)

        st.markdown("#### Full Table")
        key_inc = ["Total Revenue","Gross Profit","Operating Income",
                   "EBITDA","Net Income","EPS Diluted","Basic EPS"]
        render_statement(a_inc, q_inc, "Income", key_inc)
    else:
        st.info("No income statement data from Yahoo Finance for this ticker.")

    # margin trends from EDGAR
    if fd:
        mt = margin_trends(fd)
        if mt is not None and len(mt) > 0:
            st.markdown("#### Margin Trends (from SEC EDGAR)")
            fig_m = go.Figure()
            colours = [GOLD, TEAL, BLUE]
            for i, col in enumerate(mt.columns):
                fig_m.add_trace(go.Scatter(
                    x=[str(d.year) for d in mt.index], y=mt[col]*100,
                    mode="lines+markers", name=col,
                    line=dict(color=colours[i%3], width=2)))
            fig_m.update_layout(**_base(height=300, title="Margin Trends (%)"))
            fig_m.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_m, use_container_width=True)


# ── TAB 2: Balance Sheet ───────────────────────────────
with tabs[1]:
    st.markdown("### Balance Sheet")
    if a_bal is not None and not a_bal.empty:
        col1, col2 = st.columns(2)
        with col1:
            for row in ["Total Assets","Total Liabilities Net Minority Interest"]:
                if row in a_bal.index:
                    chart_metric_over_time(a_bal, q_bal, row, row.split()[0]+" "+row.split()[1], BLUE)
        with col2:
            for row in ["Stockholders Equity","Total Debt"]:
                if row in a_bal.index:
                    chart_metric_over_time(a_bal, q_bal, row, row, GOLD)

        key_bal = ["Total Assets","Total Liabilities Net Minority Interest",
                   "Stockholders Equity","Total Debt","Cash And Cash Equivalents",
                   "Net Receivables","Inventory","Long Term Debt",
                   "Current Assets","Current Liabilities"]
        render_statement(a_bal, q_bal, "Balance", key_bal)
    else:
        st.info("No balance sheet data from Yahoo Finance.")

    # quick ratios
    try:
        ca = q_bal.loc["Current Assets"].iloc[0] if q_bal is not None and "Current Assets" in q_bal.index else None
        cl = q_bal.loc["Current Liabilities"].iloc[0] if q_bal is not None and "Current Liabilities" in q_bal.index else None
        td = q_bal.loc["Total Debt"].iloc[0] if q_bal is not None and "Total Debt" in q_bal.index else None
        eq = q_bal.loc["Stockholders Equity"].iloc[0] if q_bal is not None and "Stockholders Equity" in q_bal.index else None
        if ca and cl and float(cl) > 0:
            cr = float(ca)/float(cl)
            dr = float(td)/float(eq) if td and eq and float(eq) != 0 else None
            r1,r2 = st.columns(2)
            r1.metric("Current Ratio (latest Q)", f"{cr:.2f}x",
                help=">1.5 healthy, <1 potential liquidity issues")
            if dr: r2.metric("Debt/Equity (latest Q)", f"{dr:.2f}x")
    except Exception:
        pass


# ── TAB 3: Cash Flow ──────────────────────────────────
with tabs[2]:
    st.markdown("### Cash Flow Statement")
    if a_cf is not None and not a_cf.empty:
        col1, col2 = st.columns(2)
        with col1:
            for row in ["Operating Cash Flow","Free Cash Flow"]:
                if row in a_cf.index:
                    chart_metric_over_time(a_cf, q_cf, row, row, TEAL)
        with col2:
            for row in ["Capital Expenditure","Investing Cash Flow"]:
                if row in a_cf.index:
                    chart_metric_over_time(a_cf, q_cf, row, row, GOLD)

        key_cf = ["Operating Cash Flow","Investing Cash Flow","Financing Cash Flow",
                  "Free Cash Flow","Capital Expenditure",
                  "Repurchase Of Capital Stock","Cash Dividends Paid"]
        render_statement(a_cf, q_cf, "CashFlow", key_cf)
    else:
        st.info("No cash flow data from Yahoo Finance.")


# ── TAB 4: Valuation Multiples ────────────────────────
with tabs[3]:
    st.markdown("### Valuation Multiples")
    st.caption("Current snapshot vs historical context and sector peers")

    # valuation grid
    multiples = {
        "P/E (TTM)":        (pe,    "< 15 cheap, 15-25 fair, > 25 expensive"),
        "Forward P/E":      (fpe,   "< 12 cheap, market avg ~18"),
        "P/S (TTM)":        (ps,    "< 1 deep value, 1-3 fair, > 10 richly valued"),
        "P/B":              (pb,    "< 1 below book value, > 3 premium"),
        "EV/EBITDA":        (ev_eb, "< 8 cheap, 8-15 fair, > 20 expensive"),
        "EV/Revenue":       (info.get("enterpriseToRevenue"), "sector dependent"),
        "PEG Ratio":        (info.get("pegRatio"),  "< 1 undervalued vs growth, > 2 expensive"),
        "Dividend Yield":   (info.get("dividendYield"), "income investors: > 2% attractive"),
        "ROE":              (info.get("returnOnEquity"), "> 15% good, > 25% excellent"),
        "ROA":              (info.get("returnOnAssets"), "> 5% good"),
        "Profit Margin":    (info.get("profitMargins"), "> 10% good"),
        "Gross Margin":     (info.get("grossMargins"), "> 40% high quality"),
        "Operating Margin": (info.get("operatingMargins"), "> 15% good"),
        "Revenue Growth":   (info.get("revenueGrowth"), "> 10% growth company"),
        "Earnings Growth":  (info.get("earningsGrowth"), "> 10% growth company"),
        "Debt/Equity":      (info.get("debtToEquity"), "< 1 conservative, > 2 leveraged"),
        "Quick Ratio":      (info.get("quickRatio"), "> 1 healthy"),
        "Current Ratio":    (info.get("currentRatio"), "> 1.5 healthy"),
    }

    rows = []
    for name, (val, context) in multiples.items():
        if val is not None:
            fmt_val = f"{val*100:.1f}%" if name in ["Dividend Yield","ROE","ROA","Profit Margin","Gross Margin","Operating Margin","Revenue Growth","Earnings Growth"] else f"{val:.2f}x" if isinstance(val,float) else str(val)
            rows.append({"Metric": name, "Value": fmt_val, "Context": context})

    if rows:
        mult_df = pd.DataFrame(rows)
        st.dataframe(mult_df, hide_index=True, use_container_width=True)

    # EPS trend
    try:
        if a_inc is not None and "Diluted EPS" in a_inc.index:
            eps_row  = a_inc.loc["Diluted EPS"].dropna()
            eps_vals = [float(v) for v in eps_row.values]
            eps_dates= [str(c)[:7] for c in eps_row.index]
            fig_eps  = go.Figure(go.Bar(x=eps_dates, y=eps_vals,
                marker_color=[TEAL if v>=0 else CORAL for v in eps_vals],
                text=[f"${v:.2f}" for v in eps_vals], textposition="outside"))
            fig_eps.update_layout(**_base(height=300, title=f"{primary} — Diluted EPS (Annual)"))
            st.plotly_chart(fig_eps, use_container_width=True)
    except Exception:
        pass


# ── TAB 5: DCF Model ──────────────────────────────────
with tabs[4]:
    st.markdown("### Discounted Cash Flow (DCF) Valuation")
    st.caption("Simple DCF using revenue projections and target FCF margin. Adjust assumptions in sidebar.")

    # get latest revenue and shares
    rev_latest = None
    shares     = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")

    if a_inc is not None and not a_inc.empty and "Total Revenue" in a_inc.index:
        rev_series = a_inc.loc["Total Revenue"].dropna()
        if len(rev_series) > 0:
            rev_latest = float(rev_series.iloc[0])   # most recent annual

    if rev_latest is None and fd and fd.revenue is not None:
        rv = fd.revenue.dropna()
        if len(rv) > 0:
            rev_latest = float(rv.iloc[-1])

    if rev_latest is None:
        st.warning("No revenue data available for DCF. Company may be pre-revenue.")
    elif shares is None or shares == 0:
        st.warning("Cannot calculate per-share DCF — shares outstanding not available.")
    else:
        g1 = dcf_growth1 / 100
        g2 = dcf_growth2 / 100
        gt = dcf_terminal / 100
        w  = dcf_wacc / 100
        fm = dcf_margin / 100

        # project 10 years of FCF
        years_     = list(range(1, 11))
        rev_proj   = []
        fcf_proj   = []
        r = rev_latest
        for yr in years_:
            g   = g1 if yr <= 5 else g2
            r   = r * (1 + g)
            rev_proj.append(r)
            fcf_proj.append(r * fm)

        # discount FCF
        pv_fcfs   = [fcf / (1+w)**yr for yr, fcf in zip(years_, fcf_proj)]
        terminal  = fcf_proj[-1] * (1+gt) / (w - gt)
        pv_terminal = terminal / (1+w)**10
        intrinsic = (sum(pv_fcfs) + pv_terminal) / shares

        # sensitivity: WACC × terminal growth
        waccs     = [0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
        tgrowths  = [0.01, 0.02, 0.03, 0.04, 0.05]
        sens = pd.DataFrame(index=[f"{int(w_*100)}% WACC" for w_ in waccs],
                             columns=[f"{int(g_*100)}% TG" for g_ in tgrowths])
        for w_ in waccs:
            for g_ in tgrowths:
                pv = sum(fcf / (1+w_)**yr for yr, fcf in zip(years_, fcf_proj))
                tv = fcf_proj[-1]*(1+g_)/(w_-g_) / (1+w_)**10 if w_ > g_ else 0
                sens.loc[f"{int(w_*100)}% WACC", f"{int(g_*100)}% TG"] = round((pv+tv)/shares, 2)

        m1, m2, m3 = st.columns(3)
        m1.metric("Intrinsic Value / Share", f"${intrinsic:.2f}")
        m2.metric("Current Price",           f"${curr_price:.2f}" if curr_price else "N/A")
        if curr_price:
            upside = (intrinsic/curr_price - 1)*100
            m3.metric("Upside / Downside",   f"{upside:+.1f}%",
                delta=f"{'Undervalued' if upside > 0 else 'Overvalued'}")

        # projection chart
        fig_dcf = go.Figure()
        fig_dcf.add_trace(go.Bar(x=[f"Y{y}" for y in years_],
            y=[r/1e9 for r in rev_proj], name="Projected Revenue ($B)",
            marker_color=BLUE, opacity=0.6))
        fig_dcf.add_trace(go.Bar(x=[f"Y{y}" for y in years_],
            y=[f/1e9 for f in fcf_proj], name="Projected FCF ($B)",
            marker_color=TEAL))
        fig_dcf.update_layout(**_base(height=320, title="10-Year Revenue & FCF Projection ($B)",
                                       barmode="overlay"))
        st.plotly_chart(fig_dcf, use_container_width=True)

        # sensitivity table
        st.markdown("#### Sensitivity Analysis — Intrinsic Value per Share ($)")
        st.caption("Rows = WACC, Columns = Terminal Growth Rate")
        st.dataframe(sens.style.background_gradient(cmap="RdYlGn", axis=None),
                     use_container_width=True)


# ── TAB 6: Peer Comparison ────────────────────────────
with tabs[5]:
    st.markdown("### Peer Comparison")
    if st.button("▶  Run Peer Comparison"):
        peer_rows = []
        prog = st.progress(0)
        for i, t in enumerate(tickers):
            try:
                info_p = yf.Ticker(t).info or {}
                fd_p, fs_p, zs_p = load_edgar(t, False)
                rv_growth = "N/A"
                if fd_p and fd_p.revenue is not None:
                    rv = fd_p.revenue.dropna()
                    if len(rv) >= 2:
                        rv_growth = f"{float(rv.pct_change().dropna().iloc[-1])*100:+.1f}%"

                peer_rows.append({
                    "Ticker":       t,
                    "Company":      info_p.get("shortName", t)[:20],
                    "Mkt Cap":      fmt_M(info_p.get("marketCap")),
                    "P/E":          f"{info_p.get('trailingPE',0):.1f}x" if info_p.get("trailingPE") else "N/A",
                    "Fwd P/E":      f"{info_p.get('forwardPE',0):.1f}x"  if info_p.get("forwardPE")  else "N/A",
                    "P/S":          f"{info_p.get('priceToSalesTrailing12Months',0):.1f}x" if info_p.get("priceToSalesTrailing12Months") else "N/A",
                    "EV/EBITDA":    f"{info_p.get('enterpriseToEbitda',0):.1f}x" if info_p.get("enterpriseToEbitda") else "N/A",
                    "Gross Margin": f"{info_p.get('grossMargins',0)*100:.1f}%" if info_p.get("grossMargins") else "N/A",
                    "ROE":          f"{info_p.get('returnOnEquity',0)*100:.1f}%" if info_p.get("returnOnEquity") else "N/A",
                    "Rev Growth":   rv_growth,
                    "Piotroski":    fs_p.total_score if fs_p else "N/A",
                    "Altman Z":     f"{zs_p.z_score:.2f}" if zs_p else "N/A",
                })
            except Exception:
                peer_rows.append({"Ticker": t, "Company": t})
            prog.progress((i+1)/len(tickers))

        if peer_rows:
            peer_df = pd.DataFrame(peer_rows)
            st.dataframe(peer_df, hide_index=True, use_container_width=True)

            # visual comparison of P/E
            numeric_pe = []
            for row in peer_rows:
                try:
                    numeric_pe.append((row["Ticker"], float(row["P/E"].replace("x","").replace("N/A","0"))))
                except: pass
            if numeric_pe:
                labels, vals = zip(*[(t, v) for t, v in numeric_pe if v > 0])
                fig_peer = go.Figure(go.Bar(x=list(labels), y=list(vals),
                    marker_color=[GOLD if t==primary else BLUE for t in labels],
                    text=[f"{v:.1f}x" for v in vals], textposition="outside"))
                fig_peer.update_layout(**_base(height=300, title="P/E Ratio — Peer Comparison"))
                st.plotly_chart(fig_peer, use_container_width=True)


# ── TAB 7: Piotroski + Altman ─────────────────────────
with tabs[6]:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Piotroski F-Score")
        if fscore:
            score_col = TEAL if fscore.total_score >= 7 else (CORAL if fscore.total_score <= 2 else GOLD)
            st.markdown(f"#### **{fscore.total_score}/9** — {fscore.interpretation}")
            cats = {
                "Profitability": {
                    "ROA positive":      fscore.roa_positive,
                    "OCF positive":      fscore.ocf_positive,
                    "ROA improved":      fscore.roa_increase,
                    "Low accruals":      fscore.accruals_low,
                },
                "Leverage": {
                    "Debt fell":         fscore.leverage_dec,
                    "Current ratio up":  fscore.liquidity_inc,
                    "No dilution":       fscore.no_dilution,
                },
                "Efficiency": {
                    "Gross margin up":   fscore.margin_inc,
                    "Asset turnover up": fscore.turnover_inc,
                },
            }
            for cat, signals in cats.items():
                st.markdown(f"**{cat}**")
                cols = st.columns(len(signals))
                for col, (name, val) in zip(cols, signals.items()):
                    col.metric(name, f"{'✅' if val else '❌'} {val}/1")
        else:
            st.warning("Insufficient EDGAR data for Piotroski score.")

    with col_r:
        st.markdown("### Altman Z-Score")
        if zscore:
            zone_col = TEAL if zscore.zone=="Safe" else (CORAL if zscore.zone=="Distress" else GOLD)
            z_icon2  = {"Safe":"🟢","Grey":"🟡","Distress":"🔴"}.get(zscore.zone,"⚪")
            st.markdown(f"#### **{zscore.z_score:.4f}** — {z_icon2} {zscore.zone}")
            st.markdown(f"*{zscore.interpretation}*")

            fig_z = go.Figure(go.Indicator(
                mode="gauge+number", value=zscore.z_score,
                title=dict(text="Altman Z-Score", font=dict(color=TEXT)),
                gauge=dict(
                    axis=dict(range=[0,5], tickcolor=TEXT, tickfont=dict(color=TEXT)),
                    bar=dict(color=zone_col),
                    steps=[
                        dict(range=[0,1.81],    color="rgba(231,76,60,.2)"),
                        dict(range=[1.81,2.99], color="rgba(42,56,64,.2)"),
                        dict(range=[2.99,5],    color="rgba(46,204,154,.2)"),
                    ],
                    threshold=dict(line=dict(color=GOLD,width=3), value=2.99),
                ),
                number=dict(font=dict(color=GOLD)),
            ))
            fig_z.update_layout(plot_bgcolor=BG, paper_bgcolor=BG,
                                 font=dict(color=TEXT), height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig_z, use_container_width=True)
            for label, val in [("X1 Working Cap/Assets", zscore.x1),
                                ("X2 Retained E/Assets",  zscore.x2),
                                ("X3 EBIT/Assets",        zscore.x3),
                                ("X4 MktCap/Liab",        zscore.x4),
                                ("X5 Revenue/Assets",     zscore.x5)]:
                st.markdown(f"`{label}` = **{val:.4f}**")
        else:
            st.warning("Insufficient EDGAR data for Altman score.")


# ── TAB 8: MD&A ───────────────────────────────────────
with tabs[7]:
    st.markdown("### MD&A Analysis — Management Discussion & Analysis")
    if not run_mda:
        st.info("Enable 'Fetch & analyse MD&A' in the sidebar to run this tab.")
        st.caption("Fetches the actual MD&A section from the most recent 10-K SEC filing, then uses Claude (optional) to extract risks, growth drivers, and assess management tone. Works without Claude key using regex extraction.")
    elif mda_result is None:
        st.warning("Could not fetch MD&A. This can happen with very small companies or non-standard filings.")
    else:
        tone_col = TEAL if mda_result.tone=="positive" else (CORAL if mda_result.tone=="negative" else GOLD)
        st.markdown(f"**Filing:** {mda_result.filing_date or 'Most recent 10-K'} &nbsp;|&nbsp; "
                    f"**Tone:** <span style='color:{tone_col};font-family:IBM Plex Mono;'>{mda_result.tone.upper()}</span>",
                    unsafe_allow_html=True)
        col_r, col_g = st.columns(2)
        with col_r:
            st.markdown("#### ⚠️ Key Risks")
            for risk in mda_result.key_risks:
                st.markdown(f"- {risk}")
        with col_g:
            st.markdown("#### 🚀 Growth Drivers")
            for d in mda_result.growth_drivers:
                st.markdown(f"- {d}")
        st.markdown("#### Summary")
        st.markdown(mda_result.llm_summary)
        with st.expander("Raw MD&A excerpt"):
            st.text(mda_result.mda_excerpt)
        if not anthropic_key:
            st.caption("ℹ️ Regex extraction used — add Anthropic API key for Claude analysis")