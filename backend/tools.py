"""
tools.py — Financial tool implementations for the AI Wealth Advisor.

Each tool is a plain Python function that accepts typed parameters and returns
a JSON-serialisable dict.  The TOOLS_SCHEMA list at the bottom describes every
tool to the LLM so it knows what arguments to supply.
"""

import json
import math
import yfinance as yf
from datetime import datetime


# ─────────────────────────────────────────────
#  TOOL 1 — Get live stock price via yfinance
# ─────────────────────────────────────────────

def get_stock_price(ticker: str) -> dict:
    """
    Fetch the latest price and key stats for a stock ticker.
    Returns a dict with price, change %, market cap, P/E ratio, and sector.
    """
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        price     = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        prev_close = info.get("previousClose", None)
        change_pct = (
            round(((price - prev_close) / prev_close) * 100, 2)
            if price != "N/A" and prev_close
            else "N/A"
        )

        return {
            "ticker":       ticker,
            "price_usd":    price,
            "change_pct":   change_pct,
            "market_cap":   info.get("marketCap", "N/A"),
            "pe_ratio":     info.get("trailingPE", "N/A"),
            "sector":       info.get("sector", "N/A"),
            "company_name": info.get("longName", ticker),
            "52w_high":     info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low":      info.get("fiftyTwoWeekLow",  "N/A"),
        }

    except Exception as e:
        return {"error": f"Could not fetch data for {ticker}: {str(e)}"}


# ─────────────────────────────────────────────
#  TOOL 2 — Calculate risk profile
# ─────────────────────────────────────────────

def calculate_risk(budget: float, risk_level: str, age: int = 30) -> dict:
    """
    Derive a quantitative risk profile from the user's stated preferences.

    risk_level: "low" | "medium" | "high"
    Returns recommended equity %, bond %, and max drawdown tolerance.
    """
    try: budget = float(budget)
    except (ValueError, TypeError): budget = 100000.0
    
    try: age = int(age)
    except (ValueError, TypeError): age = 30

    risk_level = str(risk_level).lower().strip()

    # Base allocation rules (classic finance 101)
    allocations = {
        "low":    {"equity": 30, "bonds": 50, "gold": 10, "cash": 10},
        "medium": {"equity": 60, "bonds": 25, "gold": 10, "cash":  5},
        "high":   {"equity": 80, "bonds": 10, "gold":  5, "cash":  5},
    }

    if risk_level not in allocations:
        risk_level = "medium"   # safe default

    alloc = allocations[risk_level]

    # Age-adjustment: reduce equity by 1% per year above 30 (classic rule)
    age_adj = max(0, age - 30)
    alloc["equity"] = max(10, alloc["equity"] - age_adj)
    alloc["bonds"]  = min(80, alloc["bonds"]  + age_adj)

    drawdown_map = {"low": 10, "medium": 25, "high": 40}
    expected_return_map = {"low": "5–7%", "medium": "8–12%", "high": "12–18%"}

    return {
        "risk_level":          risk_level,
        "budget_inr":          budget,
        "age":                 age,
        "allocation":          alloc,          # percentage breakdown
        "max_drawdown_pct":    drawdown_map[risk_level],
        "expected_annual_return": expected_return_map[risk_level],
        "risk_score":          {"low": 3, "medium": 6, "high": 9}[risk_level],
        "investment_horizon":  "3–5 yrs" if risk_level == "low" else ("5–10 yrs" if risk_level == "medium" else "10+ yrs"),
    }


# ─────────────────────────────────────────────
#  TOOL 3 — Suggest a portfolio
# ─────────────────────────────────────────────

def suggest_portfolio(budget: float, risk_level: str, goals: str = "", custom_allocations: str = None) -> dict:
    """
    Agent provides custom_allocations with weights as a JSON string.
    If missing, provides a default portfolio based on risk_level.
    """
    try: budget = float(budget)
    except (ValueError, TypeError): budget = 100000.0
    
    risk_level = str(risk_level).lower().strip()
    
    # 1. Parse custom allocations if provided
    selected_assets = []
    if custom_allocations and isinstance(custom_allocations, str) and custom_allocations.strip() != "null":
        try:
            selected_assets = json.loads(custom_allocations)
            if not isinstance(selected_assets, list):
                selected_assets = []
        except Exception:
            pass

    # 2. Fallback to defaults if no assets provided
    if not selected_assets:
        defaults = {
            "low": [
                {"symbol": "SGB-AUG28-IV.NS", "weight_pct": 50, "why": "Gold Bonds for safety"},
                {"symbol": "ICICILIQUID.NS",   "weight_pct": 30, "why": "Liquid funds for stability"},
                {"symbol": "RELIANCE.NS",      "weight_pct": 20, "why": "Blue-chip exposure"}
            ],
            "medium": [
                {"symbol": "RELIANCE.NS",      "weight_pct": 30, "why": "Stable market leader"},
                {"symbol": "HDFCBANK.NS",      "weight_pct": 30, "why": "Banking backbone"},
                {"symbol": "INFY.NS",          "weight_pct": 20, "why": "IT growth"},
                {"symbol": "GOLD",             "weight_pct": 20, "why": "Inflation hedge"}
            ],
            "high": [
                {"symbol": "NVDA",             "weight_pct": 40, "why": "Aggressive AI growth"},
                {"symbol": "RELIANCE.NS",      "weight_pct": 30, "why": "Core equity strength"},
                {"symbol": "ZOMATO.NS",        "weight_pct": 20, "why": "High growth tech"},
                {"symbol": "TATASTEEL.NS",     "weight_pct": 10, "why": "Cyclical play"}
            ]
        }
        selected_assets = defaults.get(risk_level, defaults["medium"])

    instruments = []
    total_weight = sum([float(a.get("weight_pct", 0)) for a in selected_assets])
    if total_weight == 0: total_weight = 1

    for asset in selected_assets:
        symbol = asset.get("symbol", "CASH").upper().strip()
        raw_weight = float(asset.get("weight_pct", 0))
        why = asset.get("why", "Selected by AI agent")

        weight = (raw_weight / total_weight) * 100

        if symbol in ["CASH", "GOLD"]:
            name = "Cash Savings" if symbol == "CASH" else "Gold (Physical/ETF)"
            itype = "Commodity" if symbol == "GOLD" else "Cash"
        else:
            try:
                t = yf.Ticker(symbol)
                info = t.info
                name = info.get("longName") or info.get("shortName") or symbol
                itype = info.get("quoteType", "Equity").capitalize()
            except Exception:
                name, itype = symbol, "Investment"

        instruments.append({
            "name": name,
            "type": itype,
            "symbol": symbol,
            "weight_pct": round(weight, 2),
            "amount_inr": round(budget * weight / 100, 2),
            "why": why
        })

    return {
        "portfolio_name": "AI Generated Portfolio",
        "total_budget_inr": budget,
        "risk_level": risk_level,
        "instruments": instruments
    }
# ─────────────────────────────────────────────
#  TOOL 4 — Quick market summary
# ─────────────────────────────────────────────

def get_market_summary() -> dict:
    """
    Fetch a snapshot of key benchmark indices.
    Uses history() which is more reliable than info for indices.
    """
    tickers = {
        "Nifty 50":   "^NSEI",
        "S&P 500":    "^GSPC",
        "Sensex":     "^BSESN",
        "Gold (USD)": "GC=F",
        "USD/INR":    "INR=X",
    }

    summary = {}
    for name, symbol in tickers.items():
        try:
            t = yf.Ticker(symbol)
            # Fetch latest 2 days to calculate change if info is missing
            hist = t.history(period="2d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else price
                change = price - prev_price
                pct_change = (change / prev_price) * 100 if prev_price != 0 else 0
                
                summary[symbol] = {
                    "name": name,
                    "price": float(price),
                    "change": float(change),
                    "percent_change": float(pct_change)
                }
            else:
                summary[symbol] = {"name": name, "price": 0.0, "change": 0.0, "percent_change": 0.0, "error": True}
        except Exception:
            summary[symbol] = {"name": name, "price": 0.0, "change": 0.0, "percent_change": 0.0, "error": True}

    return summary


# ─────────────────────────────────────────────
#  TOOL 5 — Trend history for Line Chart
# ─────────────────────────────────────────────

def get_market_trends(symbols: any = "^NSEI,^GSPC") -> dict:
    """
    Fetch the last 30 days of historical percentage returns for multiple symbols.
    Ensures all series are aligned to the same set of dates.
    """
    if isinstance(symbols, str):
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        symbol_list = symbols
    
    INDEX_NAMES = {
        "^NSEI": "Nifty 50 (India)",
        "^BSESN": "BSE Sensex",
        "^GSPC": "S&P 500 (US)",
        "GC=F": "Gold (MCX)",
        "GOLD": "Gold (MCX)",
        "CASH": "Cash / Savings",
    }
    
    # Map 'GOLD' to 'GC=F' for consistency
    symbol_list = ["GC=F" if s.upper() == "GOLD" else s for s in symbol_list]
    
    raw_data = {}
    all_dates = set()
    names = {}
    
    # 1. Fetch raw history for each symbol
    for symbol in symbol_list:
        if symbol == "CASH":
            names[symbol] = INDEX_NAMES["CASH"]
            continue
            
        try:
            t = yf.Ticker(symbol)
            # Use 1mo to match the UI "30-Day" label
            hist = t.history(period="1mo")
            if not hist.empty:
                # Convert index to string dates
                hist.index = hist.index.strftime("%d %b %Y")
                raw_data[symbol] = hist['Close']
                all_dates.update(hist.index.tolist())
                
                # Get readable Name
                if symbol in INDEX_NAMES:
                    names[symbol] = INDEX_NAMES[symbol]
                else:
                    info = t.info
                    name = info.get('longName') or info.get('shortName') or symbol
                    name = name.split(" Limited")[0].split(" Ltd")[0]
                    names[symbol] = name
        except Exception:
            pass

    # 2. Sort dates and align series
    sorted_dates = sorted(list(all_dates), key=lambda d: datetime.strptime(d, "%d %b %Y"))
    
    aligned_trends = {}
    for symbol, series in raw_data.items():
        # Reindex and forward fill to handle different market holidays
        pct_list = []
        first_price = None
        
        # We'll use a simple approach: find the price for each date, or use previous
        last_price = None
        for d in sorted_dates:
            price = series.get(d)
            if price is None:
                price = last_price
            else:
                last_price = price
            
            if first_price is None and price is not None:
                first_price = price
            
            if first_price and price is not None:
                val = ((price / first_price) - 1) * 100
                pct_list.append(round(val, 2))
            else:
                pct_list.append(0.0)
        
        aligned_trends[symbol] = pct_list

    # 3. Handle CASH (flat line)
    if "CASH" in symbol_list:
        aligned_trends["CASH"] = [0.0] * len(sorted_dates)
        names["CASH"] = INDEX_NAMES["CASH"]

    return {
        "dates": sorted_dates,
        "trends": aligned_trends,
        "names": names
    }


# ─────────────────────────────────────────────
#  REGISTRY — maps tool names → functions
# ─────────────────────────────────────────────

TOOLS: dict = {
    "get_stock_price":   get_stock_price,
    "calculate_risk":    calculate_risk,
    "suggest_portfolio": suggest_portfolio,
    "get_market_summary": get_market_summary,
    "get_market_trends":  get_market_trends,
}

# ─────────────────────────────────────────────
#  SCHEMA — tells the LLM what tools exist
# ─────────────────────────────────────────────

TOOLS_SCHEMA: list = [
    {
        "name": "get_stock_price",
        "description": "Fetch the latest stock price, P/E ratio, market cap, 52-week range, and sector for a given ticker symbol.",
        "parameters": {
            "ticker": "string — e.g. 'AAPL', 'RELIANCE.NS', 'TCS.NS'"
        },
    },
    {
        "name": "calculate_risk",
        "description": "Calculate a quantitative risk profile and recommended asset allocation percentages based on budget, risk level, and age.",
        "parameters": {
            "budget":     "float — total investable amount in INR",
            "risk_level": "string — one of 'low', 'medium', or 'high'",
            "age":        "int   — investor age (optional, default 30)",
        },
    },
    {
        "name": "suggest_portfolio",
        "description": "Generate a named, concrete portfolio. Use 'custom_allocations' to specify your own picks as a JSON list of objects: '[{\"symbol\":\"TCS.NS\", \"weight_pct\":20, \"why\":\"...\"}]'.",
        "parameters": {
            "budget":             "float  — total investable amount in INR",
            "risk_level":         "string — one of 'low', 'medium', or 'high'",
            "goals":              "string — investor's goals",
            "custom_allocations": "string — (optional) JSON string of picked instruments",
        },
    },
    {
        "name": "get_market_summary",
        "description": "Fetch a real-time snapshot of Nifty 50, Sensex, S&P 500, Gold, and USD/INR exchange rate. No parameters needed.",
        "parameters": {},
    },
    {
        "name": "get_market_trends",
        "description": "Fetch the 30-day historical trend data for symbols to compare performance. Returns percentage returns starting from 0%.",
        "parameters": {
            "symbols": "string — comma-separated list of symbols (e.g. '^NSEI,RELIANCE.NS,NVDA')"
        },
    },
]
