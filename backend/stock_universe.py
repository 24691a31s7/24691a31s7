"""
Stock universe. Two modes (item #24 - "check every company"):

1. Curated (~90 liquid, large/mid-cap NSE names) - default. Fast, reliable,
   good enough for a demo/hackathon and for anyone without a paid data feed.

2. Full NSE universe (~2,000 symbols) - set USE_FULL_NSE_UNIVERSE=true.
   Pulled from NSE's public equity-list archive and cached to disk for 24h
   so you're not re-downloading it on every restart. NSE occasionally
   blocks non-browser traffic, so this degrades to the curated list
   automatically if the fetch fails - the app never hard-fails because the
   symbol list was unreachable.

`symbol` = NSE trading symbol used by both Groww API and yfinance
(yfinance needs a ".NS" suffix, added at call sites via yf_symbol()).
"""
import csv
import io
import json
import time
from pathlib import Path

from config import settings
from utils.logger import get_logger

log = get_logger("stocks.universe")

CACHE_FILE = Path(__file__).parent / ".nse_universe_cache.json"
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
CACHE_TTL_SECONDS = 24 * 3600

# --- Curated fallback / default universe -----------------------------------
CURATED_UNIVERSE = [
    # --- Financial Services ---
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "sector": "Financial Services"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "sector": "Financial Services"},
    {"symbol": "SBIN", "name": "State Bank of India", "sector": "Financial Services"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "sector": "Financial Services"},
    {"symbol": "AXISBANK", "name": "Axis Bank", "sector": "Financial Services"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "sector": "Financial Services"},
    {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv", "sector": "Financial Services"},
    {"symbol": "INDUSINDBK", "name": "IndusInd Bank", "sector": "Financial Services"},
    {"symbol": "SHRIRAMFIN", "name": "Shriram Finance", "sector": "Financial Services"},

    # --- Information Technology ---
    {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "IT"},
    {"symbol": "INFY", "name": "Infosys", "sector": "IT"},
    {"symbol": "HCLTECH", "name": "HCL Technologies", "sector": "IT"},
    {"symbol": "WIPRO", "name": "Wipro", "sector": "IT"},
    {"symbol": "TECHM", "name": "Tech Mahindra", "sector": "IT"},
    {"symbol": "LTIM", "name": "LTIMindtree", "sector": "IT"},

    # --- Energy & Oil/Gas ---
    {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "Energy"},
    {"symbol": "ONGC", "name": "Oil & Natural Gas Corp", "sector": "Energy"},
    {"symbol": "NTPC", "name": "NTPC", "sector": "Energy"},
    {"symbol": "POWERGRID", "name": "Power Grid Corp", "sector": "Energy"},
    {"symbol": "COALINDIA", "name": "Coal India", "sector": "Energy"},
    {"symbol": "BPCL", "name": "Bharat Petroleum", "sector": "Energy"},

    # --- FMCG & Consumer ---
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "sector": "FMCG"},
    {"symbol": "ITC", "name": "ITC", "sector": "FMCG"},
    {"symbol": "NESTLEIND", "name": "Nestle India", "sector": "FMCG"},
    {"symbol": "BRITANNIA", "name": "Britannia Industries", "sector": "FMCG"},
    {"symbol": "TATACONSUM", "name": "Tata Consumer Products", "sector": "FMCG"},

    # --- Automobiles ---
    {"symbol": "MARUTI", "name": "Maruti Suzuki", "sector": "Automobiles"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors", "sector": "Automobiles"},
    {"symbol": "M&M", "name": "Mahindra & Mahindra", "sector": "Automobiles"},
    {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto", "sector": "Automobiles"},
    {"symbol": "EICHERMOT", "name": "Eicher Motors", "sector": "Automobiles"},
    {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp", "sector": "Automobiles"},

    # --- Pharma & Healthcare ---
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical", "sector": "Pharma"},
    {"symbol": "DRREDDY", "name": "Dr. Reddy's Labs", "sector": "Pharma"},
    {"symbol": "CIPLA", "name": "Cipla", "sector": "Pharma"},
    {"symbol": "APOLLOHOSP", "name": "Apollo Hospitals", "sector": "Pharma"},
    {"symbol": "DIVISLAB", "name": "Divi's Laboratories", "sector": "Pharma"},

    # --- Metals & Mining ---
    {"symbol": "TATASTEEL", "name": "Tata Steel", "sector": "Metals"},
    {"symbol": "JSWSTEEL", "name": "JSW Steel", "sector": "Metals"},
    {"symbol": "HINDALCO", "name": "Hindalco Industries", "sector": "Metals"},
    {"symbol": "VEDL", "name": "Vedanta", "sector": "Metals"},

    # --- Construction & Capital Goods ---
    {"symbol": "LT", "name": "Larsen & Toubro", "sector": "Construction & Capital Goods"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement", "sector": "Construction & Capital Goods"},
    {"symbol": "GRASIM", "name": "Grasim Industries", "sector": "Construction & Capital Goods"},
    {"symbol": "SIEMENS", "name": "Siemens", "sector": "Construction & Capital Goods"},
    {"symbol": "ABB", "name": "ABB India", "sector": "Construction & Capital Goods"},

    # --- Telecom, Services & Consumer Durables ---
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "sector": "Telecom & Services"},
    {"symbol": "ADANIPORTS", "name": "Adani Ports & SEZ", "sector": "Telecom & Services"},
    {"symbol": "TITAN", "name": "Titan Company", "sector": "Telecom & Services"},
    {"symbol": "TRENT", "name": "Trent Ltd.", "sector": "Telecom & Services"},
    {"symbol": "ADANIENT", "name": "Adani Enterprises", "sector": "Telecom & Services"},
]


def _fetch_full_nse_universe() -> list[dict]:
    """Best-effort pull of the full NSE-listed equity roster. Returns []
    on any failure so the caller can fall back cleanly."""
    import requests

    try:
        resp = requests.get(
            NSE_EQUITY_LIST_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StocksApp/1.0)"},
            timeout=10,
        )
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = []
        for r in reader:
            sym = (r.get("SYMBOL") or "").strip()
            name = (r.get("NAME OF COMPANY") or sym).strip()
            if sym:
                rows.append({"symbol": sym, "name": name, "sector": "Unclassified"})
        return rows
    except Exception as exc:  # noqa: BLE001
        log.warning("Full NSE universe fetch failed (%s); using curated list.", exc)
        return []


def _load_universe() -> list[dict]:
    if not settings.USE_FULL_NSE_UNIVERSE:
        return CURATED_UNIVERSE

    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if time.time() - cached.get("fetched_at", 0) < CACHE_TTL_SECONDS and cached.get("stocks"):
                return cached["stocks"][: settings.MAX_UNIVERSE_SIZE]
        except Exception:  # noqa: BLE001
            pass

    fetched = _fetch_full_nse_universe()
    if not fetched:
        return CURATED_UNIVERSE

    try:
        CACHE_FILE.write_text(json.dumps({"fetched_at": time.time(), "stocks": fetched}))
    except Exception:  # noqa: BLE001
        pass
    return fetched[: settings.MAX_UNIVERSE_SIZE]


STOCK_UNIVERSE: list[dict] = _load_universe()
SYMBOL_TO_NAME = {s["symbol"]: s["name"] for s in STOCK_UNIVERSE}
SYMBOL_SET = set(SYMBOL_TO_NAME)


def yf_symbol(nse_symbol: str) -> str:
    """Convert an NSE trading symbol to its yfinance ticker."""
    return f"{nse_symbol}.NS"


def sector_of(symbol: str) -> str:
    return next((s["sector"] for s in STOCK_UNIVERSE if s["symbol"] == symbol), "Unknown")
