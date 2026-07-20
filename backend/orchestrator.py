"""
Orchestrator (Root Agent): the only module allowed to call other agents
directly (items #1-#2). Independent I/O-bound work (live quote, price
history, fundamentals, news) runs concurrently via asyncio instead of a
sequential chain, then CPU-light downstream agents (technical, prediction,
risk, validation, recommendation, explanation) run in dependency order.

    Tier 1 (parallel, I/O-bound):  MarketData | Fundamental | Sentiment
    Tier 2 (depends on Tier 1):    Technical -> Prediction, Risk
    Tier 3 (depends on Tier 2):    Validation -> Recommendation
    Tier 4 (depends on Tier 3):    Explanation, Memory (log the prediction)

This single-stock path is what /api/stock/{symbol} calls. run_full_scan_async
fans this out across the whole tracked universe with a concurrency cap
(item #24), and writes results into MarketIntelligence so future requests
are served from precomputed data instead of a cold start.
"""
import asyncio
import time
from datetime import datetime

from agents import (
    explanation_agent,
    fundamental_agent,
    market_data_agent,
    memory_agent,
    pattern_recognition_agent,
    portfolio_agent,
    prediction_agent,
    recommendation_agent,
    risk_agent,
    sentiment_agent,
    technical_agent,
    validation_agent,
)
from config import settings
from stock_universe import STOCK_UNIVERSE, SYMBOL_TO_NAME, sector_of
from utils.logger import get_logger

log = get_logger("stocks.orchestrator")

SCAN_CONCURRENCY = settings.SCAN_CONCURRENCY


# ---------------------------------------------------------------------------
# Single-symbol analysis: the core agent pipeline
# ---------------------------------------------------------------------------
async def analyze_symbol_async(symbol: str, capital: float = None, risk_per_trade_pct: float = None,
                                record_prediction: bool = True) -> dict:
    symbol = symbol.upper()
    name = SYMBOL_TO_NAME.get(symbol, symbol)

    # --- Tier 1: independent I/O-bound calls run concurrently ---
    quote_task = asyncio.to_thread(market_data_agent.get_live_quote, symbol)
    history_task = asyncio.to_thread(market_data_agent.get_price_history, symbol)
    fund_task = asyncio.to_thread(fundamental_agent.analyze, symbol)
    sent_task = asyncio.to_thread(sentiment_agent.get_news_and_sentiment, symbol)

    quote, history, fund, sent = await asyncio.gather(quote_task, history_task, fund_task, sent_task)

    last_price = quote.get("last_price") or (float(history["close"].iloc[-1]) if not history.empty else None)

    # --- Tier 2: depends on Tier 1's price history ---
    tech = technical_agent.analyze(history)
    pattern = pattern_recognition_agent.analyze(history)
    pred = prediction_agent.forecast_return(
        technical_score=tech["score"], fundamental_score=fund["score"],
        sentiment_score=sent["score"], price_history=history, pattern_score=pattern["score"],
    )
    flagged_events = sent.get("details", {}).get("flagged_risk_events", [])
    risk = risk_agent.assess(history, flagged_events)

    # --- Tier 3: validation + recommendation ---
    data_completeness = sum([bool(quote), not history.empty, bool(fund.get("details")), True]) / 4
    valid = validation_agent.validate(tech, fund, sent, data_completeness, pattern=pattern)
    valid_details = valid["details"]

    decision = recommendation_agent.decide(
        expected_return_pct=pred["expected_return_pct"], profit_probability_pct=pred["profit_probability_pct"],
        risk_label=risk["risk_label"], confidence_pct=valid["confidence"],
        conflict_detected=valid_details["conflict_detected"],
    )
    sizing = recommendation_agent.size_position(
        entry_price=last_price or 0, risk_label=risk["risk_label"],
        capital=capital, risk_per_trade_pct=risk_per_trade_pct,
    ) if last_price else {"quantity": 0, "estimated_investment_inr": 0, "stop_loss_price": 0, "target_price": 0}

    # --- Tier 4: explanation + memory ---
    reasons = explanation_agent.explain(decision, tech, fund, sent, risk, pred, pattern=pattern)

    # Predicted close price in rupee terms (request #8) - directly derived
    # from expected_return_pct, so it's exactly as reliable as that number
    # (i.e. a plausible-range estimate, NOT a guarantee - see validation
    # agent's confidence cap and the README).
    prev_close = quote.get("ohlc", {}).get("close") if quote.get("ohlc") else None
    today_open = quote.get("ohlc", {}).get("open") if quote.get("ohlc") else None
    predicted_close_price = (
        round(last_price * (1 + pred["expected_return_pct"] / 100), 2) if last_price else None
    )
    predicted_close_range = (
        [round(last_price * (1 + pred["range_low_pct"] / 100), 2),
         round(last_price * (1 + pred["range_high_pct"] / 100), 2)] if last_price else None
    )

    if record_prediction and last_price:
        try:
            memory_agent.record_prediction(
                symbol=symbol, price_at_prediction=last_price,
                predicted_return_pct=pred["expected_return_pct"], predicted_direction=decision,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not log prediction for %s: %s", symbol, exc)

    return {
        "symbol": symbol, "name": name, "sector": sector_of(symbol),
        "last_price": last_price, "day_change_pct": quote.get("day_change_perc"),
        "previous_close_price": prev_close, "today_open_price": today_open,
        "predicted_close_price": predicted_close_price, "predicted_close_range": predicted_close_range,
        "expected_return_pct": pred["expected_return_pct"], "range_low_pct": pred["range_low_pct"],
        "range_high_pct": pred["range_high_pct"], "profit_probability_pct": pred["profit_probability_pct"],
        "loss_probability_pct": pred["loss_probability_pct"], "risk_label": risk["risk_label"],
        "confidence_pct": valid["confidence"], "recommendation": decision, "reasons": reasons,
        "patterns_detected": pattern.get("patterns", []),
        "flagged_risk_events": flagged_events,
        "suggested_quantity": sizing["quantity"], "estimated_investment_inr": sizing["estimated_investment_inr"],
        "stop_loss_price": sizing["stop_loss_price"], "target_price": sizing["target_price"],
        "scanned_at": datetime.utcnow().isoformat(),
        "agent_breakdown": [tech, fund, sent, risk, pattern, valid],  # structured, per-agent transparency
    }


def analyze_symbol(symbol: str, capital: float = None, risk_per_trade_pct: float = None) -> dict:
    """Sync entrypoint for callers outside an event loop (e.g. Celery tasks)."""
    return asyncio.run(analyze_symbol_async(symbol, capital, risk_per_trade_pct))


# ---------------------------------------------------------------------------
# Agent router (item #18): run only the agent(s) actually needed instead of
# the full pipeline, for lightweight endpoints.
# ---------------------------------------------------------------------------
async def run_single_agent(agent_name: str, symbol: str) -> dict:
    symbol = symbol.upper()
    history = await asyncio.to_thread(market_data_agent.get_price_history, symbol)

    if agent_name == "technical":
        return technical_agent.analyze(history)
    if agent_name == "fundamental":
        return await asyncio.to_thread(fundamental_agent.analyze, symbol)
    if agent_name == "sentiment":
        return await asyncio.to_thread(sentiment_agent.get_news_and_sentiment, symbol)
    if agent_name == "risk":
        return risk_agent.assess(history, [])
    if agent_name == "pattern":
        return pattern_recognition_agent.analyze(history)
    raise ValueError(f"Unknown agent: {agent_name}")


# ---------------------------------------------------------------------------
# Full-universe scan -> continuous market intelligence layer (item #24)
# Applies universe filters (request #4) and is mode-aware (request #6):
# pre-market scans lean on technicals+fundamentals for TODAY's setup,
# post-market scans lean on fresh news/sentiment for TOMORROW's setup.
# ---------------------------------------------------------------------------
async def run_full_scan_async(top_n: int = None, universe: list[dict] = None,
                               apply_filters: bool = True, mode: str = "open") -> list[dict]:
    top_n = top_n or settings.TOP_N_RECOMMENDATIONS
    universe = universe or STOCK_UNIVERSE
    sem = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def bounded(stock):
        async with sem:
            try:
                return await analyze_symbol_async(stock["symbol"], record_prediction=False)
            except Exception as exc:  # noqa: BLE001
                log.error("Analysis failed for %s: %s", stock["symbol"], exc)
                return None

    start = time.time()
    results = await asyncio.gather(*(bounded(s) for s in universe))
    results = [r for r in results if r is not None]

    if apply_filters:
        results = await _apply_universe_filters(results, sem)

    log.info("Scan of %d/%d stocks completed in %.1fs (mode=%s)", len(results), len(universe),
              time.time() - start, mode)

    risk_weight = {"Low": 1.0, "Medium": 0.7, "High": 0.4}

    if mode == "post_market":
        # Tomorrow's picks: weight sentiment/news heavier than technicals,
        # since a fresh after-hours headline matters more than today's
        # close-of-day technical snapshot (request #6).
        def rank_key(r):
            sentiment_boost = next((a["score"] for a in r["agent_breakdown"] if a.get("agent") == "sentiment"), 0)
            return (
                r["expected_return_pct"] * 0.25 + r["profit_probability_pct"] * 0.25
                + r["confidence_pct"] * 0.15 + sentiment_boost * 100 * 0.25
                + risk_weight.get(r["risk_label"], 0.5) * 20 * 0.10
            )
    else:
        def rank_key(r):
            return (
                r["expected_return_pct"] * 0.4 + r["profit_probability_pct"] * 0.3
                + r["confidence_pct"] * 0.2 + risk_weight.get(r["risk_label"], 0.5) * 20 * 0.1
            )

    for r in results:
        r["rank_score"] = round(rank_key(r), 4)
    results.sort(key=lambda r: r["rank_score"], reverse=True)
    return results[:top_n] if top_n else results


async def _apply_universe_filters(results: list[dict], sem: asyncio.Semaphore) -> list[dict]:
    from universe_filters import passes_universe_filters

    async def check(r):
        async with sem:
            try:
                hist = await asyncio.to_thread(market_data_agent.get_price_history, r["symbol"], "max")
                ok, reason = passes_universe_filters(
                    last_price=r["last_price"], history_len_trading_days=len(hist),
                    flagged_risk_events=r.get("flagged_risk_events", []), symbol=r["symbol"],
                )
                if not ok:
                    log.debug("Filtered out %s: %s", r["symbol"], reason)
                return r if ok else None
            except Exception as exc:  # noqa: BLE001
                log.warning("Universe filter check failed for %s: %s", r["symbol"], exc)
                return r  # fail-open: don't silently drop a stock just because the age check errored

    checked = await asyncio.gather(*(check(r) for r in results))
    return [r for r in checked if r is not None]


def run_full_scan(top_n: int = None, mode: str = "open", apply_filters: bool = True) -> list[dict]:
    """Sync entrypoint for the scheduler / Celery tasks."""
    return asyncio.run(run_full_scan_async(top_n, mode=mode, apply_filters=apply_filters))


# ---------------------------------------------------------------------------
# Portfolio construction (item #11)
# ---------------------------------------------------------------------------
async def build_portfolio_async(symbols: list[str], capital: float, max_positions: int = 8,
                                 risk_per_trade_pct: float = None) -> dict:
    sem = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def bounded(sym):
        async with sem:
            try:
                return await analyze_symbol_async(sym, capital=capital, risk_per_trade_pct=risk_per_trade_pct,
                                                    record_prediction=False)
            except Exception as exc:  # noqa: BLE001
                log.error("Analysis failed for %s: %s", sym, exc)
                return None

    analyses = await asyncio.gather(*(bounded(s) for s in symbols))
    analyses = [a for a in analyses if a is not None]
    return portfolio_agent.build_portfolio(analyses, capital=capital, max_positions=max_positions)
