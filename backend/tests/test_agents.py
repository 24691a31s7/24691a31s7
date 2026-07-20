"""
Unit tests for the deterministic, non-network parts of the pipeline.
Run with:  pytest tests/ -v   (from inside backend/, with the venv activated)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import recommendation_agent, risk_agent, technical_agent, validation_agent  # noqa: E402
from agents.pattern_recognition_agent import pattern_recognition_agent  # noqa: E402
from agents.portfolio_agent import portfolio_agent  # noqa: E402


def _fake_uptrend_df(n=300):
    rng = np.random.default_rng(42)
    price = 100 + np.cumsum(rng.normal(0.3, 1.0, n))
    return pd.DataFrame({
        "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
        "volume": rng.integers(100000, 500000, n),
    })


def _fake_downtrend_df(n=300):
    rng = np.random.default_rng(7)
    price = 200 - np.cumsum(rng.normal(0.3, 1.0, n))
    price = np.clip(price, 1, None)
    return pd.DataFrame({
        "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
        "volume": rng.integers(100000, 500000, n),
    })


class TestTechnicalAgent:
    def test_returns_structured_result(self):
        result = technical_agent.analyze(_fake_uptrend_df())
        assert set(["agent", "score", "confidence", "reason", "details"]).issubset(result.keys())
        assert -1 <= result["score"] <= 1
        assert 0 <= result["confidence"] <= 100

    def test_insufficient_history_is_low_confidence(self):
        result = technical_agent.analyze(pd.DataFrame({"close": [1, 2, 3]}))
        assert result["confidence"] < 30

    def test_uptrend_scores_higher_than_downtrend(self):
        up = technical_agent.analyze(_fake_uptrend_df())
        down = technical_agent.analyze(_fake_downtrend_df())
        assert up["score"] > down["score"]


class TestRiskAgent:
    def test_risk_label_in_expected_set(self):
        result = risk_agent.assess(_fake_uptrend_df(), [])
        assert result["risk_label"] in ("Low", "Medium", "High")

    def test_flagged_events_increase_risk(self):
        low = risk_agent.assess(_fake_uptrend_df(), [])
        high = risk_agent.assess(_fake_uptrend_df(), ["fraud probe", "CEO resigns", "SEBI penalty"])
        assert high["aggregate_risk_score"] >= low["aggregate_risk_score"]


class TestRecommendationAgent:
    def test_strong_bullish_signal_yields_buy(self):
        decision = recommendation_agent.decide(
            expected_return_pct=1.5, profit_probability_pct=80,
            risk_label="Low", confidence_pct=85, conflict_detected=False,
        )
        assert decision == "BUY"

    def test_low_confidence_yields_watchlist(self):
        decision = recommendation_agent.decide(
            expected_return_pct=1.5, profit_probability_pct=80,
            risk_label="Low", confidence_pct=25, conflict_detected=False,
        )
        assert decision == "WATCHLIST"

    def test_position_sizing_respects_max_allocation(self):
        sizing = recommendation_agent.size_position(
            entry_price=1000, risk_label="Low", capital=100000, risk_per_trade_pct=1.0,
        )
        assert sizing["estimated_investment_inr"] <= 100000 * 0.10 + 1000
        assert sizing["quantity"] >= 0

    def test_zero_price_returns_zero_quantity(self):
        sizing = recommendation_agent.size_position(entry_price=0, risk_label="Low")
        assert sizing["quantity"] == 0


class TestValidationAgent:
    def test_conflicting_signals_reduce_confidence(self):
        tech = {"score": 0.8, "confidence": 80}
        fund = {"score": -0.8, "confidence": 80}
        sent = {"score": 0.0, "confidence": 50}
        result = validation_agent.validate(tech, fund, sent)
        assert result["details"]["conflict_detected"] is True


class TestPortfolioAgent:
    def test_no_buys_returns_empty_portfolio(self):
        result = portfolio_agent.reason(analyses=[{"recommendation": "HOLD"}], capital=100000)
        assert result["positions"] == []

    def test_respects_sector_cap(self):
        analyses = [
            {"symbol": f"S{i}", "name": f"S{i}", "sector": "IT", "recommendation": "BUY",
             "last_price": 100, "expected_return_pct": 2.0, "confidence_pct": 80, "risk_label": "Low"}
            for i in range(5)
        ]
        result = portfolio_agent.reason(analyses=analyses, capital=100000, max_positions=5)
        it_total = result["sector_breakdown"].get("IT", 0)
        assert it_total <= 100000 * (portfolio_agent.MAX_PER_SECTOR_PCT / 100) + 1


class TestPatternRecognitionAgent:
    def test_returns_structured_result(self):
        result = pattern_recognition_agent.reason(price_history=_fake_uptrend_df())
        assert set(["score", "confidence", "patterns", "details"]).issubset(result.keys())
        assert -1 <= result["score"] <= 1
        assert 0 <= result["confidence"] <= 90  # never near-certain, by design

    def test_insufficient_history_returns_neutral(self):
        result = pattern_recognition_agent.reason(price_history=pd.DataFrame({"close": [1, 2, 3]}))
        assert result["score"] == 0.0
        assert result["patterns"] == []

    def test_double_top_detected_in_synthetic_series(self):
        # Two equal peaks with a meaningful trough between them.
        close = np.concatenate([
            np.linspace(100, 140, 20), np.linspace(140, 110, 15),
            np.linspace(110, 139, 15), np.linspace(139, 90, 20),
        ])
        df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                            "close": close, "volume": np.full(len(close), 200000)})
        result = pattern_recognition_agent.reason(price_history=df)
        names = [p["name"] for p in result["patterns"]]
        assert "Double Top" in names


class TestUniverseFilters:
    def test_price_over_cap_excluded(self):
        from universe_filters import passes_universe_filters
        ok, reason = passes_universe_filters(3000, 252 * 6, [], "TESTCO")
        assert ok is False
        assert "price" in reason

    def test_short_history_excluded(self):
        from universe_filters import passes_universe_filters
        ok, reason = passes_universe_filters(500, 252 * 2, [], "TESTCO")
        assert ok is False
        assert "history" in reason

    def test_litigation_keyword_excluded(self):
        from universe_filters import passes_universe_filters
        ok, reason = passes_universe_filters(500, 252 * 6, ["SEBI probe into accounting fraud"], "TESTCO")
        assert ok is False

    def test_clean_stock_passes(self):
        from universe_filters import passes_universe_filters
        ok, reason = passes_universe_filters(500, 252 * 6, ["strong quarterly results"], "TESTCO")
        assert ok is True


class TestGrowwAuthCooldown:
    def test_repeated_rate_limited_auth_does_not_retry_within_cooldown(self):
        """Regression test for the bug where every stock re-triggered a full
        Groww auth attempt after a rate-limited failure, cascading into
        dozens of repeat 'rate limit exceeded' errors within one scan."""
        import groww_client

        groww_client._groww_client = None
        groww_client._groww_available = False
        groww_client._auth_cooldown_until = 0.0

        original_key = groww_client.settings.GROWW_API_KEY
        try:
            groww_client.settings.GROWW_API_KEY = "fake-key-no-real-auth"
            # First call attempts auth, fails (no growwapi credentials work
            # in tests), and should start a cooldown.
            groww_client._init_groww()
            cooldown_after_first = groww_client._auth_cooldown_until
            assert cooldown_after_first > 0

            # A second call within the cooldown window must NOT attempt
            # auth again (cooldown timestamp should be unchanged).
            groww_client._init_groww()
            assert groww_client._auth_cooldown_until == cooldown_after_first
        finally:
            groww_client.settings.GROWW_API_KEY = original_key
            groww_client._groww_client = None
            groww_client._auth_cooldown_until = 0.0


class TestRateLimiters:
    def test_provider_semaphores_exist_and_are_bounded(self):
        from config import settings
        from utils.rate_limit import groww_limiter, newsapi_limiter, yfinance_limiter

        assert yfinance_limiter._value == settings.YFINANCE_MAX_CONCURRENT
        assert newsapi_limiter._value == settings.NEWSAPI_MAX_CONCURRENT
        assert groww_limiter._value == settings.GROWW_MAX_CONCURRENT


class TestWithRetry:
    def test_succeeds_without_retry_when_first_call_works(self):
        from utils.http_session import with_retry

        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        assert with_retry(fn, attempts=3, base_delay=0.01) == "ok"
        assert calls["n"] == 1

    def test_retries_then_succeeds(self):
        from utils.http_session import with_retry

        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 Too Many Requests")
            return "ok"

        assert with_retry(fn, attempts=5, base_delay=0.01) == "ok"
        assert calls["n"] == 3

    def test_raises_after_exhausting_attempts(self):
        from utils.http_session import with_retry

        def fn():
            raise RuntimeError("boom")

        try:
            with_retry(fn, attempts=2, base_delay=0.01)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
