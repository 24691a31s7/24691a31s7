"""
Portfolio Agent (item #11): the recommendation layer works per-stock;
this agent takes a *set* of per-stock analyses and turns them into a
diversified portfolio - sector caps, risk-weighted capital allocation,
and an expected blended return. Still plain, auditable arithmetic (no
black box) so every allocation is traceable back to the inputs.
"""
from agents.base_agent import BaseAgent
from config import settings


class PortfolioAgent(BaseAgent):
    name = "portfolio_agent"
    goal = "Turn a set of per-stock analyses into a diversified, risk-weighted portfolio allocation."
    output_schema = {"positions": list, "expected_portfolio_return_pct": float}
    MAX_PER_SECTOR_PCT = 35.0  # no single sector gets more than this share of capital

    def plan(self, **inputs) -> list[str]:
        candidates = inputs.get("analyses") or []
        buys = [a for a in candidates if a.get("recommendation") == "BUY"]
        return ["allocate_buys"] if buys else ["no_qualifying_positions"]

    def reason(self, analyses: list[dict] = None, capital: float = None, max_positions: int = 8, **_) -> dict:
        analyses = analyses or []
        capital = capital or settings.DEFAULT_CAPITAL_INR

        candidates = [a for a in analyses if a.get("recommendation") == "BUY"]
        if not candidates:
            return {"positions": [], "expected_portfolio_return_pct": 0.0,
                     "note": "No BUY-rated candidates in the supplied set."}

        # Rank by a blend of expected return, confidence, and (inverted) risk
        risk_weight = {"Low": 1.0, "Medium": 0.75, "High": 0.5}

        def score(a):
            return (
                a.get("expected_return_pct", 0) * 0.4
                + a.get("confidence_pct", 0) * 0.3
                + risk_weight.get(a.get("risk_label"), 0.6) * 100 * 0.3
            )

        candidates.sort(key=score, reverse=True)
        candidates = candidates[:max_positions]

        # Diversification: cap capital per sector, then allocate remaining
        # weight proportional to score within the sector cap.
        sector_capital_used: dict[str, float] = {}
        total_score = sum(max(score(a), 0.01) for a in candidates)
        positions = []
        allocated = 0.0

        for a in candidates:
            sector = a.get("sector", "Unknown")
            weight = max(score(a), 0.01) / total_score
            raw_alloc = capital * weight
            sector_cap = capital * (self.MAX_PER_SECTOR_PCT / 100)
            used = sector_capital_used.get(sector, 0.0)
            room = max(sector_cap - used, 0.0)
            alloc = min(raw_alloc, room)
            if alloc < a.get("last_price", 0):
                continue  # can't buy even one share within the cap

            qty = int(alloc // a["last_price"]) if a.get("last_price") else 0
            invested = round(qty * a["last_price"], 2)
            if qty <= 0:
                continue

            sector_capital_used[sector] = used + invested
            allocated += invested
            positions.append({
                "symbol": a["symbol"], "name": a.get("name"), "sector": sector,
                "quantity": qty, "invested_inr": invested,
                "weight_pct": round(invested / capital * 100, 2),
                "expected_return_pct": a.get("expected_return_pct"),
                "risk_label": a.get("risk_label"), "confidence_pct": a.get("confidence_pct"),
            })

        expected_portfolio_return = (
            sum(p["invested_inr"] * (p["expected_return_pct"] or 0) for p in positions) / allocated
            if allocated else 0.0
        )

        return {
            "positions": positions,
            "capital": capital,
            "capital_allocated_inr": round(allocated, 2),
            "capital_unallocated_inr": round(capital - allocated, 2),
            "expected_portfolio_return_pct": round(expected_portfolio_return, 2),
            "sector_breakdown": {s: round(v, 2) for s, v in sector_capital_used.items()},
        }


portfolio_agent = PortfolioAgent()


def build_portfolio(analyses: list[dict], capital: float, max_positions: int = 8) -> dict:
    return portfolio_agent.reason(analyses=analyses, capital=capital, max_positions=max_positions)
