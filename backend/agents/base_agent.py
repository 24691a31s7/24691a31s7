"""
Base Agent framework (item #1: "convert agents into true AI agents").

Every agent in this system is a module-level Python function wrapped by a
class that gives it a consistent shape:

  - goal            one-line statement of what this agent is responsible for
  - memory          short-term (per-run) + long-term (cross-run, via the
                     MemoryAgent/DB) context the agent can read and write
  - tools           the callables this agent is allowed to use (keeps the
                     dependency surface explicit and testable)
  - plan()          decides *which* tools/steps are needed for this input
                     (trivial for deterministic agents, meaningful for the
                     RecommendationAgent and PortfolioAgent, which choose
                     different strategies depending on conflicting signals)
  - reason()        the actual analysis logic
  - output_schema   declared shape of what reason() must return - checked
                     at runtime in dev/test so a bug fails loudly instead of
                     silently corrupting a downstream agent's input

This is intentionally NOT a wrapper around an LLM. Most of these agents
(technical, risk, validation, position-sizing) are deterministic and should
stay that way - correctness and auditability matter more than an LLM call
for arithmetic on OHLC data. The two agents where natural-language
reasoning genuinely helps (Sentiment/News and Explanation) call an LLM
*only* when GEMINI_API_KEY is configured, and fall back to rule-based logic
otherwise, so the system always runs end-to-end with zero paid API keys.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from utils.logger import get_logger

log = get_logger("stocks.agent")


@dataclass
class AgentMemory:
    """Per-agent memory. `short_term` is scratch space for a single run
    (cleared each call). `recall()`/`remember()` proxy to the shared
    MemoryAgent (backed by PredictionLog / MarketIntelligence in Postgres)
    for anything that needs to persist across runs or processes."""
    short_term: dict = field(default_factory=dict)

    def remember(self, key: str, value: Any):
        self.short_term[key] = value

    def recall(self, key: str, default=None):
        return self.short_term.get(key, default)

    def clear(self):
        self.short_term.clear()


class BaseAgent(ABC):
    name: str = "base_agent"
    goal: str = "Not specified"
    output_schema: dict[str, type] = {}
    tools: list[Callable] = []

    def __init__(self):
        self.memory = AgentMemory()
        self.log = get_logger(f"stocks.agent.{self.name}")

    def plan(self, **inputs) -> list[str]:
        """Default plan: run `reason` directly. Override for agents that
        pick between multiple strategies based on input signals."""
        return ["reason"]

    @abstractmethod
    def reason(self, **inputs) -> dict:
        """Core analysis logic. Must return a dict matching output_schema."""
        raise NotImplementedError

    def validate_output(self, result: dict) -> dict:
        """Cheap runtime contract check - not full validation, just enough
        to catch 'agent returned the wrong shape' bugs immediately instead
        of three agents downstream."""
        if not self.output_schema:
            return result
        missing = [k for k in self.output_schema if k not in result]
        if missing:
            self.log.warning("%s: reason() output missing keys %s", self.name, missing)
        return result

    def run(self, **inputs) -> dict:
        start = time.perf_counter()
        self.memory.clear()
        steps = self.plan(**inputs)
        self.memory.remember("plan", steps)
        result = self.reason(**inputs)
        result = self.validate_output(result)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.log.debug("%s ran in %.1fms (plan=%s)", self.name, elapsed_ms, steps)
        return result
