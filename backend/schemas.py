"""
Structured output contract every analysis agent (technical, fundamental,
sentiment) returns, instead of a bare float. The Root/Orchestrator Agent is
the only thing that reads these and combines them - agents never call each
other directly.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class AgentResult:
    agent: str
    score: float          # -1..1, direction + strength of the signal
    confidence: float      # 0..100, how much data backed this score
    reason: str             # one-line human-readable summary
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
