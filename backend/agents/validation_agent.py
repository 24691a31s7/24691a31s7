"""Validation Agent: cross-checks the other agents for conflicting signals
and produces an overall confidence score - the guardrail that stops a
'strong technical, terrible fundamentals' stock from being called a
confident BUY."""
from agents import _validation_logic as logic
from agents.base_agent import BaseAgent


class ValidationAgent(BaseAgent):
    name = "validation_agent"
    goal = "Detect conflicts between agents' signals and compute an overall confidence score."
    output_schema = {"confidence": float, "details": dict}

    def reason(self, technical: dict = None, fundamental: dict = None, sentiment: dict = None,
               data_completeness: float = 1.0, pattern: dict = None, **_) -> dict:
        return logic.validate(technical, fundamental, sentiment, data_completeness, pattern=pattern)


validation_agent = ValidationAgent()


def validate(technical: dict, fundamental: dict, sentiment: dict, data_completeness: float = 1.0,
             pattern: dict = None) -> dict:
    return validation_agent.reason(technical=technical, fundamental=fundamental, sentiment=sentiment,
                                    data_completeness=data_completeness, pattern=pattern)
