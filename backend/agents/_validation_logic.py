"""
Validation Layer: Confidence Agent + Conflict Checker + Data Quality Agent.
Now combines each upstream agent's own confidence score (not just the raw
signal) - review item #6: "Every agent should return confidence".
"""
from schemas import AgentResult


def validate(technical: dict, fundamental: dict, sentiment: dict, data_completeness: float = 1.0,
             pattern: dict = None) -> dict:
    scores = [technical["score"], fundamental["score"], sentiment["score"]]
    agent_confidences = [technical["confidence"], fundamental["confidence"], sentiment["confidence"]]

    # Pattern Recognition Agent is an optional 4th vote (item #1 from the
    # latest request) - included when supplied, weighted like the others
    # but never allowed to dominate, since chart-pattern reliability is the
    # weakest of the four signal types (see pattern_recognition_agent.py).
    if pattern is not None:
        scores.append(pattern["score"])
        agent_confidences.append(pattern["confidence"])

    signs = [1 if s > 0.1 else (-1 if s < -0.1 else 0) for s in scores]
    non_zero = [s for s in signs if s != 0]
    agreement = (non_zero.count(max(set(non_zero), key=non_zero.count)) / len(non_zero)) if non_zero else 0.5
    conflict_detected = len(set(non_zero)) > 1 if non_zero else False

    # Blend: how much each agent trusted its own data, weighted by cross-agent agreement
    avg_agent_confidence = sum(agent_confidences) / len(agent_confidences)
    base_confidence = (avg_agent_confidence / 100) * agreement * data_completeness
    if conflict_detected:
        base_confidence *= 0.75
    # Hard ceiling: no combination of agent agreement should ever be
    # displayed as near-certain. 90% is already an aggressive top end for
    # a heuristic system - see README "On prediction accuracy".
    confidence_pct = round(max(10.0, min(90.0, base_confidence * 100)), 2)

    agent_confidences = {
        "technical": technical["confidence"],
        "fundamental": fundamental["confidence"],
        "sentiment": sentiment["confidence"],
    }
    if pattern is not None:
        agent_confidences["pattern"] = pattern["confidence"]

    return AgentResult(
        agent="validation",
        score=agreement,
        confidence=confidence_pct,
        reason="Signals conflict across agents" if conflict_detected else "Signals broadly agree across agents",
        details={
            "conflict_detected": conflict_detected,
            "data_completeness_pct": round(data_completeness * 100, 1),
            "agent_confidences": agent_confidences,
        },
    ).to_dict()
