"""PPR/KIR metrics for measuring parametric vs contextual knowledge use."""

from .conditions import ConditionPrompts, build_conditions, jog_oracle_context, natural_oracle_context
from .ppr_kir import TurnResult, compute_kir, compute_ppr, compute_turn_metrics
from .scoring import (
    ItemScores,
    generate,
    logprobs_to_absolute_accuracy,
    logprobs_to_accuracy,
    score_logprob,
    score_match,
)

__all__ = [
    "ConditionPrompts",
    "ItemScores",
    "TurnResult",
    "build_conditions",
    "compute_kir",
    "compute_ppr",
    "compute_turn_metrics",
    "generate",
    "jog_oracle_context",
    "logprobs_to_absolute_accuracy",
    "logprobs_to_accuracy",
    "natural_oracle_context",
    "score_logprob",
    "score_match",
]