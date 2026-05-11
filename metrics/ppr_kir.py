"""
PPR (Parametric Proxy Rate) and KIR (Knowledge Integration Rate) metrics.

Based on the evaluation framework from:
  "Memory in Large Language Models: Mechanisms, Evaluation and Evolution"
  arXiv:2509.18868

Applied to the JOG (Jogging the Memory of Unlearned LLMs) setting from:
  "Jogging the Memory of Unlearned LLMs Through Targeted Relearning Attacks"
  arXiv:2406.13356  (Hu et al., ICLR 2025)

Definitions
-----------
Let Q = {(q_i, a_i)} be a set of question–answer pairs and C_t = {c_i^t} be
the conversation context accumulated through turn t (empty at turn 0).

For each question q_i at turn t, generate two responses:
  - r_i^P  = model(q_i)              # parametric only, no context
  - r_i^C  = model(C_t^i || q_i)     # with accumulated conversation context

Define correct(r, a) = 1 if the answer a appears in the response r, else 0.

PPR (Parametric Proxy Rate):
  PPR = (1/N) * Σ correct(r_i^P, a_i)
  Measures the fraction of questions answered correctly from parametric
  knowledge alone. Constant across turns (independent of context).
  High PPR → model retains strong parametric knowledge.
  Low PPR after unlearning → unlearning succeeded at suppressing weights.

KIR (Knowledge Integration Rate):
  KIR_t = (1/N) * Σ [ (1 - correct(r_i^P, a_i)) * correct(r_i^C, a_i) ]
  Measures the fraction of questions where context at turn t correctly
  compensates for missing/wrong parametric knowledge.
  Rises across turns as conversation context accumulates.
  High KIR → model effectively integrates contextual knowledge.
  KIR > 0 with low PPR → context is jogging suppressed parametric memory.

Together, PPR and KIR decompose model accuracy into four quadrants:
  - Parametric-Correct  (PPR): correct from weights, context irrelevant
  - Context-Integrated  (KIR): wrong from weights, corrected by context
  - Both-Correct        : correct from both sources
  - Neither-Correct     : wrong regardless of context (hard questions)
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class TurnResult:
    """Metric results for a single conversation turn."""

    turn: int
    n: int

    # Raw counts
    parametric_correct: int   # correct(r^P, a) = 1
    contextual_correct: int   # correct(r^C, a) = 1
    both_correct: int         # correct in both conditions
    kir_count: int            # parametric wrong, contextual correct

    # Derived rates
    ppr: float                # parametric_correct / n
    kir: float                # kir_count / n
    contextual_accuracy: float  # contextual_correct / n

    def __str__(self) -> str:
        return (
            f"Turn {self.turn:>2} | N={self.n} | "
            f"PPR={self.ppr:.3f}  KIR={self.kir:.3f}  "
            f"CtxAcc={self.contextual_accuracy:.3f}"
        )


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def _generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _default_match(response: str, ground_truth: str) -> bool:
    return ground_truth.strip().lower() in response.strip().lower()


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_ppr(
    model,
    tokenizer,
    questions: List[str],
    ground_truths: List[str],
    max_new_tokens: int = 64,
    device: str = "cuda",
    answer_fn: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """
    Parametric Proxy Rate: fraction of questions answered correctly from
    parametric knowledge alone (no conversation context).

    This is the baseline measure of what the model knows in its weights.
    In a relearning/unlearning experiment, a low PPR confirms that the
    unlearning procedure suppressed weight-level recall.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        questions: Raw question strings (no context prepended).
        ground_truths: Expected answer for each question.
        max_new_tokens: Generation budget.
        device: Torch device string.
        answer_fn: Optional custom match(response, ground_truth) -> bool.

    Returns:
        PPR as a float in [0, 1].
    """
    match = answer_fn or _default_match
    correct = 0
    for q, gt in zip(questions, ground_truths):
        response = _generate(model, tokenizer, q, max_new_tokens, device)
        correct += int(match(response, gt))
    return correct / len(questions)


def compute_kir(
    model,
    tokenizer,
    questions: List[str],
    contexts: List[str],
    ground_truths: List[str],
    max_new_tokens: int = 64,
    device: str = "cuda",
    answer_fn: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """
    Knowledge Integration Rate: fraction of questions where the provided
    conversation context correctly overrides wrong parametric knowledge.

    KIR = P(parametric wrong ∧ contextual correct)

    A rising KIR across conversation turns indicates the model is
    successfully integrating new information to compensate for gaps in
    (or suppression of) its parametric knowledge.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        questions: Raw question strings.
        contexts: Conversation context for each question (empty = no context).
        ground_truths: Expected answer for each question.
        max_new_tokens: Generation budget.
        device: Torch device string.
        answer_fn: Optional custom match(response, ground_truth) -> bool.

    Returns:
        KIR as a float in [0, 1].
    """
    match = answer_fn or _default_match
    kir_count = 0
    for q, ctx, gt in zip(questions, contexts, ground_truths):
        param_resp = _generate(model, tokenizer, q, max_new_tokens, device)
        ctx_prompt = f"{ctx}\n{q}".strip() if ctx else q
        ctx_resp = _generate(model, tokenizer, ctx_prompt, max_new_tokens, device)
        if not match(param_resp, gt) and match(ctx_resp, gt):
            kir_count += 1
    return kir_count / len(questions)


def compute_turn_metrics(
    model,
    tokenizer,
    questions: List[str],
    contexts: List[str],
    ground_truths: List[str],
    turn: int,
    max_new_tokens: int = 64,
    device: str = "cuda",
    answer_fn: Optional[Callable[[str, str], bool]] = None,
) -> TurnResult:
    """
    Compute PPR and KIR jointly for a single conversation turn, avoiding
    redundant parametric-only inference by caching the no-context generation.

    At turn 0 contexts should all be empty strings; contexts grow on
    subsequent turns as conversation history is accumulated.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        questions: Raw question strings.
        contexts: Accumulated conversation context per question at this turn.
        ground_truths: Expected answer for each question.
        turn: Turn index (0 = parametric-only baseline).
        max_new_tokens: Generation budget.
        device: Torch device string.
        answer_fn: Optional custom match(response, ground_truth) -> bool.

    Returns:
        TurnResult with PPR, KIR, and raw counts.
    """
    match = answer_fn or _default_match
    n = len(questions)
    param_correct = ctx_correct = both_correct = kir_count = 0

    for q, ctx, gt in zip(questions, contexts, ground_truths):
        param_resp = _generate(model, tokenizer, q, max_new_tokens, device)
        ctx_prompt = f"{ctx}\n{q}".strip() if ctx else q
        ctx_resp = _generate(model, tokenizer, ctx_prompt, max_new_tokens, device)

        p_ok = match(param_resp, gt)
        c_ok = match(ctx_resp, gt)

        param_correct += int(p_ok)
        ctx_correct += int(c_ok)
        both_correct += int(p_ok and c_ok)
        kir_count += int(not p_ok and c_ok)

    return TurnResult(
        turn=turn,
        n=n,
        parametric_correct=param_correct,
        contextual_correct=ctx_correct,
        both_correct=both_correct,
        kir_count=kir_count,
        ppr=param_correct / n,
        kir=kir_count / n,
        contextual_accuracy=ctx_correct / n,
    )
