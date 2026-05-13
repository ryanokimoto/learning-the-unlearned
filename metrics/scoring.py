"""
Scoring primitives for PPR/KIR evaluation.

Two scoring functions are exposed:

  score_logprob(model, tok, prompt, target) -> float
      The token-length-normalized log-probability of `target` appearing as the
      continuation of `prompt`. This is the *primary* signal for PPR/KIR
      because it is continuous, low-variance across small wording changes,
      and responds smoothly to context. It does not require generation.

  score_match(response, ground_truth) -> bool
      Substring-match accuracy used as a secondary diagnostic and for the
      quadrant counts (parametric-correct, context-rescued, etc.). Requires
      generation. Kept for backwards compatibility with the old metric.

A higher log-prob is "better" (closer to 0). To turn log-probs into accuracies
we use a *paired threshold* across conditions on the same item: an item is
"correct" under condition C if its log-prob under C exceeds the per-item
median log-prob across all conditions evaluated for that item. This makes
accuracy comparable across conditions without needing an external threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class ItemScores:
    """Per-item scores under each evaluation condition."""

    parametric: float           # log P(answer | question)
    oracle: float               # log P(answer | oracle_context, question)
    hybrid: float               # log P(answer | turn_t_context, question)
    persistence: Optional[float] = None  # log P(answer | question) AFTER seeing context_t
    generated_parametric: Optional[str] = None
    generated_hybrid: Optional[str] = None


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def _model_device(model, fallback: str = "cuda") -> torch.device:
    """
    Resolve the device tensors should be moved to.

    With device_map="auto" (multi-GPU sharding), the model's embedding layer
    may live on a non-zero GPU; passing inputs on cuda:0 produces a
    cross-device error. We probe the actual embedding device when available,
    falling back to the first parameter, then to the user-supplied string.
    """
    # Path 1: HF accelerate annotates layers with hf_device_map
    dmap = getattr(model, "hf_device_map", None)
    if dmap:
        # Prefer the embedding entry; otherwise take any value.
        for key in ("transformer.wte", "model.embed_tokens", "wte", "embed_tokens"):
            if key in dmap:
                return torch.device(dmap[key])
        # Any device entry will do — inputs flow through whatever owns the embeddings.
        return torch.device(next(iter(dmap.values())))
    # Path 2: single-device model
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


# ---------------------------------------------------------------------------
# Log-probability scoring (the primary signal)
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_logprob(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device: str = "cuda",
    length_normalize: bool = True,
) -> float:
    """
    Return log P(target | prompt) under the model.

    The score is computed by teacher-forcing: concatenate prompt+target,
    run a single forward pass, and sum the log-probabilities of the target
    tokens. Token-length-normalized by default so that answers of different
    lengths are comparable.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        prompt: Conditioning text (no trailing separator needed; a space is
            inserted before target).
        target: The string whose probability we want to measure.
        device: Torch device string.
        length_normalize: Divide total log-prob by number of target tokens.

    Returns:
        A float; higher = more likely. Length-normalized log-prob is in
        roughly [-15, 0] for a healthy model.
    """
    # Encode prompt and full sequence separately so we know where target starts.
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    # Use leading space so target tokens align with mid-sentence tokenization.
    full_text = prompt + (" " if not prompt.endswith((" ", "\n")) else "") + target
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    target_start = prompt_ids.shape[0]
    target_len = full_ids.shape[0] - target_start
    if target_len <= 0:
        # Tokenizer merged prompt and target — fall back to encoding target alone.
        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        full_ids = torch.cat([prompt_ids, target_ids])
        target_start = prompt_ids.shape[0]
        target_len = target_ids.shape[0]

    input_ids = full_ids.unsqueeze(0).to(_model_device(model, device))
    logits = model(input_ids).logits[0]  # [seq_len, vocab]

    # Standard causal-LM shift: logits at position i predict token at i+1.
    # We want log P of tokens [target_start ... target_start+target_len-1],
    # which are predicted by logits at positions [target_start-1 ... target_start+target_len-2].
    pred_logits = logits[target_start - 1 : target_start - 1 + target_len]
    target_ids = full_ids[target_start : target_start + target_len].to(logits.device)

    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    total = token_logps.sum().item()
    return total / target_len if length_normalize else total


# ---------------------------------------------------------------------------
# Substring-match scoring (secondary diagnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> str:
    """Greedy generation for the response side of substring-match scoring."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(_model_device(model, device))
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def score_match(response: str, ground_truth: str) -> bool:
    """Case-insensitive substring containment. Cheap, noisy, but interpretable."""
    return ground_truth.strip().lower() in response.strip().lower()


# ---------------------------------------------------------------------------
# Turning per-item log-probs into accuracies
# ---------------------------------------------------------------------------

def logprobs_to_accuracy(
    condition_scores: List[float],
    reference_scores: List[float],
    margin: float = 0.0,
) -> float:
    """
    Convert a list of (condition, reference) log-prob pairs into an accuracy.

    An item is counted as "correct" under the condition if its log-prob
    exceeds the reference log-prob by at least `margin` nats. The reference
    is typically the *oracle* condition: an item is "answerable" under C
    iff C does almost as well as oracle. This sidesteps the need for an
    absolute log-prob threshold, which would depend on model and tokenizer.

    Args:
        condition_scores: log-probs under the condition being evaluated.
        reference_scores: log-probs under the reference condition (e.g. oracle).
        margin: how many nats below reference still counts as correct.
                Default 0.0 = condition must match or exceed reference.
                A small positive margin (e.g. 0.5) gives the condition slack.

    Returns:
        Fraction in [0, 1].
    """
    if len(condition_scores) != len(reference_scores):
        raise ValueError("condition and reference must have same length")
    if not condition_scores:
        return 0.0
    correct = sum(
        1 for c, r in zip(condition_scores, reference_scores)
        if c >= r - margin
    )
    return correct / len(condition_scores)


def logprobs_to_absolute_accuracy(
    scores: List[float],
    threshold: float = -2.0,
) -> float:
    """
    Convert log-probs to accuracy using an absolute threshold.

    Useful when no oracle reference is available, or for sanity checks.
    Default threshold -2.0 nats/token corresponds to roughly 1/e^2 ≈ 13.5%
    probability per token, a generous bar for "the model knows the answer."

    Args:
        scores: length-normalized log-probs.
        threshold: items with log-prob >= threshold count as correct.

    Returns:
        Fraction in [0, 1].
    """
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)