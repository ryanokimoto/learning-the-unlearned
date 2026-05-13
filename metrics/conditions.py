"""
Prompt construction for the four PPR/KIR evaluation conditions.

For each (question, gold_answer, turn_t_context) triple we build prompts for:

  1. parametric  — question alone. Tests weight-stored knowledge.
  2. oracle      — question + a context guaranteed to contain the answer.
                   This is the *upper bound* the model can reach when context
                   is maximally helpful. It is the denominator in PPR.
  3. hybrid      — question + the actual turn-t context being studied.
                   This is what we are measuring evolution over.
  4. persistence — question alone, but only meaningful when paired with a
                   prior exposure to context_t (see notes below). The prompt
                   itself is the same as parametric; the *difference from
                   parametric* across turns is what reveals soft relearning.

Persistence note
----------------
The persistence probe cannot be expressed as a single prompt — it requires
running the model on context_t first, then re-querying without context, and
checking whether the parametric score has risen relative to baseline. In a
stateless HuggingFace forward pass there is no carryover, so persistence is
only meaningful for models that have been *finetuned* on context_t (the
actual JOG relearning attack scenario) or for stateful agents. For the
synthetic JOG experiments we keep `persistence_prompt` identical to the
parametric prompt and compare its score across turns of a relearning
trajectory rather than within a single inference call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConditionPrompts:
    """The four prompts to evaluate for a single QA item at one turn."""

    parametric: str
    oracle: str
    hybrid: str
    persistence: str  # equal to parametric for stateless eval; see module docstring


def build_conditions(
    question: str,
    gold_answer: str,
    hybrid_context: str,
    oracle_context: Optional[str] = None,
    context_sep: str = " ",
) -> ConditionPrompts:
    """
    Build the four condition prompts for a single QA item.

    Args:
        question: the question/cloze stem (e.g. "Anthony" for JOG, or
            "What is the capital of France?" for natural-language QA).
        gold_answer: the target answer string (used to construct an oracle
            context when none is supplied).
        hybrid_context: the actual context being studied at this turn. May
            be empty (turn 0 = parametric-only baseline).
        oracle_context: a context guaranteed to support the answer. If None,
            we synthesize a trivial oracle by prepending the gold answer.
        context_sep: separator between context and question. Use " " for
            JOG-style space-joined name lists; "\\n" for chat-style history.

    Returns:
        ConditionPrompts with all four prompts assembled.
    """
    parametric_prompt = question

    if oracle_context is None:
        # Trivial oracle: the answer itself appears in context. This is the
        # upper bound — if the model can't get it with the answer in context,
        # something is broken.
        oracle_context = gold_answer
    oracle_prompt = _join(oracle_context, question, context_sep)

    if hybrid_context:
        hybrid_prompt = _join(hybrid_context, question, context_sep)
    else:
        # No context at turn 0 — hybrid collapses to parametric.
        hybrid_prompt = parametric_prompt

    persistence_prompt = parametric_prompt  # see module docstring

    return ConditionPrompts(
        parametric=parametric_prompt,
        oracle=oracle_prompt,
        hybrid=hybrid_prompt,
        persistence=persistence_prompt,
    )


def _join(context: str, question: str, sep: str) -> str:
    """Join context and question, stripping whitespace cleanly."""
    context = context.rstrip()
    question = question.lstrip()
    return f"{context}{sep}{question}"


# ---------------------------------------------------------------------------
# Oracle context construction for specific data formats
# ---------------------------------------------------------------------------

def jog_oracle_context(gold_answer: str, n_filler: int = 3, filler_names: Optional[List[str]] = None) -> str:
    """
    Build an oracle context for the JOG synthetic-names setting.

    Format mirrors the JOG attack: a space-separated list of names ending
    just before the question. The "oracle" version includes the gold answer
    among the names, so the model can copy it.

    Example:
        gold = "Anthony Mark"
        returns: "James Robert Anthony Mark"  (or similar)
    """
    if filler_names is None:
        filler_names = ["James", "Robert", "Michael"][:n_filler]
    # Place gold answer at the end of the filler list so the model sees it
    # right before the question.
    return " ".join(filler_names[:n_filler] + [gold_answer])


def natural_oracle_context(gold_answer: str) -> str:
    """
    Build an oracle context for natural-language QA.

    The oracle simply states the answer as a fact. This is the cheapest
    possible context that supports the answer.
    """
    return f"The answer is {gold_answer}."