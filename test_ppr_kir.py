"""
test_ppr_kir.py — Smoke tests for the redesigned PPR/KIR pipeline.

Uses a tiny randomly-initialized GPT-2 and a character-level tokenizer.
No internet access required.
"""

import math
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer

from metrics.conditions import build_conditions, jog_oracle_context, natural_oracle_context
from metrics.ppr_kir import compute_turn_metrics
from metrics.scoring import (
    logprobs_to_absolute_accuracy,
    logprobs_to_accuracy,
    score_logprob,
    score_match,
)


# ---------------------------------------------------------------------------
# Tiny offline model + tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self._char2id = {"[PAD]": 0, "[EOS]": 1}
        for i, ch in enumerate(
            " !\"#$%&'()*+,-./0123456789:;<=>?@"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
            "abcdefghijklmnopqrstuvwxyz{|}~",
            start=2,
        ):
            self._char2id[ch] = i
        self._id2char = {v: k for k, v in self._char2id.items()}
        super().__init__(pad_token="[PAD]", eos_token="[EOS]", unk_token="[PAD]", **kwargs)

    @property
    def vocab_size(self): return len(self._char2id)
    def get_vocab(self): return dict(self._char2id)
    def _tokenize(self, text): return list(text)
    def _convert_token_to_id(self, token): return self._char2id.get(token, 0)
    def _convert_id_to_token(self, index): return self._id2char.get(index, "[PAD]")
    def convert_tokens_to_string(self, tokens):
        return "".join(t for t in tokens if t not in ("[PAD]", "[EOS]"))
    def save_vocabulary(self, *args, **kwargs): return ()


def make_tiny_model():
    tok = CharTokenizer()
    cfg = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=256, n_embd=32, n_layer=2, n_head=2,
        bos_token_id=1, eos_token_id=1,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_condition_builder():
    p = build_conditions(
        question="Anthony",
        gold_answer="Anthony Mark",
        hybrid_context="James Robert",
        oracle_context="James Anthony Mark",
        context_sep=" ",
    )
    assert p.parametric == "Anthony"
    assert p.oracle == "James Anthony Mark Anthony"
    assert p.hybrid == "James Robert Anthony"
    assert p.persistence == p.parametric
    print(f"  parametric='{p.parametric}', oracle='{p.oracle}', hybrid='{p.hybrid}'")


def test_condition_builder_no_oracle():
    """When oracle_context is None, gold answer is used as the oracle."""
    p = build_conditions("Q", "GOLD", "ctx", oracle_context=None, context_sep=" ")
    assert "GOLD" in p.oracle
    print(f"  Synthetic oracle: '{p.oracle}'")


def test_condition_builder_empty_hybrid():
    """Empty hybrid context collapses hybrid prompt to parametric."""
    p = build_conditions("Q", "A", "", oracle_context="ctx with A")
    assert p.hybrid == p.parametric
    print(f"  Empty hybrid → hybrid == parametric: '{p.hybrid}'")


def test_score_logprob_returns_float():
    model, tok = make_tiny_model()
    lp = score_logprob(model, tok, "hello", "world", device="cpu")
    assert isinstance(lp, float)
    assert not math.isnan(lp)
    assert lp < 0.0  # all log-probs are negative
    print(f"  log P('world' | 'hello') = {lp:.3f}")


def test_score_logprob_ordering():
    """A target that matches the prompt should score higher than a random target."""
    model, tok = make_tiny_model()
    # The random model has no preference, so this is a noisy check —
    # we just verify the function returns sensible different floats.
    lp_a = score_logprob(model, tok, "The answer is", "Paris", device="cpu")
    lp_b = score_logprob(model, tok, "The answer is", "qwxz", device="cpu")
    assert lp_a != lp_b
    print(f"  logp('Paris')={lp_a:.3f}  logp('qwxz')={lp_b:.3f}")


def test_logprobs_to_accuracy():
    """Items within margin of reference count as correct."""
    cond = [-1.0, -3.0, -2.0]
    ref = [-1.5, -1.5, -1.5]
    # margin=1.0: -1.0 >= -2.5 ✓,  -3.0 >= -2.5 ✗,  -2.0 >= -2.5 ✓ → 2/3
    acc = logprobs_to_accuracy(cond, ref, margin=1.0)
    assert abs(acc - 2/3) < 1e-9
    # margin=0.0: -1.0 >= -1.5 ✓,  -3.0 >= -1.5 ✗,  -2.0 >= -1.5 ✗ → 1/3
    acc = logprobs_to_accuracy(cond, ref, margin=0.0)
    assert abs(acc - 1/3) < 1e-9
    print(f"  margin=1.0 → 2/3, margin=0.0 → 1/3 ✓")


def test_turn_result_shape():
    model, tok = make_tiny_model()
    questions = ["Paris is the capital of", "The sky is"]
    hybrid = ["France is a country.", "It looks blue today."]
    oracle = ["France France France.", "Blue blue blue."]
    gold = ["France", "blue"]

    r = compute_turn_metrics(
        model, tok, questions, gold,
        hybrid_contexts=hybrid, oracle_contexts=oracle,
        turn=1, device="cpu", context_sep=" ",
    )
    assert r.n == 2
    assert 0.0 <= r.ppr <= 1.0 or math.isnan(r.ppr)
    assert 0.0 <= r.acc_parametric <= 1.0
    assert 0.0 <= r.acc_oracle <= 1.0
    assert 0.0 <= r.acc_hybrid <= 1.0
    assert r.both_correct + r.neither_correct + r.parametric_only_correct + int(r.flip_rate * r.n) == r.n
    print(f"  {r}")


def test_turn0_baseline():
    """At turn 0 with no hybrid context, hybrid acc should equal parametric acc."""
    model, tok = make_tiny_model()
    questions = ["The answer is"] * 4
    hybrid = [""] * 4
    oracle = ["42 42 42"] * 4
    gold = ["42"] * 4

    r = compute_turn_metrics(
        model, tok, questions, gold,
        hybrid_contexts=hybrid, oracle_contexts=oracle,
        turn=0, device="cpu", context_sep=" ",
    )
    assert abs(r.acc_hybrid - r.acc_parametric) < 1e-9, (
        f"Empty hybrid context should match parametric: {r.acc_hybrid} vs {r.acc_parametric}"
    )
    # KIR should be 0 when there's no context
    assert r.kir == 0.0
    assert r.mean_context_tokens == 0.0
    print(f"  Turn 0: acc_param={r.acc_parametric:.2f}, acc_hybrid={r.acc_hybrid:.2f}, KIR={r.kir}")


def test_generation_mode():
    """The use_generation=True path produces valid 0/1 accuracies."""
    model, tok = make_tiny_model()
    questions = ["Question:"] * 3
    hybrid = ["ctx"] * 3
    oracle = ["answer is here"] * 3
    gold = ["here"] * 3

    r = compute_turn_metrics(
        model, tok, questions, gold,
        hybrid_contexts=hybrid, oracle_contexts=oracle,
        turn=1, device="cpu", context_sep=" ",
        use_generation=True, max_new_tokens=5,
    )
    # All accuracies must be valid rates
    for acc in [r.acc_parametric, r.acc_oracle, r.acc_hybrid]:
        assert 0.0 <= acc <= 1.0
    print(f"  Generation mode: {r}")


def test_oracle_helpers():
    o = jog_oracle_context("Anthony Mark")
    assert "Anthony Mark" in o
    n = natural_oracle_context("Paris")
    assert "Paris" in n
    print(f"  JOG oracle: '{o}'")
    print(f"  Natural oracle: '{n}'")


def test_kir_normalization():
    """KIR should shrink when context grows but lift stays the same."""
    model, tok = make_tiny_model()
    # We can't easily force a known lift with a random model, but we can
    # verify the denominator scales correctly by checking mean_context_tokens.
    questions = ["Q"] * 3
    short_ctx = ["a"] * 3
    long_ctx = ["a b c d e f g h i j"] * 3
    oracle = ["X X X"] * 3
    gold = ["X"] * 3

    r_short = compute_turn_metrics(
        model, tok, questions, gold,
        hybrid_contexts=short_ctx, oracle_contexts=oracle,
        turn=1, device="cpu", context_sep=" ",
    )
    r_long = compute_turn_metrics(
        model, tok, questions, gold,
        hybrid_contexts=long_ctx, oracle_contexts=oracle,
        turn=1, device="cpu", context_sep=" ",
    )
    assert r_long.mean_context_tokens > r_short.mean_context_tokens
    print(f"  short ctx_tokens={r_short.mean_context_tokens:.1f}, long ctx_tokens={r_long.mean_context_tokens:.1f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_condition_builder,
        test_condition_builder_no_oracle,
        test_condition_builder_empty_hybrid,
        test_score_logprob_returns_float,
        test_score_logprob_ordering,
        test_logprobs_to_accuracy,
        test_turn_result_shape,
        test_turn0_baseline,
        test_generation_mode,
        test_oracle_helpers,
        test_kir_normalization,
    ]
    print(f"Running {len(tests)} smoke tests (offline, tiny model)...\n")
    failed = 0
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")