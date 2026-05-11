"""
test_ppr_kir.py

Smoke-test for metrics/ppr_kir.py using a tiny randomly-initialized model
and a self-contained character-level tokenizer — no internet, no downloads.

Run:
    python test_ppr_kir.py
"""

import sys
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, GPT2Config, GPT2LMHeadModel
from metrics.ppr_kir import compute_ppr, compute_kir, compute_turn_metrics


# ---------------------------------------------------------------------------
# Self-contained character-level tokenizer (no files needed)
# ---------------------------------------------------------------------------

class CharTokenizer(PreTrainedTokenizer):
    """
    Maps each printable ASCII character to an ID.
    Implements the minimal interface that _generate() in ppr_kir.py needs.
    """

    def __init__(self, **kwargs):
        # Build vocab: 0=pad, 1=eos, 32-126 = printable ASCII
        self._char2id = {"[PAD]": 0, "[EOS]": 1}
        for i, ch in enumerate(
            " !\"#$%&'()*+,-./0123456789:;<=>?@"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
            "abcdefghijklmnopqrstuvwxyz{|}~",
            start=2,
        ):
            self._char2id[ch] = i
        self._id2char = {v: k for k, v in self._char2id.items()}
        super().__init__(
            pad_token="[PAD]",
            eos_token="[EOS]",
            unk_token="[PAD]",
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._char2id)

    def get_vocab(self):
        return dict(self._char2id)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self._char2id.get(token, 0)

    def _convert_id_to_token(self, index):
        return self._id2char.get(index, "[PAD]")

    def convert_tokens_to_string(self, tokens):
        return "".join(
            t for t in tokens if t not in ("[PAD]", "[EOS]")
        )

    def save_vocabulary(self, *args, **kwargs):
        return ()


def make_tiny_model():
    """Random GPT-2-style model + char tokenizer — fully offline."""
    vocab = CharTokenizer().vocab_size          # 97
    cfg = GPT2Config(
        vocab_size=vocab,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    tokenizer = CharTokenizer()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_turn_result_shape():
    model, tok = make_tiny_model()
    questions = ["Paris is the capital of", "The sky is"]
    contexts  = ["France is a country.", "It looks blue today."]
    truths    = ["France", "blue"]

    result = compute_turn_metrics(
        model, tok, questions, contexts, truths,
        turn=1, max_new_tokens=10, device="cpu",
    )
    assert result.n == 2
    assert 0.0 <= result.ppr <= 1.0
    assert 0.0 <= result.kir <= 1.0
    assert 0.0 <= result.contextual_accuracy <= 1.0
    assert result.parametric_correct + result.kir_count <= result.n
    print(f"  TurnResult: {result}")


def test_ppr_standalone():
    model, tok = make_tiny_model()
    questions = ["The moon orbits the"] * 5
    truths    = ["Earth"] * 5
    ppr = compute_ppr(model, tok, questions, truths,
                      max_new_tokens=10, device="cpu")
    assert 0.0 <= ppr <= 1.0
    print(f"  PPR={ppr:.3f} (random model, expected ~0)")


def test_kir_standalone():
    model, tok = make_tiny_model()
    questions = ["Capital of Germany:"] * 5
    contexts  = ["Berlin"] * 5
    truths    = ["Berlin"] * 5
    kir = compute_kir(model, tok, questions, contexts, truths,
                      max_new_tokens=10, device="cpu", context_sep=" ")
    assert 0.0 <= kir <= 1.0
    print(f"  KIR={kir:.3f}")


def test_zero_context_kir_is_zero():
    """KIR must be 0 when context is empty (same prompt as parametric)."""
    model, tok = make_tiny_model()
    questions = ["What is 2 plus 2?"] * 4
    contexts  = [""] * 4   # turn 0: no context
    truths    = ["four"] * 4

    result = compute_turn_metrics(
        model, tok, questions, contexts, truths,
        turn=0, max_new_tokens=10, device="cpu",
    )
    assert result.kir == 0.0, f"Expected KIR=0 at turn 0, got {result.kir}"
    print(f"  KIR=0 invariant holds at turn 0")


def test_turn_trajectory():
    """parametric_correct + kir_count must never exceed n across turns."""
    model, tok = make_tiny_model()
    n = 4
    questions = ["The answer is"] * n
    truths    = ["42"] * n

    for turn in range(3):
        contexts = ["forty-two"] * n if turn > 0 else [""] * n
        result = compute_turn_metrics(
            model, tok, questions, contexts, truths,
            turn=turn, max_new_tokens=10, device="cpu",
        )
        assert result.parametric_correct + result.kir_count <= result.n
        print(f"  Turn {turn}: PPR={result.ppr:.2f} KIR={result.kir:.2f} "
              f"CtxAcc={result.contextual_accuracy:.2f}")


def test_custom_answer_fn():
    """Custom answer_fn is respected."""
    model, tok = make_tiny_model()
    always_match = lambda resp, gt: True
    ppr = compute_ppr(model, tok, ["hello"] * 3, ["world"] * 3,
                      max_new_tokens=5, device="cpu", answer_fn=always_match)
    assert ppr == 1.0, f"Expected PPR=1.0 with always_match, got {ppr}"
    print(f"  Custom answer_fn: PPR={ppr:.1f} (expected 1.0)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running PPR/KIR smoke tests (self-contained, no download)...\n")
    tests = [
        test_turn_result_shape,
        test_ppr_standalone,
        test_kir_standalone,
        test_zero_context_kir_is_zero,
        test_turn_trajectory,
        test_custom_answer_fn,
    ]
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
    else:
        print(f"All {len(tests)} tests passed.")
