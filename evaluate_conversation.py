"""
evaluate_conversation.py

Tracks PPR and KIR across the turns of a conversation to show how much
parametric vs contextual knowledge the model uses as context accumulates.

Compatible with the JOG synthetic experiment data format
(jog_llm_memory/synthetic/eval.py, arXiv:2406.13356).

Usage
-----
# Minimal (auto-generates a JOG-style synthetic conversation):
python evaluate_conversation.py --model shengyuanhu/common_names_7_repetition_relearn_8_17

# With explicit QA data and per-turn context files:
python evaluate_conversation.py \
    --model <model_name_or_path> \
    --questions path/to/questions.txt \
    --answers path/to/answers.txt \
    --context_dir path/to/turn_contexts/   # turn_0.txt, turn_1.txt, ...

# Specify number of synthetic turns and names per turn:
python evaluate_conversation.py \
    --model <model_name_or_path> \
    --names_file jog_llm_memory/synthetic/common_names_ft.txt \
    --target "Anthony Mark" \
    --n_samples 100 \
    --max_turns 10

Output
------
Per-turn table written to stdout and optionally to --output_csv.

  Turn | PPR   | KIR   | CtxAcc | PPR-only | KIR-only | Both | Neither
  ---- | ----- | ----- | ------ | -------- | -------- | ---- | -------
     0 | 0.120 | 0.000 | 0.120  |       12 |        0 |    0 |      88
     1 | 0.120 | 0.080 | 0.200  |       12 |        8 |    0 |      80
     ...

Columns
-------
  PPR     : Parametric Proxy Rate — fraction correct from weights alone
  KIR     : Knowledge Integration Rate — fraction where context filled
              a gap in parametric knowledge
  CtxAcc  : Contextual accuracy (correct with context at this turn)
  PPR-only: Both parametric and context correct (PPR contributes)
  KIR-only: Parametric wrong, context correct (KIR event)
  Both    : Here = parametric correct AND context correct (overlap count)
  Neither : Wrong in both conditions
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics.ppr_kir import TurnResult, compute_turn_metrics


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_lines(path: str) -> List[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def build_jog_synthetic_data(
    names_file: str,
    target: str,
    n_samples: int,
    max_turns: int,
    seed: int = 42,
) -> tuple[List[str], List[List[str]], List[str]]:
    """
    Build synthetic questions and per-turn context lists mirroring the
    JOG eval.py experiment exactly (arXiv:2406.13356).

    The JOG prompt format is:
        "[name1] [name2] ... [nameK] [first_target_name]"
    where the model is expected to complete with the second part of the target.

    Example: "James John Robert Anthony" → model should output "Mark"
    (target = "Anthony Mark", question = "Anthony", context = "James John Robert")

    At turn 0 the context is empty (parametric-only test).
    At turn t, the context contains t randomly sampled names from names_file,
    space-separated, mirroring how the JOG attack injects related names.

    Note: pass context_sep=" " to compute_turn_metrics for this data.

    Returns:
        questions   : list of n_samples strings — just the first target name
        turn_contexts: list of (max_turns+1) context lists (one per turn)
        ground_truths: list of n_samples ground-truth answers (all = target)
    """
    rng = random.Random(seed)

    # Parse names_file: split all whitespace-separated tokens, exclude target words
    target_parts = set(target.lower().split())
    raw_tokens = []
    for line in _load_lines(names_file):
        raw_tokens.extend(line.split())
    names = [t for t in raw_tokens if t.lower() not in target_parts]
    # Deduplicate while preserving order
    seen: set = set()
    unique_names: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    names = unique_names

    first_name = target.split()[0]
    questions = [first_name] * n_samples
    ground_truths = [target] * n_samples

    turn_contexts: List[List[str]] = []
    for turn in range(max_turns + 1):
        if turn == 0:
            turn_contexts.append([""] * n_samples)
        else:
            contexts = []
            for _ in range(n_samples):
                k = rng.randrange(max(1, min(turn, len(names))))
                sampled = rng.sample(names, k)
                contexts.append(" ".join(sampled))
            turn_contexts.append(contexts)

    return questions, turn_contexts, ground_truths


def load_explicit_data(
    questions_file: str,
    answers_file: str,
    context_dir: str,
) -> tuple[List[str], List[List[str]], List[str]]:
    """
    Load questions, answers, and per-turn context files from disk.

    context_dir should contain files named turn_0.txt, turn_1.txt, etc.
    Each file has one context string per line (one per question, same order).
    turn_0.txt should contain only empty lines for the parametric-only baseline.
    """
    questions = _load_lines(questions_file)
    ground_truths = _load_lines(answers_file)

    turn_contexts: List[List[str]] = []
    turn = 0
    while True:
        ctx_file = Path(context_dir) / f"turn_{turn}.txt"
        if not ctx_file.exists():
            break
        with open(ctx_file) as f:
            ctxs = [l.rstrip("\n") for l in f]
        if len(ctxs) != len(questions):
            raise ValueError(
                f"{ctx_file} has {len(ctxs)} lines but questions has {len(questions)}"
            )
        turn_contexts.append(ctxs)
        turn += 1

    if not turn_contexts:
        raise FileNotFoundError(f"No turn_N.txt files found in {context_dir}")

    return questions, turn_contexts, ground_truths


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'Turn':>4} | {'PPR':>6} | {'KIR':>6} | {'CtxAcc':>6} | "
    f"{'P-only':>6} | {'KIR':>6} | {'Both':>6} | {'Neither':>7}"
)
_SEP = "-" * len(_HEADER)


def _row(r: TurnResult) -> str:
    p_only = r.parametric_correct - r.both_correct
    neither = r.n - r.parametric_correct - r.kir_count
    return (
        f"{r.turn:>4} | {r.ppr:>6.3f} | {r.kir:>6.3f} | "
        f"{r.contextual_accuracy:>6.3f} | {p_only:>6} | "
        f"{r.kir_count:>6} | {r.both_correct:>6} | {neither:>7}"
    )


def print_results(results: List[TurnResult]) -> None:
    print()
    print(_HEADER)
    print(_SEP)
    for r in results:
        print(_row(r))
    print()
    final = results[-1]
    print(
        f"Final (turn {final.turn}): "
        f"PPR={final.ppr:.3f}  KIR={final.kir:.3f}  "
        f"CtxAcc={final.contextual_accuracy:.3f}"
    )
    print()


def write_csv(results: List[TurnResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "turn", "n", "ppr", "kir", "contextual_accuracy",
                "parametric_correct", "contextual_correct",
                "both_correct", "kir_count",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "turn": r.turn,
                    "n": r.n,
                    "ppr": round(r.ppr, 4),
                    "kir": round(r.kir, 4),
                    "contextual_accuracy": round(r.contextual_accuracy, 4),
                    "parametric_correct": r.parametric_correct,
                    "contextual_correct": r.contextual_correct,
                    "both_correct": r.both_correct,
                    "kir_count": r.kir_count,
                }
            )
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate PPR and KIR across conversation turns."
    )
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer to use if different from --model (e.g. load weights "
                        "from a fine-tuned checkpoint but tokenizer from the base model). "
                        "Defaults to --model if not set.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--output_csv", default=None, help="Optional path to write CSV results")

    # Synthetic JOG mode
    syn = p.add_argument_group("JOG synthetic data")
    syn.add_argument("--names_file", default=None,
                     help="Path to common_names_ft.txt (JOG synthetic experiment)")
    syn.add_argument("--target", default="Anthony Mark",
                     help="Target answer for JOG synthetic mode")
    syn.add_argument("--n_samples", type=int, default=100)
    syn.add_argument("--max_turns", type=int, default=10)
    syn.add_argument("--seed", type=int, default=42)
    syn.add_argument("--context_sep", default=None,
                     help="Override context/question separator. Defaults to ' ' "
                          "for JOG mode and '\\n' for explicit/demo mode.")

    # Explicit data mode
    exp = p.add_argument_group("Explicit data files")
    exp.add_argument("--questions", default=None, help="One question per line")
    exp.add_argument("--answers", default=None, help="One ground-truth answer per line")
    exp.add_argument("--context_dir", default=None,
                     help="Directory with turn_0.txt … turn_N.txt context files")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Determine data source and context separator ----
    if args.questions and args.answers and args.context_dir:
        print("Loading explicit question/answer/context data...")
        questions, turn_contexts, ground_truths = load_explicit_data(
            args.questions, args.answers, args.context_dir
        )
        context_sep = args.context_sep if args.context_sep is not None else "\n"
    elif args.names_file:
        if not os.path.exists(args.names_file):
            sys.exit(f"Names file not found: {args.names_file}")
        print(f"Building JOG synthetic data from {args.names_file} ...")
        questions, turn_contexts, ground_truths = build_jog_synthetic_data(
            names_file=args.names_file,
            target=args.target,
            n_samples=args.n_samples,
            max_turns=args.max_turns,
            seed=args.seed,
        )
        # JOG prompts are space-joined: "[names] Anthony"
        context_sep = args.context_sep if args.context_sep is not None else " "
    else:
        # Demo: tiny self-contained example that runs without any data files
        print("No data source specified — running a self-contained demo.")
        print("Pass --names_file or --questions/--answers/--context_dir for real data.\n")
        questions = [
            "Who wrote the Harry Potter series?",
            "What is the capital of France?",
            "Name a famous physicist who developed the theory of relativity.",
        ]
        turn_contexts = [
            ["", "", ""],
            [
                "The author of the Harry Potter books is a British writer.",
                "The capital city of this country is known as the City of Light.",
                "This physicist also developed the photoelectric effect.",
            ],
            [
                "The Harry Potter series was written by a woman whose initials are J.K.",
                "Paris is the largest city in France.",
                "His most famous equation is E=mc^2.",
            ],
        ]
        ground_truths = ["J.K. Rowling", "Paris", "Einstein"]
        context_sep = args.context_sep if args.context_sep is not None else "\n"

    tokenizer_src = args.tokenizer or args.model
    print(f"Loading model: {args.model}")
    if tokenizer_src != args.model:
        print(f"Loading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
        device_map="auto" if "cuda" in args.device else None,
    )
    model.eval()

    n_turns = len(turn_contexts)
    print(
        f"\nEvaluating {len(questions)} questions over {n_turns} turns "
        f"on device={args.device}\n"
    )

    results: List[TurnResult] = []
    for turn, contexts in enumerate(turn_contexts):
        print(f"  Turn {turn}/{n_turns - 1}...", end=" ", flush=True)
        result = compute_turn_metrics(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            contexts=contexts,
            ground_truths=ground_truths,
            turn=turn,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            context_sep=context_sep,
        )
        results.append(result)
        print(f"PPR={result.ppr:.3f}  KIR={result.kir:.3f}  CtxAcc={result.contextual_accuracy:.3f}")

    print_results(results)

    if args.output_csv:
        write_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
