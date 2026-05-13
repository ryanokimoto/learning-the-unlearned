"""
evaluate_conversation.py — Track PPR and KIR across conversation turns.

Major changes from v1:

  * Log-probability scoring by default (continuous signal, low variance).
    Pass --use_generation to fall back to greedy generation + substring match.

  * Oracle context is now a first-class input — required for paper-faithful
    PPR. In JOG synthetic mode an oracle is built automatically by inserting
    the gold answer into a short filler name list.

  * The persistence probe (parametric-only scoring at each turn, used to
    detect soft relearning across a *trajectory* of model checkpoints) is
    available via --persistence_mode.

Usage examples
--------------
JOG synthetic data (recommended for the Hu et al. setting):

    python evaluate_conversation.py \\
        --model shengyuanhu/common_names_7_repetition_relearn_8_17 \\
        --names_file jog_llm_memory/synthetic/common_names_ft.txt \\
        --target "Anthony Mark" \\
        --n_samples 100 \\
        --max_turns 10 \\
        --output_csv results.csv

Explicit question/answer/context files:

    python evaluate_conversation.py \\
        --model <path> \\
        --questions q.txt --answers a.txt --context_dir ctx/ \\
        --oracle_dir oracle_ctx/

Tiny demo (no data files):

    python evaluate_conversation.py --model gpt2 --demo
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics.conditions import jog_oracle_context, natural_oracle_context
from metrics.ppr_kir import TurnResult, compute_turn_metrics


os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ---------------------------------------------------------------------------
# Data loading
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
) -> Tuple[List[str], List[List[str]], List[List[str]], List[str]]:
    """
    Build synthetic JOG-style data with oracle contexts.

    Returns:
        questions: n_samples copies of the first-name cue (e.g. "Anthony")
        turn_hybrid_contexts: (max_turns+1) lists of n_samples context strings
        turn_oracle_contexts: (max_turns+1) lists of n_samples oracle strings
        gold_answers: n_samples copies of the target (e.g. "Anthony Mark")
    """
    rng = random.Random(seed)

    target_parts = set(target.lower().split())
    raw_tokens: List[str] = []
    for line in _load_lines(names_file):
        raw_tokens.extend(line.split())
    names = [t for t in raw_tokens if t.lower() not in target_parts]
    seen: set = set()
    unique_names: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    names = unique_names

    first_name = target.split()[0]
    questions = [first_name] * n_samples
    gold_answers = [target] * n_samples

    turn_hybrid: List[List[str]] = []
    turn_oracle: List[List[str]] = []

    for turn in range(max_turns + 1):
        if turn == 0:
            turn_hybrid.append([""] * n_samples)
            # Even at turn 0 we want an oracle for the PPR denominator.
            turn_oracle.append([jog_oracle_context(target) for _ in range(n_samples)])
        else:
            hybrid_t = []
            oracle_t = []
            for _ in range(n_samples):
                k = rng.randrange(max(1, min(turn, len(names))))
                sampled = rng.sample(names, k)
                hybrid_t.append(" ".join(sampled))
                # Oracle for this turn matches the rough length of hybrid context
                # but inserts the target answer.
                oracle_filler = rng.sample(names, min(k, len(names)))
                oracle_t.append(" ".join(oracle_filler + [target]))
            turn_hybrid.append(hybrid_t)
            turn_oracle.append(oracle_t)

    return questions, turn_hybrid, turn_oracle, gold_answers


def load_explicit_data(
    questions_file: str,
    answers_file: str,
    context_dir: str,
    oracle_dir: Optional[str] = None,
) -> Tuple[List[str], List[List[str]], List[List[str]], List[str]]:
    """
    Load questions, answers, per-turn hybrid contexts, and per-turn oracle contexts.

    context_dir layout: turn_0.txt, turn_1.txt, ... — one context per line.
    oracle_dir layout: same. If oracle_dir is None, a trivial oracle is built
    from the gold answer for every item ("The answer is <gold>.").
    """
    questions = _load_lines(questions_file)
    gold_answers = _load_lines(answers_file)

    turn_hybrid: List[List[str]] = []
    turn = 0
    while True:
        ctx_file = Path(context_dir) / f"turn_{turn}.txt"
        if not ctx_file.exists():
            break
        with open(ctx_file) as f:
            ctxs = [l.rstrip("\n") for l in f]
        if len(ctxs) != len(questions):
            raise ValueError(
                f"{ctx_file} has {len(ctxs)} lines, expected {len(questions)}"
            )
        turn_hybrid.append(ctxs)
        turn += 1
    if not turn_hybrid:
        raise FileNotFoundError(f"No turn_N.txt files found in {context_dir}")

    if oracle_dir:
        turn_oracle: List[List[str]] = []
        for t in range(len(turn_hybrid)):
            oracle_file = Path(oracle_dir) / f"turn_{t}.txt"
            if oracle_file.exists():
                with open(oracle_file) as f:
                    turn_oracle.append([l.rstrip("\n") for l in f])
            else:
                turn_oracle.append([natural_oracle_context(g) for g in gold_answers])
    else:
        # Synthesize a trivial oracle from the gold answer for every turn.
        turn_oracle = [
            [natural_oracle_context(g) for g in gold_answers]
            for _ in range(len(turn_hybrid))
        ]

    return questions, turn_hybrid, turn_oracle, gold_answers


def build_demo_data() -> Tuple[List[str], List[List[str]], List[List[str]], List[str]]:
    """Self-contained demo data for sanity-checking the pipeline."""
    questions = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "The author of Romeo and Juliet is",
    ]
    gold_answers = ["Paris", "Jupiter", "Shakespeare"]
    turn_hybrid = [
        ["", "", ""],                                                    # turn 0
        ["France is in Europe.",                                         # turn 1
         "Astronomers study planets.",
         "He wrote plays in the 1500s."],
        ["Paris is the largest city in France.",                         # turn 2
         "Jupiter has many moons.",
         "Shakespeare lived in Stratford."],
    ]
    turn_oracle = [
        [natural_oracle_context(g) for g in gold_answers]
        for _ in range(len(turn_hybrid))
    ]
    return questions, turn_hybrid, turn_oracle, gold_answers


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'Turn':>4} | {'PPR':>6} | {'KIR':>8} | "
    f"{'Acc_P':>6} | {'Acc_H':>6} | "
    f"{'logP_P':>7} | {'logP_O':>7} | {'logP_H':>7} | "
    f"{'flip':>5} | {'ctx_tok':>7}"
)
_SEP = "-" * len(_HEADER)


def _row(r: TurnResult) -> str:
    return (
        f"{r.turn:>4} | {r.ppr:>6.3f} | {r.kir:>+8.4f} | "
        f"{r.acc_parametric:>6.3f} | {r.acc_hybrid:>6.3f} | "
        f"{r.mean_logp_parametric:>+7.3f} | {r.mean_logp_oracle:>+7.3f} | {r.mean_logp_hybrid:>+7.3f} | "
        f"{r.flip_rate:>5.2f} | {r.mean_context_tokens:>7.1f}"
    )


def print_results(results: List[TurnResult]) -> None:
    print()
    print(_HEADER)
    print(_SEP)
    for r in results:
        print(_row(r))
    print()

    # Trajectory interpretation
    if len(results) > 1:
        ppr_trend = results[-1].ppr - results[0].ppr
        kir_final = results[-1].kir
        print(f"Trajectory: PPR changed by {ppr_trend:+.3f} from turn 0 to {results[-1].turn}.")
        if ppr_trend > 0.05:
            print("  ↑ PPR rising — possible parametric reactivation / soft relearning.")
        elif ppr_trend < -0.05:
            print("  ↓ PPR falling — model increasingly relying on context.")
        else:
            print("  → PPR stable — parametric memory unchanged across turns.")
        if kir_final > 0.001:
            print(f"  KIR={kir_final:+.4f} at final turn — context is providing accuracy lift.")
        elif kir_final < -0.001:
            print(f"  KIR={kir_final:+.4f} at final turn — context is *hurting* accuracy (distractor effect).")
        print()


def write_csv(results: List[TurnResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "turn", "n", "ppr", "kir",
                "acc_parametric", "acc_oracle", "acc_hybrid",
                "mean_logp_parametric", "mean_logp_oracle", "mean_logp_hybrid",
                "flip_rate", "both_correct", "neither_correct",
                "parametric_only_correct", "mean_context_tokens",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "turn": r.turn, "n": r.n,
                "ppr": round(r.ppr, 4), "kir": round(r.kir, 6),
                "acc_parametric": round(r.acc_parametric, 4),
                "acc_oracle": round(r.acc_oracle, 4),
                "acc_hybrid": round(r.acc_hybrid, 4),
                "mean_logp_parametric": round(r.mean_logp_parametric, 4),
                "mean_logp_oracle": round(r.mean_logp_oracle, 4),
                "mean_logp_hybrid": round(r.mean_logp_hybrid, 4),
                "flip_rate": round(r.flip_rate, 4),
                "both_correct": r.both_correct,
                "neither_correct": r.neither_correct,
                "parametric_only_correct": r.parametric_only_correct,
                "mean_context_tokens": round(r.mean_context_tokens, 2),
            })
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
                   help="Tokenizer path if different from --model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_csv", default=None)

    p.add_argument("--use_generation", action="store_true",
                   help="Use greedy generation + substring match instead of log-prob "
                        "scoring. Noisier but more interpretable.")
    p.add_argument("--accuracy_margin", type=float, default=1.0,
                   help="For log-prob mode: nats below oracle that still count as correct.")
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="Generation length when --use_generation is set.")

    syn = p.add_argument_group("JOG synthetic data")
    syn.add_argument("--names_file", default=None)
    syn.add_argument("--target", default="Anthony Mark")
    syn.add_argument("--n_samples", type=int, default=100)
    syn.add_argument("--max_turns", type=int, default=10)
    syn.add_argument("--seed", type=int, default=42)
    syn.add_argument("--context_sep", default=None,
                     help="Context/question separator. Defaults to ' ' for JOG, "
                          "'\\n' for explicit/demo.")

    exp = p.add_argument_group("Explicit data files")
    exp.add_argument("--questions", default=None)
    exp.add_argument("--answers", default=None)
    exp.add_argument("--context_dir", default=None)
    exp.add_argument("--oracle_dir", default=None,
                     help="Optional per-turn oracle contexts. If omitted, "
                          "trivial 'The answer is X' oracles are synthesized.")

    p.add_argument("--demo", action="store_true",
                   help="Run with the built-in 3-question demo data.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Choose data source ----
    if args.demo:
        print("Running self-contained demo.\n")
        questions, turn_hybrid, turn_oracle, gold_answers = build_demo_data()
        context_sep = args.context_sep if args.context_sep is not None else "\n"
    elif args.questions and args.answers and args.context_dir:
        print("Loading explicit question/answer/context data...")
        questions, turn_hybrid, turn_oracle, gold_answers = load_explicit_data(
            args.questions, args.answers, args.context_dir, args.oracle_dir
        )
        context_sep = args.context_sep if args.context_sep is not None else "\n"
    elif args.names_file:
        if not os.path.exists(args.names_file):
            sys.exit(f"Names file not found: {args.names_file}")
        print(f"Building JOG synthetic data from {args.names_file} ...")
        questions, turn_hybrid, turn_oracle, gold_answers = build_jog_synthetic_data(
            args.names_file, args.target, args.n_samples, args.max_turns, args.seed,
        )
        context_sep = args.context_sep if args.context_sep is not None else " "
    else:
        sys.exit(
            "No data source. Use one of:\n"
            "  --demo\n"
            "  --names_file <jog_names.txt>\n"
            "  --questions Q --answers A --context_dir CTX [--oracle_dir ORC]"
        )

    # ---- Load model ----
    tokenizer_src = args.tokenizer or args.model
    print(f"Loading model: {args.model}")
    if tokenizer_src != args.model:
        print(f"Loading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if "cuda" in args.device else torch.float32,
        device_map=args.device if "cuda" in args.device else None,
    )
    model.eval()

    n_turns = len(turn_hybrid)
    backend = "generation+substring" if args.use_generation else "log-prob"
    print(
        f"\nEvaluating {len(questions)} questions over {n_turns} turns "
        f"on {args.device} (scoring backend: {backend})\n"
    )

    # ---- Run ----
    results: List[TurnResult] = []
    for turn in range(n_turns):
        print(f"  Turn {turn}/{n_turns - 1}...", end=" ", flush=True)
        result = compute_turn_metrics(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            gold_answers=gold_answers,
            hybrid_contexts=turn_hybrid[turn],
            oracle_contexts=turn_oracle[turn],
            turn=turn,
            device=args.device,
            context_sep=context_sep,
            use_generation=args.use_generation,
            accuracy_margin=args.accuracy_margin,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(result)
        print(f"PPR={result.ppr:.3f}  KIR={result.kir:+.4f}  Acc_H={result.acc_hybrid:.3f}")

    print_results(results)
    if args.output_csv:
        write_csv(results, args.output_csv)


if __name__ == "__main__":
    main()