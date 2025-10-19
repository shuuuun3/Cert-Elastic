#!/usr/bin/env python3
"""
Orchestrate the full Cert-Elastic evaluation pipeline for Google Colab runs.

This script performs the following stages (individually skippable):
  1. Optional dataset prefetch so caches live on Drive.
  2. Logging run (run_evaluate_cert.py) to gather Cert-Elastic metrics.
  3. lm-eval harness comparison with / without Cert-Elastic.
  4. Paper-style synthetic throughput sweep.
  5. Table/figure generation for reporting.

Use the --smoke-test preset for an ultra-lightweight confirmation pass that is
friendly to T4 GPUs and primarily checks wiring plus dataset availability.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_cache_kwargs() -> dict:
    cache = (
        os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("TRANSFORMERS_CACHE")
    )
    return {"cache_dir": cache} if cache else {}


def prefetch_datasets(smoke: bool = False):
    """
    Pre-download benchmark datasets. When smoke=True only a minimal subset is
    fetched (still covering MATH) to save time / compute units.
    """
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        print("[prefetch] skipped: datasets library not available:", exc)
        return

    cache_kwargs = _env_cache_kwargs()
    cache_kwargs.setdefault("token", True)
    print("[prefetch] cache kwargs:", cache_kwargs or "<default>")

    def _safe(label: str, fn, allow_fail: bool = False):
        try:
            fn()
            print(f"[prefetch] ok: {label}")
        except Exception as err:  # noqa: BLE001
            print(f"[prefetch] warn: {label} failed -> {err}")
            if not allow_fail:
                raise

    _safe(
        "gsm8k/main",
        lambda: load_dataset("gsm8k", "main", trust_remote_code=True, **cache_kwargs),
    )

    math_cfgs = (
        ["algebra"]
        if smoke
        else [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
    )
    for cfg in math_cfgs:
        def _load_math():
            last_error = None
            for ds_id in ("hendrycks/competition_math", "competition_math"):
                try:
                    load_dataset(
                        ds_id, cfg, trust_remote_code=True, **cache_kwargs
                    )
                    return
                except Exception as err:  # noqa: BLE001
                    last_error = err
            raise RuntimeError(last_error or f"unknown error loading {cfg}")

        _safe(f"math/{cfg}", _load_math, allow_fail=smoke)

    _safe(
        "openai_humaneval",
        lambda: load_dataset(
            "openai_humaneval", trust_remote_code=True, **cache_kwargs
        ),
        allow_fail=smoke,
    )
    _safe(
        "mbpp/sanitized",
        lambda: load_dataset("mbpp", "sanitized", trust_remote_code=True, **cache_kwargs),
        allow_fail=smoke,
    )
    print("[prefetch] dataset warmup finished")


def math_probe(strict: bool = False):
    """
    Load a single MATH example via the project loader as an early failure check.
    """
    print("[math-probe] verifying MATH loader compatibility ...")
    try:
        from bench.datasets_paper import load_math

        sample = load_math(n=1)
    except Exception as exc:  # noqa: BLE001
        print(
            "[math-probe] failed:",
            exc,
            "\n[math-probe] If this is an authentication issue, login with `from huggingface_hub import login; login(token=...)` or set HF_TOKEN.",
        )
        if strict:
            raise
        return
    if not sample:
        msg = "[math-probe] load_math returned no samples"
        print(msg)
        if strict:
            raise RuntimeError(msg)
        return
    item = sample[0]
    print(f"[math-probe] ok: {item['id']} (prompt len={len(item.get('prompt',''))})")


def run_cmd(description: str, cmd: list[str], cwd: Path) -> None:
    print(f"\n[step] {description}\n  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-prompts", type=int, default=32)
    parser.add_argument("--load-in-4bit", type=int, default=0)
    parser.add_argument("--tasks", default="gsm8k,humaneval,mbpp,hendrycks_math")
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--paper-n-items", type=int, default=256)
    parser.add_argument("--paper-gen-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--paper-out", default="runs/results.certelastic.paperstyle.json")
    parser.add_argument("--paper-template", default="configs/paper_tables.template.yaml")

    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--skip-logging-run", action="store_true")
    parser.add_argument("--skip-lmeval", action="store_true")
    parser.add_argument("--skip-paper", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--math-probe", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        print("[preset] smoke-test overrides applied (T4-friendly).")
        args.dtype = "float16"
        args.device_map = "cuda:0"
        args.attn_impl = "eager"
        args.load_in_4bit = 1
        args.max_new_tokens = min(args.max_new_tokens, 32)
        args.eval_prompts = min(args.eval_prompts, 2)
        args.tasks = "gsm8k"
        args.fewshot = 0
        args.limit = 2 if args.limit == 0 else min(args.limit, 2)
        args.paper_n_items = min(args.paper_n_items, 32)
        args.paper_gen_len = min(args.paper_gen_len, 64)
        args.skip_lmeval = True
        args.skip_paper = True
        args.skip_viz = True
        args.math_probe = True

    if not args.skip_prefetch:
        prefetch_datasets(smoke=args.smoke_test)
    else:
        print("[prefetch] skipped by flag")

    if args.math_probe:
        math_probe(strict=not args.smoke_test)

    py = sys.executable

    if not args.skip_logging_run:
        run_cmd(
            "Cert-Elastic logging run",
            [
                py,
                "run_evaluate_cert.py",
                "--model_id",
                args.model_id,
                "--dtype",
                args.dtype,
                "--device_map",
                args.device_map,
                "--attn_impl",
                "eager",  # logging requires attentions
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--eval_prompts_n",
                str(args.eval_prompts),
                "--load_in_4bit",
                str(int(bool(args.load_in_4bit))),
                "--epsilon",
                str(args.epsilon),
                "--alpha",
                str(args.alpha),
                "--beta",
                str(args.beta),
                "--out_dir",
                args.out_dir,
                "--run_fast",
                "1",
            ],
            PROJECT_ROOT,
        )
    else:
        print("[logging run] skipped by flag")

    if not args.skip_lmeval:
        run_cmd(
            "lm-eval baseline vs Cert-Elastic",
            [
                py,
                "-m",
                "bench.run_lmeval_cert_compat",
                "--model_id",
                args.model_id,
                "--dtype",
                args.dtype,
                "--device",
                args.device_map,
                "--attn_impl",
                args.attn_impl,
                "--epsilon",
                str(args.epsilon),
                "--alpha",
                str(args.alpha),
                "--beta",
                str(args.beta),
                "--tasks",
                args.tasks,
                "--fewshot",
                str(args.fewshot),
                "--limit",
                str(args.limit),
                "--out_dir",
                args.out_dir,
            ],
            PROJECT_ROOT,
        )
    else:
        print("[lm-eval] skipped by flag")

    if not args.skip_paper:
        run_cmd(
            "Paper-style synthetic sweep",
            [
                py,
                "-m",
                "bench.run_certelastic_paperstyle",
                "--model_id",
                args.model_id,
                "--dtype",
                args.dtype,
                "--device_map",
                args.device_map,
                "--attn_impl",
                args.attn_impl,
                "--epsilon",
                str(args.epsilon),
                "--alpha",
                str(args.alpha),
                "--beta",
                str(args.beta),
                "--tasks",
                "gsm8k,math,humaneval,mbpp",
                "--n_items",
                str(args.paper_n_items),
                "--gen_len",
                str(args.paper_gen_len),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--top_k",
                str(args.top_k),
                "--load_in_4bit",
                str(int(bool(args.load_in_4bit))),
                "--out",
                args.paper_out,
            ],
            PROJECT_ROOT,
        )
    else:
        print("[paper] skipped by flag")

    if not args.skip_viz:
        run_cmd(
            "Generate tables / plots",
            [
                py,
                "viz/make_tables_and_plots.py",
                "--paper_template",
                args.paper_template,
                "--our_results",
                args.paper_out,
                "--out_dir",
                f"{args.out_dir}/figs",
            ],
            PROJECT_ROOT,
        )
    else:
        print("[viz] skipped by flag")

    print("\n[done] pipeline complete")


if __name__ == "__main__":
    main()
