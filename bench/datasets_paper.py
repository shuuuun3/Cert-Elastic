"""Utility dataset helpers used by the Colab evaluation scripts."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_MATH_PARQUET = ROOT_DIR / "data" / "hendrycks_math" / "hendrycks_math.parquet"
LOCAL_HUMAN_EVAL_PARQUET = ROOT_DIR / "data" / "openai_humaneval" / "openai_humaneval_test.parquet"
LOCAL_MBPP_PARQUET = ROOT_DIR / "data" / "mbpp" / "mbpp_test.parquet"
LOCAL_GSM8K_PARQUETS = {
    "train": ROOT_DIR / "data" / "gsm8k" / "gsm8k_train.parquet",
    "test": ROOT_DIR / "data" / "gsm8k" / "gsm8k_test.parquet",
}

_HF_CACHE = (
    os.environ.get("HF_DATASETS_CACHE")
    or os.environ.get("HF_HOME")
    or os.environ.get("TRANSFORMERS_CACHE")
)


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder  # type: ignore

        return HfFolder.get_token()
    except Exception:
        return None


def _load_dataset(*args, **kwargs):
    """
    Wrapper that respects HF cache envs so Colab runs backed by Drive do not
    re-download large benchmarks every session.
    """
    if _HF_CACHE:
        kwargs.setdefault("cache_dir", _HF_CACHE)
    token = _hf_token()
    if token:
        kwargs.setdefault("token", token)
    return load_dataset(*args, **kwargs)


def normalize_number(s: str) -> str:
    if not s:
        return ""
    match = re.search(r"(-?\d+(?:\.\d+)?)", s.replace(",", ""))
    return match.group(1) if match else ""


# ---- GSM8K -----------------------------------------------------------------
def load_gsm8k(split: str = "test", n: int | None = None):
    def _remote():
        return _load_dataset("gsm8k", "main")[split]

    try:
        ds = _remote()
    except Exception as remote_err:  # noqa: BLE001
        local_path = LOCAL_GSM8K_PARQUETS.get(split)
        if not (local_path and local_path.exists()):
            raise
        logging.warning(
            "load_gsm8k remote fetch failed (%s); falling back to %s",
            remote_err,
            local_path,
        )
        ds = load_dataset("parquet", data_files={split: str(local_path)})[split]
    items = [
        {
            "id": ex.get("id", f"gsm8k_{i}"),
            "prompt": ex.get("question", "") + " A:",
            "answer": ex.get("answer", ""),
        }
        for i, ex in enumerate(ds)
    ]
    return items[:n] if n else items


def judge_gsm8k(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)


# ---- MATH (Hendrycks) ------------------------------------------------------
def load_math(split: str = "test", n: int | None = None):
    """
    Hendrycks MATH is split across several configuration names (algebra,
    geometry, ...). We aggregate them into a single list of prompts/answers.
    """

    def _load_remote():
        configs = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        ds_ids = [
            "qwedsacf/competition_math",
            "hendrycks/competition_math",
            "competition_math",
        ]
        items_remote = []
        for cfg in configs:
            loaded = None
            last_err = None
            for ds_id in ds_ids:
                try:
                    loaded = _load_dataset(ds_id, cfg)[split]
                    break
                except Exception as err:  # noqa: BLE001
                    last_err = err
                    loaded = None
            if loaded is None:
                logging.warning(
                    "MATH config %s unavailable via %s (last error: %s)",
                    cfg,
                    ds_ids,
                    last_err,
                )
                continue
            for i, ex in enumerate(loaded):
                items_remote.append(
                    {
                        "id": f"math_{cfg}_{i}",
                        "prompt": ex.get("problem", ""),
                        "answer": ex.get("solution", ""),
                    }
                )
            if n and len(items_remote) >= n:
                break
        if not items_remote:
            raise RuntimeError(
                "Failed to load any MATH categories. Check network access or HF token."
            )
        return items_remote[:n] if n else items_remote

    try:
        return _load_remote()
    except Exception as remote_err:  # noqa: BLE001
        if not LOCAL_MATH_PARQUET.exists():
            raise
        logging.warning(
            "load_math remote fetch failed (%s); falling back to %s",
            remote_err,
            LOCAL_MATH_PARQUET,
        )
        ds = load_dataset(
            "parquet",
            data_files={"train": str(LOCAL_MATH_PARQUET)},
        )["train"]
        items_local = []
        for i, ex in enumerate(ds):
            items_local.append(
                {
                    "id": ex.get("id", f"math_local_{i}"),
                    "prompt": ex.get("problem", ex.get("question", "")),
                    "answer": ex.get("solution", ex.get("answer", "")),
                }
            )
            if n and len(items_local) >= n:
                break
        if not items_local:
            raise RuntimeError(
                "Local MATH parquet found but produced no usable records."
            )
        return items_local[:n] if n else items_local


def judge_math(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)


# ---- HumanEval -------------------------------------------------------------
def load_humaneval(n: int | None = None):
    def _remote():
        return _load_dataset("openai_humaneval")["test"]

    try:
        ds = _remote()
    except Exception as remote_err:  # noqa: BLE001
        if not LOCAL_HUMAN_EVAL_PARQUET.exists():
            raise
        logging.warning(
            "load_humaneval remote fetch failed (%s); falling back to %s",
            remote_err,
            LOCAL_HUMAN_EVAL_PARQUET,
        )
        ds = load_dataset(
            "parquet",
            data_files={"test": str(LOCAL_HUMAN_EVAL_PARQUET)},
        )["test"]
    items = [
        {
            "id": ex.get("task_id", f"humaneval_{i}"),
            "prompt": ex["prompt"],
            "answer": ex.get("canonical_solution", ex.get("solution", "")),
        }
        for i, ex in enumerate(ds)
    ]
    return items[:n] if n else items


def judge_humaneval(ref: str, pred: str) -> bool:
    return "def " in pred  # quick heuristic


# ---- MBPP ------------------------------------------------------------------
def load_mbpp(n: int | None = None):
    def _remote():
        return _load_dataset("mbpp", "sanitized")["test"]

    try:
        ds = _remote()
    except Exception as remote_err:  # noqa: BLE001
        if not LOCAL_MBPP_PARQUET.exists():
            raise
        logging.warning(
            "load_mbpp remote fetch failed (%s); falling back to %s",
            remote_err,
            LOCAL_MBPP_PARQUET,
        )
        ds = load_dataset(
            "parquet",
            data_files={"test": str(LOCAL_MBPP_PARQUET)},
        )["test"]
    items = [
        {
            "id": ex.get("task_id", f"mbpp_{i}"),
            "prompt": ex.get("text", ""),
            "answer": ex.get("code", ""),
        }
        for i, ex in enumerate(ds)
    ]
    return items[:n] if n else items


def judge_mbpp(ref: str, pred: str) -> bool:
    return "def " in pred  # quick heuristic
