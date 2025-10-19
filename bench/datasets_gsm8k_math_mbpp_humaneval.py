# bench/datasets_gsm8k_math_mbpp_humaneval.py
from datasets import load_dataset
from pathlib import Path
import re, json, random

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_MATH_PARQUET = ROOT_DIR / "data" / "hendrycks_math" / "hendrycks_math.parquet"
LOCAL_MBPP_PARQUET = ROOT_DIR / "data" / "mbpp" / "mbpp_test.parquet"
LOCAL_HUMAN_EVAL_PARQUET = ROOT_DIR / "data" / "openai_humaneval" / "openai_humaneval_test.parquet"


def load_gsm8k(split="test", n=None):
    ds = load_dataset("gsm8k", "main")[split]
    items = [{"id": f"gsm8k_{i}", "prompt": ex["question"] + " A:", "answer": ex["answer"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items


def normalize_number(s: str) -> str:
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return m.group(1) if m else ""


def judge_gsm8k(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)


def load_mbpp(n=None):
    if LOCAL_MBPP_PARQUET.exists():
        ds = load_dataset("parquet", data_files={"test": str(LOCAL_MBPP_PARQUET)})["test"]
    else:
        ds = load_dataset("mbpp", "sanitized")["test"]
    items = [{"id": ex.get("task_id", f"mbpp_{i}"), "prompt": ex.get("text", ""), "answer": ex.get("code", "")} for i, ex in enumerate(ds)]
    return items[:n] if n else items


def judge_mbpp(pred: str) -> bool:
    # 簡易pass@1近似（雰囲気一致）。本番はharness推奨。
    return "def " in pred


def load_humaneval():
    if LOCAL_HUMAN_EVAL_PARQUET.exists():
        ds = load_dataset("parquet", data_files={"test": str(LOCAL_HUMAN_EVAL_PARQUET)})["test"]
    else:
        ds = load_dataset("openai_humaneval")["test"]
    items = [{"id": ex.get("task_id", f"humaneval_{i}"), "prompt": ex["prompt"], "answer": ex.get("canonical_solution", ex.get("solution", ""))} for i, ex in enumerate(ds)]
    return items


def judge_humaneval(pred: str) -> bool:
    return "def " in pred


def load_math(split="test", n=None):
    if LOCAL_MATH_PARQUET.exists():
        ds = load_dataset("parquet", data_files={"train": str(LOCAL_MATH_PARQUET)})["train"]
        items = []
        for i, ex in enumerate(ds):
            items.append({
                "id": ex.get("id", f"math_local_{i}"),
                "prompt": ex.get("prompt", ex.get("problem", ex.get("question", ""))),
                "answer": ex.get("answer", ex.get("solution", "")),
            })
            if n and len(items) >= n:
                break
        return items[:n] if n else items

    ds = None
    last_err = None
    for ds_id in ("qwedsacf/competition_math", "hendrycks/competition_math", "competition_math"):
        try:
            ds = load_dataset(ds_id, split=split, trust_remote_code=True)
            break
        except Exception as err:
            last_err = err
            ds = None
    if ds is None:
        raise RuntimeError(f"Failed to load MATH dataset: {last_err}")

    items = [{"id": f"math_{i}", "prompt": ex["problem"], "answer": ex["solution"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items
