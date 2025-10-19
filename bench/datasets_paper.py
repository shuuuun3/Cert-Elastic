# bench/datasets_paper.py
from datasets import load_dataset
import os, re, json, logging

_HF_CACHE = (
    os.environ.get("HF_DATASETS_CACHE")
    or os.environ.get("HF_HOME")
    or os.environ.get("TRANSFORMERS_CACHE")
)

def _load_dataset(*args, **kwargs):
    """
    Wrapper that respects HF cache envs so Colab runs backed by Drive do not
    redownload large benchmarks every session.
    """
    if _HF_CACHE:
        kwargs.setdefault("cache_dir", _HF_CACHE)
    return load_dataset(*args, **kwargs)

def normalize_number(s: str) -> str:
    if not s: return ""
    m = re.search(r"(-?\d+(?:\.\d+)?)", s.replace(",", ""))
    return m.group(1) if m else ""

# ---- GSM8K
def load_gsm8k(split="test", n=None):
    ds = _load_dataset("gsm8k", "main")[split]
    items = [{"id": f"gsm8k_{i}", "prompt": ex["question"] + " A:", "answer": ex["answer"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items

def judge_gsm8k(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)

# ---- MATH (Hendrycks)
def load_math(split="test", n=None):
    """
    Hendrycks MATH は複数コンフィグ（algebra, geometry 等）に分割されているため、
    代表的な全カテゴリを順番に読み込み、結合して返す。
    Datasets のバージョンや命名差異に備え、データセットIDも冗長にフォールバック。
    """
    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    ds_ids = ["hendrycks/competition_math", "competition_math"]
    items = []
    for cfg in configs:
        loaded = None
        last_err = None
        for ds_id in ds_ids:
            try:
                loaded = _load_dataset(ds_id, cfg)[split]
                break
            except Exception as e:
                last_err = e
                loaded = None
        if loaded is None:
            # 1カテゴリ取得に失敗した場合はスキップ（最後に少なくとも1カテゴリあれば続行）
            logging.warning("MATH config %s unavailable via %s (last error: %s)", cfg, ds_ids, last_err)
            continue
        for i, ex in enumerate(loaded):
            items.append({
                "id": f"math_{cfg}_{i}",
                "prompt": ex.get("problem", ""),
                "answer": ex.get("solution", ""),
            })
        if n and len(items) >= n:
            break
    if not items:
        raise RuntimeError(
            "Failed to load any MATH categories. Ensure internet access or install datasets cache."
        )
    return items[:n] if n else items

def judge_math(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)

# ---- HumanEval（簡易。厳密pass@1が必要なら lm-eval を別途使用）
def load_humaneval(n=None):
    ds = _load_dataset("openai_humaneval")["test"]
    items = [{"id": ex["task_id"], "prompt": ex["prompt"], "answer": ex["canonical_solution"]} for ex in ds]
    return items[:n] if n else items

def judge_humaneval(ref: str, pred: str) -> bool:
    return "def " in pred  # 近似

# ---- MBPP（簡易）
def load_mbpp(n=None):
    ds = _load_dataset("mbpp", "sanitized")["test"]
    items = [{"id": f"mbpp_{i}", "prompt": ex["text"], "answer": ex["code"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items

def judge_mbpp(ref: str, pred: str) -> bool:
    return "def " in pred  # 近似
