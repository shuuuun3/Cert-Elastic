from datasets import DatasetDict, Features, Sequence, Value, load_dataset, load_from_disk
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = ROOT_DIR / "data"
LOCAL_HF_ROOT = LOCAL_DATA_ROOT / "hf_cache"

LOCAL_GSM8K_PARQUETS = {
    "train": LOCAL_DATA_ROOT / "gsm8k" / "gsm8k_train.parquet",
    "test": LOCAL_DATA_ROOT / "gsm8k" / "gsm8k_test.parquet",
}
LOCAL_MBPP_PARQUETS = {
    "train": LOCAL_DATA_ROOT / "mbpp" / "mbpp_train.parquet",
    "validation": LOCAL_DATA_ROOT / "mbpp" / "mbpp_validation.parquet",
    "test": LOCAL_DATA_ROOT / "mbpp" / "mbpp_test.parquet",
    "prompt": LOCAL_DATA_ROOT / "mbpp" / "mbpp_prompt.parquet",
}
LOCAL_HUMAN_EVAL_PARQUET = LOCAL_DATA_ROOT / "openai_humaneval" / "openai_humaneval_test.parquet"
LOCAL_MATH_PARQUET = LOCAL_DATA_ROOT / "hendrycks_math" / "hendrycks_math.parquet"


def _load_from_disk(name: str) -> DatasetDict | None:
    path = LOCAL_HF_ROOT / name
    if path.exists():
        try:
            ds = load_from_disk(path)
            if isinstance(ds, DatasetDict):
                data = dict(ds.items())
                if "train" not in data and "test" in data:
                    data["train"] = data["test"]
                return DatasetDict(data)
        except Exception:
            pass
    return None


def _load_from_parquet(files: dict[str, Path], features: Features | None = None) -> DatasetDict | None:
    if not all(path.exists() for path in files.values()):
        return None
    data = {}
    for split, path in files.items():
        ds_raw = load_dataset("parquet", data_files={split: str(path)})[split]
        if features is not None:
            try:
                ds_raw = ds_raw.cast(features)
            except Exception:
                pass
        data[split] = ds_raw
    if "train" not in data and "test" in data:
        data["train"] = data["test"]
    return DatasetDict(data)


def _normalize_number(s: str) -> str:
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return m.group(1) if m else ""


def load_gsm8k(split="test", n=None):
    local = _load_from_disk("gsm8k_main") or _load_from_parquet(
        LOCAL_GSM8K_PARQUETS,
        Features({"question": Value("string"), "answer": Value("string")}),
    )
    ds = local[split] if local else load_dataset("gsm8k", "main")[split]
    items = [{"id": ex.get("id", f"gsm8k_{i}"), "prompt": ex.get("question", "") + " A:", "answer": ex.get("answer", "")} for i, ex in enumerate(ds)]
    return items[:n] if n else items


def judge_gsm8k(ref: str, pred: str) -> bool:
    return _normalize_number(ref) == _normalize_number(pred)


def load_mbpp(n=None):
    local = _load_from_disk("mbpp_sanitized") or _load_from_parquet(
        LOCAL_MBPP_PARQUETS,
        Features({
            "source_file": Value("string"),
            "task_id": Value("int32"),
            "prompt": Value("string"),
            "code": Value("string"),
            "test_imports": Sequence(Value("string")),
            "test_list": Sequence(Value("string")),
        }),
    )
    ds = local["test"] if local else load_dataset("mbpp", "sanitized")["test"]
    items = [{"id": ex.get("task_id", f"mbpp_{i}"), "prompt": ex.get("prompt", ex.get("text", "")), "answer": ex.get("code", "")} for i, ex in enumerate(ds)]
    return items[:n] if n else items


def judge_mbpp(pred: str) -> bool:
    return "def " in pred


def load_humaneval():
    local = _load_from_disk("openai_humaneval")
    if local is None and LOCAL_HUMAN_EVAL_PARQUET.exists():
        local = DatasetDict({
            "test": load_dataset("parquet", data_files={"test": str(LOCAL_HUMAN_EVAL_PARQUET)})["test"]
        })
    ds = local["test"] if local else load_dataset("openai_humaneval")["test"]
    items = [{"id": ex.get("task_id", f"humaneval_{i}"), "prompt": ex["prompt"], "answer": ex.get("canonical_solution", ex.get("solution", ""))} for i, ex in enumerate(ds)]
    return items


def judge_humaneval(pred: str) -> bool:
    return "def " in pred


def load_math(split="test", n=None):
    local = _load_from_disk("hendrycks_math")
    if local is None and LOCAL_MATH_PARQUET.exists():
        ds = load_dataset("parquet", data_files={"train": str(LOCAL_MATH_PARQUET)})["train"]
        local = DatasetDict({"train": ds, "test": ds})
    if local is None:
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
        items = [{"id": f"math_{i}", "prompt": ex.get("problem", ""), "answer": ex.get("solution", "")} for i, ex in enumerate(ds)]
        return items[:n] if n else items

    ds_split = local[split]
    items = [{
        "id": ex.get("id", f"math_local_{i}"),
        "prompt": ex.get("prompt", ex.get("problem", "")),
        "answer": ex.get("answer", ex.get("solution", "")),
    } for i, ex in enumerate(ds_split)]
    return items[:n] if n else items
