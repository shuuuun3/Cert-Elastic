from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset


DATA_DIR = Path("data")
HF_DATA_DIR = DATA_DIR / "hf_cache"


def _ensure_dir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def save_split(ds, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out_path)


def save_dataset_dict(name: str, ds_dict: DatasetDict) -> None:
    out_dir = HF_DATA_DIR / name
    _ensure_dir_clean(out_dir)
    ds_dict.save_to_disk(out_dir)


def _dataset_present(parquet_files: List[Path], hf_dir: Path) -> bool:
    parquet_ready = all(p.exists() for p in parquet_files)
    hf_ready = hf_dir.exists()
    return parquet_ready and hf_ready


def _download_with_guard(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to prepare dataset '{name}': {exc}") from exc


def download_gsm8k() -> None:
    parquet_targets = [
        DATA_DIR / "gsm8k" / "gsm8k_train.parquet",
        DATA_DIR / "gsm8k" / "gsm8k_test.parquet",
    ]
    if _dataset_present(parquet_targets, HF_DATA_DIR / "gsm8k_main"):
        print("[dataset] gsm8k already prepared. skipping download.")
        return

    ds = load_dataset("gsm8k", "main")
    out_dir = DATA_DIR / "gsm8k"
    for split_name, split in ds.items():
        save_split(split, out_dir / f"gsm8k_{split_name}.parquet")
    save_dataset_dict("gsm8k_main", ds)
    print("[dataset] gsm8k prepared.")


def download_mbpp() -> None:
    parquet_targets = [
        DATA_DIR / "mbpp" / f"mbpp_{split}.parquet"
        for split in ("train", "validation", "test", "prompt")
    ]
    if _dataset_present(parquet_targets, HF_DATA_DIR / "mbpp_sanitized"):
        print("[dataset] mbpp already prepared. skipping download.")
        return

    ds = load_dataset("mbpp", "sanitized")
    out_dir = DATA_DIR / "mbpp"
    for split_name, split in ds.items():
        save_split(split, out_dir / f"mbpp_{split_name}.parquet")
    save_dataset_dict("mbpp_sanitized", ds)
    print("[dataset] mbpp prepared.")


def download_humaneval() -> None:
    parquet_path = DATA_DIR / "openai_humaneval" / "openai_humaneval_test.parquet"
    hf_dir = HF_DATA_DIR / "openai_humaneval"
    if _dataset_present([parquet_path], hf_dir):
        print("[dataset] openai_humaneval already prepared. skipping download.")
        return

    ds = load_dataset("openai_humaneval")
    out_dir = DATA_DIR / "openai_humaneval"
    save_split(ds["test"], parquet_path)
    save_dataset_dict("openai_humaneval", ds)
    print("[dataset] openai_humaneval prepared.")


def _iter_math_sources() -> Iterable[Tuple[str, dict]]:
    yield "qwedsacf/competition_math", {}
    yield "hendrycks/competition_math", {}
    yield "competition_math", {}


def download_hendrycks_math() -> None:
    parquet_path = DATA_DIR / "hendrycks_math" / "hendrycks_math.parquet"
    hf_dir = HF_DATA_DIR / "hendrycks_math"
    if _dataset_present([parquet_path], hf_dir):
        print("[dataset] hendrycks_math already prepared. skipping download.")
        return

    dataset = None
    last_err: Optional[Exception] = None
    for ds_id, kwargs in _iter_math_sources():
        try:
            loaded = load_dataset(ds_id, **kwargs)
            if isinstance(loaded, DatasetDict):
                dataset = loaded.get("train") or loaded.get("test")
            else:
                dataset = loaded
            if dataset is not None:
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            dataset = None
    if dataset is None:
        raise RuntimeError(f"Failed to download hendrycks_math dataset: {last_err}")

    records = []
    for i, ex in enumerate(dataset):
        records.append({
            "id": ex.get("id", f"math_{i}"),
            "prompt": ex.get("problem", ex.get("question", "")),
            "answer": ex.get("solution", ex.get("answer", "")),
            "category": ex.get("category", ex.get("type", "")),
        })

    features = Features({
        "id": Value("string"),
        "prompt": Value("string"),
        "answer": Value("string"),
        "category": Value("string"),
    })
    math_dataset = Dataset.from_list(records, features=features)
    save_split(math_dataset, parquet_path)
    save_dataset_dict("hendrycks_math", DatasetDict({"train": math_dataset, "test": math_dataset}))
    print("[dataset] hendrycks_math prepared.")


def main() -> None:
    _download_with_guard("gsm8k", download_gsm8k)
    _download_with_guard("mbpp", download_mbpp)
    _download_with_guard("openai_humaneval", download_humaneval)
    _download_with_guard("hendrycks_math", download_hendrycks_math)


if __name__ == "__main__":
    main()
