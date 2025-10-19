from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset


DATA_DIR = Path("data")


def save_split(ds, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out_path)


def download_mbpp() -> None:
    mbpp = load_dataset("mbpp", "sanitized")
    out_dir = DATA_DIR / "mbpp"
    for split_name, split in mbpp.items():
        save_split(split, out_dir / f"mbpp_{split_name}.parquet")


def download_humaneval() -> None:
    humaneval = load_dataset("openai_humaneval")
    out_dir = DATA_DIR / "openai_humaneval"
    save_split(humaneval["test"], out_dir / "openai_humaneval_test.parquet")


def _iter_math_sources() -> Iterable[tuple[str, dict]]:
    yield "qwedsacf/competition_math", {}
    yield "hendrycks/competition_math", {}
    yield "competition_math", {}


def download_hendrycks_math() -> None:
    dataset = None
    last_err: Exception | None = None
    for ds_id, kwargs in _iter_math_sources():
        try:
            loaded = load_dataset(ds_id, **kwargs)
            if isinstance(loaded, dict):
                dataset = loaded.get("train") or next(iter(loaded.values()))
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

    out_dir = DATA_DIR / "hendrycks_math"
    save_split(Dataset.from_list(records), out_dir / "hendrycks_math.parquet")


def main() -> None:
    download_mbpp()
    download_humaneval()
    download_hendrycks_math()


if __name__ == "__main__":
    main()
