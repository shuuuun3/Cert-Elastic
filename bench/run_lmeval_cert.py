# bench/run_lmeval_cert.py（先頭～mainのtasks処理のみ重要差分）
import argparse, json
from pathlib import Path
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from bench.lmeval_cert_runner import HFCertElasticLM
from cert_elastic.utils import make_run_dir, dump_json
import os
import datasets
from datasets import DatasetDict, Features, Sequence, Value

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

ALIASES = {
    "humaneval": ["openai_humaneval", "humaneval", "human_eval", "humaneval_python"],
    "mbpp": ["mbpp", "mbpp_sanitized", "mbppplus", "mbpp_plus"],
    "gsm8k": ["gsm8k"],
    "hendrycks_math": ["hendrycks_math", "math"],
}

LOCAL_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
LOCAL_GSM8K_PARQUETS = {
    "train": LOCAL_DATA_ROOT / "gsm8k" / "gsm8k_train.parquet",
    "test": LOCAL_DATA_ROOT / "gsm8k" / "gsm8k_test.parquet",
}
LOCAL_MBPP_PATHS = {
    "train": LOCAL_DATA_ROOT / "mbpp" / "mbpp_train.parquet",
    "validation": LOCAL_DATA_ROOT / "mbpp" / "mbpp_validation.parquet",
    "test": LOCAL_DATA_ROOT / "mbpp" / "mbpp_test.parquet",
    "prompt": LOCAL_DATA_ROOT / "mbpp" / "mbpp_prompt.parquet",
}
LOCAL_HF_DATASETS = {
    "gsm8k": LOCAL_DATA_ROOT / "hf_cache" / "gsm8k_main",
    "mbpp": LOCAL_DATA_ROOT / "hf_cache" / "mbpp_sanitized",
    "openai_humaneval": LOCAL_DATA_ROOT / "hf_cache" / "openai_humaneval",
    "hendrycks_math": LOCAL_DATA_ROOT / "hf_cache" / "hendrycks_math",
}

LOCAL_HUMAN_EVAL_PARQUET = LOCAL_DATA_ROOT / "openai_humaneval" / "openai_humaneval_test.parquet"
LOCAL_MATH_PARQUET = LOCAL_DATA_ROOT / "hendrycks_math" / "hendrycks_math.parquet"

_ORIGINAL_LOAD_DATASET = datasets.load_dataset


def _load_from_disk_dataset(key: str):
    path = LOCAL_HF_DATASETS.get(key)
    if path and path.exists():
        ds = datasets.load_from_disk(path)
        if isinstance(ds, DatasetDict):
            data = {k: v for k, v in ds.items()}
            if "train" not in data and "test" in data:
                data["train"] = data["test"]
            return DatasetDict(data)
    return None


def _load_parquet_dataset(splits_map, features):
    data = {}
    for split, path in splits_map.items():
        if not path.exists():
            return None
    for split, path in splits_map.items():
        ds_raw = _ORIGINAL_LOAD_DATASET("parquet", data_files={split: str(path)})[split]
        try:
            ds = ds_raw.cast(features) if features is not None else ds_raw
        except Exception:
            ds = ds_raw
        data[split] = ds
    return DatasetDict(data)


def _local_gsm8k():
    ds_disk = _load_from_disk_dataset("gsm8k")
    if ds_disk is not None:
        return ds_disk
    feats = Features({"question": Value("string"), "answer": Value("string")})
    return _load_parquet_dataset(LOCAL_GSM8K_PARQUETS, feats)


def _local_mbpp():
    ds_disk = _load_from_disk_dataset("mbpp")
    if ds_disk is not None:
        return ds_disk
    feats = Features({
        "source_file": Value("string"),
        "task_id": Value("int32"),
        "prompt": Value("string"),
        "code": Value("string"),
        "test_imports": Sequence(Value("string")),
        "test_list": Sequence(Value("string")),
    })
    return _load_parquet_dataset(LOCAL_MBPP_PATHS, feats)


def _local_humaneval():
    ds_disk = _load_from_disk_dataset("openai_humaneval")
    if ds_disk is not None:
        return ds_disk
    if not LOCAL_HUMAN_EVAL_PARQUET.exists():
        return None
    feats = Features({
        "task_id": Value("string"),
        "prompt": Value("string"),
        "canonical_solution": Value("string"),
        "test": Value("string"),
        "entry_point": Value("string"),
    })
    ds = _ORIGINAL_LOAD_DATASET("parquet", data_files={"test": str(LOCAL_HUMAN_EVAL_PARQUET)})["test"].cast(feats)
    return DatasetDict({"test": ds})


def _local_math():
    ds_disk = _load_from_disk_dataset("hendrycks_math")
    if ds_disk is not None:
        return ds_disk
    if not LOCAL_MATH_PARQUET.exists():
        return None
    feats = Features({
        "id": Value("string"),
        "prompt": Value("string"),
        "answer": Value("string"),
        "category": Value("string"),
    })
    ds = _ORIGINAL_LOAD_DATASET("parquet", data_files={"test": str(LOCAL_MATH_PARQUET)})["test"].cast(feats)
    return DatasetDict({"test": ds, "train": ds})


def _local_dataset_for(target: str):
    key = target.lower()
    if key == "gsm8k":
        return _local_gsm8k()
    if key in ("openai_humaneval", "humaneval", "human_eval", "humaneval_python"):
        return _local_humaneval()
    if key in ("mbpp", "mbpp_sanitized", "mbppplus", "mbpp_plus"):
        return _local_mbpp()
    if key in ("hendrycks_math", "math"):
        return _local_math()
    return None


def enable_local_datasets():
    if getattr(enable_local_datasets, "_patched", False):
        return

    def wrapper(path_or_name=None, *args, **kwargs):
        args = list(args)
        if path_or_name is None and "path_or_name" in kwargs:
            path_or_name = kwargs.pop("path_or_name")
        if path_or_name is None and "path" in kwargs:
            path_or_name = kwargs.pop("path")
        if path_or_name is None and "dataset" in kwargs:
            path_or_name = kwargs.pop("dataset")
        if path_or_name is None and args:
            path_or_name = args.pop(0)
        if path_or_name is None:
            raise TypeError("datasets.load_dataset wrapper requires path_or_name")

        target = path_or_name.lower() if isinstance(path_or_name, str) else path_or_name
        split = kwargs.get("split")
        try:
            return _ORIGINAL_LOAD_DATASET(path_or_name, *args, **kwargs)
        except Exception as remote_err:  # noqa: BLE001
            local_ds = _local_dataset_for(target) if isinstance(target, str) else None
            if local_ds is None:
                raise
            print(f"[datasets] remote load failed for {path_or_name!r}; falling back to local cache: {remote_err}")
            if split:
                if split in local_ds:
                    return local_ds[split]
                if split == "train" and "test" in local_ds:
                    return local_ds["test"]
                raise KeyError(f"Local dataset missing requested split '{split}'.")
            return local_ds

    datasets.load_dataset = wrapper
    enable_local_datasets._patched = True


def resolve_tasks(user_csv: str) -> list[str]:
    tm = TaskManager()
    available = set(tm.task_index.keys())
    out = []
    for raw in [t.strip() for t in user_csv.split(",") if t.strip()]:
        # 完全一致を優先
        if raw in available:
            out.append(raw); continue
        # エイリアス解決
        cands = ALIASES.get(raw.lower(), [raw.lower()])
        hit = next((c for c in cands if c in available), None)
        if hit is None:
            raise SystemExit(
                f"[error] task '{raw}' not found. available example hits: "
                f"{[t for t in available if any(a in t for a in cands[:2])] or list(sorted(available))[:20]}"
            )
        out.append(hit)
    print("[tasks]", out)
    return out

def pick_primary(name, resdict):
    name = name.split(":")[0].lower()
    for k in ({"gsm8k":"exact_match","hendrycks_math":"exact_match",
               "humaneval":"pass@1","mbpp":"pass@1"}.get(name), "exact_match","pass@1","acc","accuracy"):
        if k and k in resdict and isinstance(resdict[k], (int,float)):
            return k, float(resdict[k])
    for k,v in resdict.items():
        if isinstance(v,(int,float)): return k, float(v)
    return None, None

def run_once(enable_cert: bool, args):
    enable_local_datasets()
    lm = HFCertElasticLM(
        pretrained=args.model_id, dtype=args.dtype, device=(args.device or "auto"),
        enable_cert=enable_cert, epsilon=args.epsilon, alpha=args.alpha, beta=args.beta,
        attn_impl=args.attn_impl, batch_size=1, trust_remote_code=True,
    )
    tasks = resolve_tasks(args.tasks)
    import sys
    if sys.platform == "win32":
        tasks = [t for t in tasks if t not in ("humaneval","openai_humaneval","mbpp","mbpp_sanitized")]
        print("[win32] code-eval 未対応のため HumanEval/MBPP を除外:", tasks)
    # Allow running tasks marked as "unsafe" (e.g. mbpp) by confirming explicitly.
    # lm_eval will execute task code when these tasks require code evaluation.
    res = evaluator.simple_evaluate(model=lm, tasks=tasks, num_fewshot=args.fewshot,
                                    limit=args.limit, bootstrap_iters=0,
                                    confirm_run_unsafe_code=True)
    ours = {}
    for tname, tres in res["results"].items():
        k, v = pick_primary(tname, tres)
        ours[tname.lower()] = {"metric_key": k, "accuracy": v}
    toks = (lm.total_new_tokens / lm.total_gen_time) if lm.total_gen_time>0 else None
    return {"results_raw": res, "ours_primary": ours, "tokens_sec": toks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--attn_impl", default="eager")
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--tasks", default="gsm8k,humaneval,mbpp,hendrycks_math")
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out_dir", default="./runs")
    args = ap.parse_args()

    run_dir = make_run_dir(args.out_dir)
    base = run_once(False, args); dump_json(base, Path(run_dir/"lmeval_baseline.json"))
    cert = run_once(True,  args); dump_json(cert, Path(run_dir/"lmeval_cert.json"))
    out_for_viz = {k: {"accuracy": v["accuracy"], "tokens_sec": cert["tokens_sec"]}
                   for k,v in cert["ours_primary"].items()}
    dump_json(out_for_viz, Path(run_dir/"results.certelastic.paperstyle.json"))
    print("[done]", Path(run_dir).resolve())

if __name__ == "__main__":
    main()
