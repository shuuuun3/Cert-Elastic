"""
Run a local end-to-end pipeline with a moderate dataset size:
- Build a small GSM8K JSONL subset
- Run bench/run_experiment.py (baseline vs cert)
- Produce simple comparison plots (throughput / exact match)

Usage (PowerShell):
  python .\run_local_pipeline.py --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n 50 --max_new_tokens 64

Notes:
- TinyLlama (1.1B) defaults to keep VRAM footprint low on Windows without bitsandbytes.
- You can switch to Mistral-7B if your environment allows it (e.g., Linux + 4bit).
"""
import argparse
from pathlib import Path
import time
import json
import re

def make_gsm8k_subset_jsonl(out_path: Path, n: int = 50, seed: int = 42):
    from datasets import load_dataset
    import random
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("gsm8k", "main")["test"]
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]

    def extract_number(ans: str) -> str:
        # GSM8K answers often contain final numeric answer like "#### 123"; fallback to last number in string
        m = re.findall(r"(-?\d+(?:\.\d+)?)", ans)
        return m[-1] if m else ans.strip()

    with out_path.open("w", encoding="utf-8") as f:
        for i in idxs:
            ex = ds[int(i)]
            q = ex["question"].strip()
            a = extract_number(ex.get("answer", "").strip())
            rec = {"id": f"gsm8k_{i}", "prompt": q + " A:", "reference": a}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path

def find_latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    cands = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--device", default=None, help="e.g. cuda:0; leave None to use loader device_map")
    ap.add_argument("--n", type=int, default=50, help="GSM8K subset size")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--out_dir", default="./runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs_root = Path(args.out_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    # 1) Create dataset subset
    data_path = Path("data")/f"gsm8k_subset_{args.n}.jsonl"
    if not data_path.exists():
        print(f"[dataset] creating {data_path} ...")
        make_gsm8k_subset_jsonl(data_path, n=args.n, seed=args.seed)
    else:
        print(f"[dataset] reuse {data_path}")

    # 2) Run experiment (baseline vs cert)
    import subprocess, sys
    cmd = [
        sys.executable, "-m", "bench.run_experiment",
        "--dataset", str(data_path),
        "--model_id", args.model_id,
        "--dtype", args.dtype,
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--epsilon", str(args.epsilon),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--out_dir", str(args.out_dir),
        "--run_cert", "1",
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("[run] ", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, check=False, text=True)
    t1 = time.time()
    if r.returncode != 0:
        raise SystemExit(f"run_experiment failed with code {r.returncode}")
    print(f"[run] finished in {t1 - t0:.1f}s")

    # 3) Find latest run dir and plot comparison
    run_dir = find_latest_run_dir(runs_root)
    if run_dir is None:
        raise SystemExit("No run directory found under ./runs")
    summary_path = run_dir/"summary_compare.json"
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")

    # Dynamic import to avoid PYTHONPATH issues
    import importlib.util
    mod_path = Path(__file__).parent / "viz" / "plot_bench_summary.py"
    spec = importlib.util.spec_from_file_location("plot_bench_summary", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "failed to load plot module"
    spec.loader.exec_module(mod)
    print(f"[viz] generating plots into {run_dir} ...")
    mod.make_plots_from_summary(summary_path, out_dir=run_dir)
    print("[done] artifacts:")
    print(" - ", summary_path)
    print(" - ", run_dir/"throughput_compare.png")
    print(" - ", run_dir/"exact_match_compare.png")

if __name__ == "__main__":
    main()
