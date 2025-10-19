# cert_elastic/estimate_speedup.py
import pandas as pd
import json
from pathlib import Path
import numpy as np

run_dir = Path("runs") / sorted([p.name for p in Path("runs").iterdir() if p.is_dir() and p.name.startswith("run_")])[-1]
csv = run_dir / "cert_checks.csv"
cfg = run_dir / "config.eval.json"

df = pd.read_csv(csv)
cfgd = json.loads(cfg.read_text()) if cfg.exists() else {}
epsilon = float(cfgd.get("epsilon", 0.02))

# safe column (f<=epsilon)
if "safe" not in df.columns:
    df["safe"] = (df["f"] <= epsilon).astype(int)

num_layers = int(df["layer"].nunique())
# per-check grouping: assume a check is identified by (prompt_id, step) if present
if "prompt_id" in df.columns and "step" in df.columns:
    group_cols = ["prompt_id","step"]
else:
    # fallback: aggregate across checks uniformly
    group_cols = ["layer"]  # will compute differently below

# compute average skipped layers per check
if set(group_cols).issubset(df.columns):
    grouped = df.groupby(group_cols)
    skipped_per_check = grouped["safe"].sum().values  # number of layers safe (=skippable) per check
    avg_skipped_layers = float(np.mean(skipped_per_check))
else:
    # fallback: use overall safe fraction * num_layers
    avg_skipped_layers = df["safe"].mean() * num_layers

skip_fraction = avg_skipped_layers / num_layers  # fraction of layers skipped per token
# naive speedup assuming cost ~ sum(layer costs) and skip reduces proportionally
if skip_fraction >= 0.9999:
    est_speedup = 100.0
else:
    est_speedup = 1.0 / max(1e-6, (1.0 - skip_fraction))

out = {
    "run_dir": str(run_dir.resolve()),
    "epsilon": epsilon,
    "num_layers": num_layers,
    "avg_skipped_layers": avg_skipped_layers,
    "skip_fraction": skip_fraction,
    "estimated_speedup_naive": est_speedup,
    "notes": "Assumes equal cost per layer. For realistic estimate weigh layers by FLOPs."
}
print(json.dumps(out, indent=2, ensure_ascii=False))
Path(run_dir / "estimated_speedup.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
