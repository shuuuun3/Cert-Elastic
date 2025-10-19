import os, json, time, gc
from pathlib import Path
import torch

def make_run_dir(root="./runs"):
    t = time.strftime("%Y%m%d_%H%M%S")
    p = Path(root) / f"run_{t}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _json_default(o):
    # functions / callables
    try:
        if callable(o):
            return getattr(o, "__name__", str(o))
    except Exception:
        pass
    # pathlib.Path
    if isinstance(o, Path):
        return str(o)
    # torch types
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, torch.device):
        return str(o)
    if isinstance(o, torch.Tensor):
        try:
            return o.item()
        except Exception:
            return o.tolist()
    # sets / tuples
    if isinstance(o, (set, tuple)):
        return list(o)
    # numpy (optional)
    try:
        import numpy as np  # type: ignore
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    # fallback
    return str(o)

def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)

def cuda_clean():
    torch.cuda.empty_cache(); gc.collect()
