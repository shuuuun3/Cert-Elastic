"""
Simple plotting utilities for bench/run_experiment.py summary output.

Generates:
- throughput_compare.png (avg tok/s baseline vs cert)
- exact_match_compare.png (if references present)
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt

def make_plots_from_summary(summary_path: Path | str, out_dir: Path | str | None = None):
    summary_path = Path(summary_path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if out_dir is None:
        out_dir = summary_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Throughput
    base = data.get("baseline", {})
    cert = data.get("cert", {})
    x = ["baseline", "cert"]
    y = [base.get("avg_tok_per_sec"), cert.get("avg_tok_per_sec")]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(x, y, color=["#6baed6", "#74c476"])  # blue, green
    ax.set_ylabel("avg tokens/sec")
    ax.set_title("Throughput: baseline vs cert-elastic")
    for i, v in enumerate(y):
        if v is not None:
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(Path(out_dir, "throughput_compare.png"), dpi=160)
    plt.close(fig)

    # Exact match (optional)
    def pick_em(d):
        em = None
        if d and isinstance(d.get("eval_metrics"), dict):
            em = d["eval_metrics"].get("exact_match")
        return em
    em_b = pick_em(base)
    em_c = pick_em(cert)
    if em_b is not None or em_c is not None:
        x = ["baseline", "cert"]
        y = [em_b, em_c]
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(x, y, color=["#9ecae1", "#a1d99b"])  # lighter
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exact Match")
        ax.set_title("Accuracy: baseline vs cert-elastic")
        for i, v in enumerate(y):
            if v is not None:
                ax.text(i, v, f"{v*100:.1f}%", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(Path(out_dir, "exact_match_compare.png"), dpi=160)
        plt.close(fig)
