# viz/make_tables_and_plots.py
import argparse, json, yaml, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_template", required=True)   # 論文の表値（手入力YAML）
    ap.add_argument("--our_results", required=True)      # run_certelastic_paperstyle のJSON
    ap.add_argument("--out_dir", default="runs/figs")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    paper = yaml.safe_load(open(args.paper_template, "r", encoding="utf-8"))
    ours  = json.load(open(args.our_results, "r", encoding="utf-8"))

    # 表名 → 我々の結果キーのエイリアス
    alias = {
        "GSM8K": ["gsm8k"],
        "HumanEval": ["humaneval", "openai_humaneval", "humaneval_python", "human_eval"],
        "MATH": ["hendrycks_math", "math"],
        "MBPP": ["mbpp", "mbpp_sanitized", "mbppplus", "mbpp_plus"],
    }

    def lookup_ours(bench_name: str):
        # 完全一致（別名候補）
        for k in alias.get(bench_name, []):
            if k in ours:
                return ours[k]
        # 小文字の部分一致
        bn = bench_name.lower()
        for k, v in ours.items():
            if bn in k:
                return v
        return None

    md_all = ["# Benchmarks with Cert-Elastic (Ours)\n"]
    for bench, tab in paper["tables"].items():
        md = [f"## {bench}",
              "| GenLen | LLaDA acc/tps | Fast-dLLM acc/tps | Cert-Elastic (Ours) acc/tps |",
              "| --- | --- | --- | --- |"]
        rows = []
        for row in tab["rows"]:
            genlen = int(row["gen_len"])
            lld = row["LLaDA"]; fdl = row["Fast-dLLM"]
            ours_row = lookup_ours(bench)
            acc_o = round(ours_row["accuracy"]*100, 2) if (ours_row and ours_row.get("accuracy") is not None) else "-"
            tps_o = round(ours_row["tokens_sec"], 1) if (ours_row and ours_row.get("tokens_sec") is not None) else "-"
            md.append(f"| {genlen} | {lld['acc']} / {lld['tps']} | {fdl['acc']} / {fdl['tps']} | {acc_o} / {tps_o} |")
            rows.append({
                "GenLen": genlen,
                "LLaDA": lld["tps"],
                "Fast-dLLM": fdl["tps"],
                "Cert-Elastic": (ours_row or {}).get("tokens_sec", 0.0) or 0.0,
            })

        Path(args.out_dir, f"{bench}_table.md").write_text("\n".join(md)+ "\n", encoding="utf-8")

        # 図（throughput）
        df = pd.DataFrame(rows)
        ax = df.set_index("GenLen")[ ["LLaDA","Fast-dLLM","Cert-Elastic"] ].plot(kind="bar")
        ax.set_ylabel("tokens/sec"); ax.set_title(f"{bench}: throughput")
        fig = ax.get_figure(); fig.tight_layout()
        fig.savefig(Path(args.out_dir, f"{bench}_throughput.png")); plt.close(fig)

        # 全体まとめにも追記
        md_all.extend(md)

    Path(args.out_dir, "table_with_cert.md").write_text("\n".join(md_all)+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
