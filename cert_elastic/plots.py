import pandas as pd
import matplotlib.pyplot as plt

def plot_safe_ratio_per_step(df: pd.DataFrame, out_png):
    step_mean = df.groupby("step").mean(numeric_only=True).reset_index()
    plt.figure(figsize=(8,4))
    plt.plot(step_mean["step"], step_mean["safe_ratio"])
    plt.xlabel("step"); plt.ylabel("safe_ratio (f<=ε)")
    plt.title("Cert-Elastic safety ratio per step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)

def plot_mean_f_by_layer(df: pd.DataFrame, out_png):
    layer_mean = df.groupby("layer").mean(numeric_only=True).reset_index()
    plt.figure(figsize=(8,4))
    plt.plot(layer_mean["layer"], layer_mean["f"])
    plt.xlabel("layer"); plt.ylabel("mean f")
    plt.title("Cert-Elastic mean f by layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
