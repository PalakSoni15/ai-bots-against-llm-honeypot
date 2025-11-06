#!/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "extended_outputs"

def save_arena_heatmap():
    path = os.path.join(OUTDIR, "arena_by_model.csv")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Skipping arena heatmap.")
        return
    df = pd.read_csv(path)
    if df.empty or "model_name" not in df.columns:
        print("[WARN] arena_by_model.csv is empty or malformed.")
        return
    top = df.sort_values("runs", ascending=False).head(20).copy()
    metrics = ["refusal_rate", "potential_leak_rate"]
    data = top[metrics].to_numpy().T  # (2, N)

    plt.figure(figsize=(10, 3))
    plt.imshow(data, aspect="auto")  # no explicit cmap
    plt.yticks(range(len(metrics)), metrics)
    plt.xticks(range(len(top)), top["model_name"], rotation=60, ha="right")
    plt.title("Arena (Top 20): Refusal & Potential-Leak Rates")
    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "arena_heatmap.png")
    plt.savefig(outpath, dpi=160)
    print(f"[OK] wrote {outpath}")

def save_honeypot_heatmap():
    path = os.path.join(OUTDIR, "honeypot_by_bot.csv")
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Skipping honeypot heatmap.")
        return
    df = pd.read_csv(path)
    if df.empty or "bot_name" not in df.columns:
        print("[WARN] honeypot_by_bot.csv is empty or malformed.")
        return
    df = df.sort_values(["bot_name","model_name"]).copy()
    labels = df.apply(lambda r: f"{r['bot_name']} ({r['model_name']})", axis=1)
    metrics = ["leak_rate", "refusal_rate"]
    data = df[metrics].to_numpy().T  # (2, M)

    plt.figure(figsize=(8, 3))
    plt.imshow(data, aspect="auto")  # no explicit cmap
    plt.yticks(range(len(metrics)), metrics)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title("Honeypot: Leak & Refusal Rates by Bot/Model")
    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "honeypot_heatmap.png")
    plt.savefig(outpath, dpi=160)
    print(f"[OK] wrote {outpath}")

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    save_arena_heatmap()
    save_honeypot_heatmap()

if __name__ == "__main__":
    main()
