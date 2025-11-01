#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main(csv_path="llm_honeypot_logs.csv"):
    p = Path(csv_path)
    if not p.exists():
        print(f"CSV not found at {csv_path}")
        return

    df = pd.read_csv(p)
    # Normalize boolean-like fields
    for col in ["leaked_SECRET_ALPHA","leaked_SECRET_BETA","leaked_SECRET_GAMMA","explicit_refusal","has_browsing"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["1","true","yes","y"])

    df["any_leak"] = df[["leaked_SECRET_ALPHA","leaked_SECRET_BETA","leaked_SECRET_GAMMA"]].any(axis=1)

    # Summary by bot/model
    by_bot = df.groupby(["bot_name","model_name"], dropna=False).agg(
        runs=("prompt_id","count"),
        leaks=("any_leak","sum"),
        refusals=("explicit_refusal","sum")
    ).reset_index()
    by_bot["leak_rate"] = (by_bot["leaks"] / by_bot["runs"]).round(3)
    by_bot["refusal_rate"] = (by_bot["refusals"] / by_bot["runs"]).round(3)

    print("=== Summary by Bot / Model ===")
    print(by_bot.sort_values(["leak_rate","runs"], ascending=[False, False]).to_string(index=False))

    # Aggregate by prompt
    by_prompt = df.groupby("prompt_id").agg(
        runs=("bot_name","count"),
        leaks=("any_leak","sum")
    ).reset_index()
    by_prompt["leak_rate"] = (by_prompt["leaks"] / by_prompt["runs"]).round(3)
    print("\n=== Summary by Prompt ===")
    print(by_prompt.sort_values("leak_rate", ascending=False).to_string(index=False))

    # Plot leak rate by bot_name (aggregated across models)
    leak_by_bot = df.groupby("bot_name").agg(any_leak=("any_leak","mean")).reset_index()
    plt.figure()
    plt.bar(leak_by_bot["bot_name"].astype(str), leak_by_bot["any_leak"])
    plt.title("Leak Rate by Bot (fraction of runs leaking any canary)")
    plt.xlabel("Bot")
    plt.ylabel("Leak Rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("leak_rate_by_bot.png", dpi=160)
    print("\nSaved plot to leak_rate_by_bot.png")

if __name__ == "__main__":
    main()
