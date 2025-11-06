import pandas as pd, json
df = pd.read_parquet(r'.\arena_data\train-00000-of-00001-cced8514c7ed782a.parquet')
print("COLUMNS:", list(df.columns))
print("\nDTYPES:\n", df.dtypes)
print("\nFIRST ROW (truncated fields to 500 chars):")
row = df.iloc[0].to_dict()
for k,v in row.items():
    s = str(v)
    if len(s) > 500: s = s[:500] + "...[truncated]"
    print(f"- {k}: {s}")
