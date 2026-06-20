import pandas as pd

df = pd.read_parquet(
    "data/raw/confluence_questions.parquet"
)

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst row:")
print(df.iloc[0])