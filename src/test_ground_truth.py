import pandas as pd

df = pd.read_parquet(
    "data/raw/confluence_questions.parquet"
)

print(df.iloc[0]["source_document_ids"])
print()
print(df.iloc[0]["ground_truth_answer"])