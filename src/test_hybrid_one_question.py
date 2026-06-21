import pandas as pd

from hybrid_retriever import HybridRetriever

questions = pd.read_parquet(
    "data/processed/questions/questions_with_chunk_ids.parquet"
)

row = questions.iloc[0]

question = row["question"]

ground_truth = row["ground_truth_chunk_ids"]["c_512"]

retriever = HybridRetriever(
    chroma_path="data/processed/embeddings",
    collection_name="c_512"
)

results = retriever.search(
    question,
    k=5
)

print("Ground Truth:")
print(ground_truth)

print()

print("Retrieved:")
print(results)