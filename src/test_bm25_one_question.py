import pandas as pd

from bm25_retriever import BM25Retriever


questions = pd.read_parquet(
    "data/processed/questions/questions_with_chunk_ids.parquet"
)

row = questions.iloc[0]

question = row["question"]

ground_truth = row["ground_truth_chunk_ids"]["c_512"]

retriever = BM25Retriever(
    chroma_path="data/processed/embeddings",
    collection_name="c_512"
)

results = retriever.search(
    question,
    k=5
)

retrieved_ids = [
    r["chunk_id"]
    for r in results
]

print()

print("Question:")
print(question)

print()

print("Ground Truth:")
print(ground_truth)

print()

print("Retrieved:")
print(retrieved_ids)

print()

hit = any(
    chunk_id in retrieved_ids
    for chunk_id in ground_truth
)

print("HitRate@5:")
print(float(hit))