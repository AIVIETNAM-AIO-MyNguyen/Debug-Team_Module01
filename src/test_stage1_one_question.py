import pandas as pd

from retrievers import DenseRetriever

df = pd.read_parquet(
    "data/processed/questions/questions_with_chunk_ids.parquet"
)

row = df.iloc[0]

question = row["question"]

ground_truth = row["ground_truth_chunk_ids"]["c_512"]

print("Question:")
print(question)

print("\nGround Truth:")
print(ground_truth)

retriever = DenseRetriever(
    "data/processed/embeddings",
    "c_512"
)

results = retriever.search(
    question,
    k=5
)

retrieved_ids = results["ids"][0]

print("\nRetrieved:")
print(retrieved_ids)

hit = any(
    chunk_id in retrieved_ids
    for chunk_id in ground_truth
)

print("\nHitRate@5:")
print(float(hit))