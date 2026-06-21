import pandas as pd

from hybrid_retriever import HybridRetriever
from metrics import (
    hit_rate_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k
)

questions = pd.read_parquet(
    "data/processed/questions/questions_with_chunk_ids.parquet"
)

retriever = HybridRetriever(
    chroma_path="data/processed/embeddings",
    collection_name="c_512"
)

total_hitrate = 0
total_recall = 0
total_mrr = 0
total_ndcg = 0

num_questions = 0

for _, row in questions.iterrows():

    question = row["question"]

    ground_truth = row["ground_truth_chunk_ids"]["c_512"]

    if not ground_truth:
        continue

    retrieved_ids = retriever.search(
        question,
        k=5
    )

    total_hitrate += hit_rate_at_k(
        retrieved_ids,
        ground_truth
    )

    total_recall += recall_at_k(
        retrieved_ids,
        ground_truth
    )

    total_mrr += mrr(
        retrieved_ids,
        ground_truth
    )

    total_ndcg += ndcg_at_k(
        retrieved_ids,
        ground_truth
    )

    num_questions += 1

print(f"Questions: {num_questions}")
print()

print(
    f"HitRate@5: "
    f"{total_hitrate / num_questions:.4f}"
)

print(
    f"Recall@5: "
    f"{total_recall / num_questions:.4f}"
)

print(
    f"MRR: "
    f"{total_mrr / num_questions:.4f}"
)

print(
    f"NDCG@5: "
    f"{total_ndcg / num_questions:.4f}"
)