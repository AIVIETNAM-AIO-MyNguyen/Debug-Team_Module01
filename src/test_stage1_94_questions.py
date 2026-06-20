import pandas as pd

from retrievers import DenseRetriever

from metrics import (
    hit_rate_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k
)

# =========================
# LOAD DATA
# =========================

df = pd.read_parquet(
    "data/processed/questions/questions_with_chunk_ids.parquet"
)

print("Questions:", len(df))

# =========================
# RETRIEVER
# =========================

retriever = DenseRetriever(
    chroma_path="data/processed/embeddings",
    collection_name="c_512"
)

# =========================
# METRICS
# =========================

total_questions = 0

total_hitrate = 0
total_recall = 0
total_mrr = 0
total_ndcg = 0

# =========================
# LOOP
# =========================

for _, row in df.iterrows():

    question = row["question"]

    gt_chunk_ids = row[
        "ground_truth_chunk_ids"
    ]["c_512"]

    if len(gt_chunk_ids) == 0:
        continue

    results = retriever.search(
        query=question,
        k=5
    )

    retrieved_ids = results["ids"][0]

    total_hitrate += hit_rate_at_k(
        retrieved_ids,
        gt_chunk_ids
    )

    total_recall += recall_at_k(
        retrieved_ids,
        gt_chunk_ids
    )

    total_mrr += mrr(
        retrieved_ids,
        gt_chunk_ids
    )

    total_ndcg += ndcg_at_k(
        retrieved_ids,
        gt_chunk_ids
    )

    total_questions += 1

# =========================
# RESULTS
# =========================

print()

print(
    "Questions:",
    total_questions
)

print(
    "HitRate@5:",
    round(
        total_hitrate / total_questions,
        4
    )
)

print(
    "Recall@5:",
    round(
        total_recall / total_questions,
        4
    )
)

print(
    "MRR:",
    round(
        total_mrr / total_questions,
        4
    )
)

print(
    "NDCG@5:",
    round(
        total_ndcg / total_questions,
        4
    )
)