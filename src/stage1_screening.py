import pandas as pd

from retrievers import DenseRetriever
from metrics import (
    hit_rate_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k
)

df = pd.read_parquet(
    "data/processed/questions/questions_with_chunk_ids.parquet"
)

collections = [
    "c_512",
    "c_1024",
    "c_rec",
    "c_sem"
]

results_log = []

for collection_name in collections:

    retriever = DenseRetriever(
        chroma_path="data/processed/embeddings",
        collection_name=collection_name
    )

    total_questions = 0

    total_hitrate = 0
    total_recall = 0
    total_mrr = 0
    total_ndcg = 0

    for _, row in df.iterrows():

        gt_chunk_ids = row["ground_truth_chunk_ids"][
            collection_name
        ]

        if len(gt_chunk_ids) == 0:
            continue

        question = row["question"]

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

    results_log.append({
        "chunking": collection_name,
        "retrieval": "dense",
        "hitrate": total_hitrate / total_questions,
        "recall": total_recall / total_questions,
        "mrr": total_mrr / total_questions,
        "ndcg": total_ndcg / total_questions
    })

results_df = pd.DataFrame(results_log)

results_df.to_csv(
    "results/stage1_screening_logs.csv",
    index=False
)

print(results_df)
print("\nSaved -> results/stage1_screening_logs.csv")