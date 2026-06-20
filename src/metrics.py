import math


def hit_rate_at_k(retrieved_ids, ground_truth_ids):

    for doc_id in retrieved_ids:
        if doc_id in ground_truth_ids:
            return 1.0

    return 0.0


def recall_at_k(retrieved_ids, ground_truth_ids):

    hits = 0

    for doc_id in retrieved_ids:
        if doc_id in ground_truth_ids:
            hits += 1

    return hits / len(ground_truth_ids)


def mrr(retrieved_ids, ground_truth_ids):

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in ground_truth_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, ground_truth_ids):

    dcg = 0

    for rank, doc_id in enumerate(retrieved_ids, start=1):

        if doc_id in ground_truth_ids:
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(
        len(ground_truth_ids),
        len(retrieved_ids)
    )

    idcg = sum(
        1 / math.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0

    return dcg / idcg