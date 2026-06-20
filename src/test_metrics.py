def hit_rate_at_k(
    retrieved_ids,
    ground_truth_ids
):

    return float(
        any(
            x in retrieved_ids
            for x in ground_truth_ids
        )
    )


def recall_at_k(
    retrieved_ids,
    ground_truth_ids
):

    hits = sum(
        1
        for x in ground_truth_ids
        if x in retrieved_ids
    )

    return hits / len(ground_truth_ids)


def mrr(
    retrieved_ids,
    ground_truth_ids
):

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if chunk_id in ground_truth_ids:

            return 1 / rank

    return 0.0


retrieved = [
    "A",
    "B",
    "C",
    "D",
    "E"
]

ground_truth = [
    "C"
]

print(
    hit_rate_at_k(
        retrieved,
        ground_truth
    )
)

print(
    recall_at_k(
        retrieved,
        ground_truth
    )
)

print(
    mrr(
        retrieved,
        ground_truth
    )
)