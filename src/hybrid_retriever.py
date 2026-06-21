from retrievers import DenseRetriever
from bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(
        self,
        chroma_path,
        collection_name
    ):

        self.dense = DenseRetriever(
            chroma_path,
            collection_name
        )

        self.bm25 = BM25Retriever(
            chroma_path,
            collection_name
        )

    def search(
        self,
        query,
        k=5,
        rrf_k=60
    ):

        dense_results = self.dense.search(
            query,
            k=20
        )

        bm25_results = self.bm25.search(
            query,
            k=20
        )

        scores = {}

        dense_ids = dense_results["ids"][0]

        for rank, chunk_id in enumerate(
            dense_ids,
            start=1
        ):

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

        bm25_ids = [
            r["chunk_id"]
            for r in bm25_results
        ]

        for rank, chunk_id in enumerate(
            bm25_ids,
            start=1
        ):

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            chunk_id
            for chunk_id, _
            in ranked[:k]
        ]