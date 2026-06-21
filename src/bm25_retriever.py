import chromadb
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(
        self,
        chroma_path,
        collection_name
    ):

        client = chromadb.PersistentClient(
            path=chroma_path
        )

        collection = client.get_collection(
            collection_name
        )

        count = collection.count()

        self.documents = []
        self.chunk_ids = []

        batch_size = 1000

        for offset in range(
            0,
            count,
            batch_size
        ):

            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents"]
            )

            self.documents.extend(
                batch["documents"]
            )

            self.chunk_ids.extend(
                batch["ids"]
            )

        tokenized_docs = [
            doc.lower().split()
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

    def search(
        self,
        query,
        k=5
    ):

        query_tokens = (
            query.lower()
            .split()
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        results = []

        for idx, score in ranked:

            results.append({
                "chunk_id":
                    self.chunk_ids[idx],
                "score":
                    float(score)
            })

        return results