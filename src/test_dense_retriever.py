import chromadb


class DenseRetriever:

    def __init__(self, chroma_path):

        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

    def search(
        self,
        collection_name,
        query,
        k=5
    ):

        collection = self.client.get_collection(
            collection_name
        )

        results = collection.query(
            query_texts=[query],
            n_results=k
        )

        return results


print("START")

retriever = DenseRetriever(
    "data/processed/embeddings"
)

print("CONNECTED")

results = retriever.search(
    collection_name="c_512",
    query="What is the validity period?",
    k=3
)

print("QUERY DONE")

print(results["documents"][0])