import chromadb


class DenseRetriever:

    def __init__(self, chroma_path, collection_name):
        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collection = self.client.get_collection(
            collection_name
        )

    def search(self, query, k=5):

        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        return results