import chromadb


class IndexManager:

    def __init__(self, chroma_path):

        self.client = chromadb.PersistentClient(
            path=chroma_path
        )

        self.collections = {}

    def load_collection(self, collection_name):

        collection = self.client.get_collection(
            collection_name
        )

        self.collections[collection_name] = collection

        return collection

    def load_all(self):

        names = [
            "c_512",
            "c_1024",
            "c_rec",
            "c_sem"
        ]

        for name in names:

            try:

                collection = self.load_collection(name)

                print(
                    f"{name}: {collection.count()} chunks"
                )

            except Exception as e:

                print(
                    f"Failed {name}: {e}"
                )

    def get_collection(self, name):

        return self.collections[name]