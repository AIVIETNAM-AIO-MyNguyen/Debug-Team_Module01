import chromadb

client = chromadb.PersistentClient(
    path="data/processed/embeddings"
)

collection = client.get_collection(
    "c_512"
)

data = collection.get(
    limit=5,
    include=["documents", "metadatas"]
)

print("Documents:", len(data["documents"]))

print("\nSample document:\n")
print(data["documents"][0][:500])

print("\nMetadata:\n")
print(data["metadatas"][0])