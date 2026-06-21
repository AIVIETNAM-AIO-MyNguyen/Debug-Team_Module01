import chromadb

client = chromadb.PersistentClient(
    path="data/processed/embeddings"
)

collection = client.get_collection(
    "c_512"
)

count = collection.count()

print("Total:", count)

data = collection.get(
    limit=5,
    include=["documents", "metadatas"]
)

print()
print(data["documents"][0][:300])

print()
print(data["metadatas"][0])