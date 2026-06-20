import chromadb

client = chromadb.PersistentClient(
    path="data/processed/embeddings"
)

collection = client.get_collection("c_512")

result = collection.get(
    limit=1,
    include=["metadatas"]
)

print(result["metadatas"])