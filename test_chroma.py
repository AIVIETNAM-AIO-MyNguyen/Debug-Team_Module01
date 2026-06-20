import chromadb

client = chromadb.PersistentClient(
    path="./data/processed/embeddings"
)

print(client.list_collections())