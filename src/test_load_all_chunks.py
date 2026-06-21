import chromadb

client = chromadb.PersistentClient(
    path="data/processed/embeddings"
)

collection = client.get_collection(
    "c_512"
)

count = collection.count()

print("Total chunks:", count)

documents = []
chunk_ids = []

batch_size = 1000

for offset in range(0, count, batch_size):

    batch = collection.get(
        limit=batch_size,
        offset=offset,
        include=["documents"]
    )

    documents.extend(
        batch["documents"]
    )

    chunk_ids.extend(
        batch["ids"]
    )

print()

print("Loaded docs:", len(documents))
print("Loaded ids:", len(chunk_ids))

print()

print("First chunk id:")
print(chunk_ids[0])

print()

print("First document:")
print(documents[0][:200])