import chromadb

client = chromadb.PersistentClient(
    path="data/processed/embeddings"
)

for name in ["c_512", "c_1024", "c_rec", "c_sem"]:

    collection = client.get_collection(name)

    print("\n")
    print("=" * 50)
    print(name)

    results = collection.query(
        query_texts=["What is the validity period?"],
        n_results=1
    )

    print(results["documents"][0][0][:500])