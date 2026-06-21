from bm25_retriever import BM25Retriever


retriever = BM25Retriever(
    chroma_path="data/processed/embeddings",
    collection_name="c_512"
)

results = retriever.search(
    "How long is the validity period for the telemetry driven runbook author certification?",
    k=5
)

for r in results:

    print(r)