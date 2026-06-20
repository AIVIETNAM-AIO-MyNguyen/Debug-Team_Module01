import pandas as pd

from retrievers import DenseRetriever


df = pd.read_parquet(
    "data/raw/confluence_questions.parquet"
)

question = df.iloc[0]["question"]

print("QUESTION:")
print(question)

retriever = DenseRetriever(
    chroma_path="data/processed/embeddings",
    collection_name="c_512"
)

results = retriever.search(
    question,
    k=5
)

print("\nTOP 5 RESULTS:\n")

for i, doc in enumerate(results["documents"][0]):

    print(f"Result {i+1}")
    print(doc[:300])
    print("-" * 50)