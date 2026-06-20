import pandas as pd
import chromadb

# Load questions
df = pd.read_parquet(
    "data/raw/confluence_questions.parquet"
)

row = df.iloc[0]

question = row["question"]

print("=" * 50)
print("QUESTION:")
print(question)

print("=" * 50)
print("EXPECTED DOC:")
print(row["source_document_ids"])

# Connect Chroma
client = chromadb.PersistentClient(
    path="data/processed/embeddings"
)

collection = client.get_collection("c_512")

results = collection.query(
    query_texts=[question],
    n_results=5
)

print("=" * 50)
print("RETRIEVED IDS")

for i, metadata in enumerate(results["metadatas"][0]):
    print(i + 1, metadata["doc_id"])