import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer


CHROMA_PATH = "data/processed/embeddings"

COLLECTIONS = [
    "c_512",
    "c_1024",
    "c_rec",
    "c_sem"
]


print("Loading questions...")

df = pd.read_parquet(
    "data/raw/confluence_questions.parquet"
)

print(f"Questions: {len(df)}")


print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Connecting Chroma...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collections = {}

for name in COLLECTIONS:

    collections[name] = client.get_collection(
        name
    )

    print(
        f"{name}: {collections[name].count()}"
    )


ground_truth_chunk_ids = []


for idx, row in df.iterrows():

    print(
        f"Processing {idx+1}/{len(df)}",
        end="\r"
    )

    answer = row["ground_truth_answer"]

    doc_ids = row["source_document_ids"]

    embedding = model.encode(
        answer
    ).tolist()

    result_per_collection = {}

    for col_name in COLLECTIONS:

        col = collections[col_name]

        ids_found = []

        for doc_id in doc_ids:

            try:

                results = col.query(
                    query_embeddings=[embedding],
                    n_results=1,
                    where={
                        "doc_id": doc_id
                    }
                )

                ids = results["ids"][0]

                if ids:
                    ids_found.append(
                        ids[0]
                    )

            except Exception as e:

                print(
                    f"\nError {col_name}: {e}"
                )

        result_per_collection[col_name] = ids_found

    ground_truth_chunk_ids.append(
        result_per_collection
    )


df["ground_truth_chunk_ids"] = (
    ground_truth_chunk_ids
)

output_path = (
    "data/processed/questions/"
    "questions_with_chunk_ids.parquet"
)

df.to_parquet(
    output_path,
    index=False
)

print()
print("DONE")
print(output_path)