from rank_bm25 import BM25Okapi

docs = [
    "The certification is valid for 18 months",
    "Redwood offices require approval",
    "Telemetry dashboards are updated daily"
]

tokenized_docs = [
    doc.lower().split()
    for doc in docs
]

bm25 = BM25Okapi(tokenized_docs)

query = "validity period certification"

scores = bm25.get_scores(
    query.lower().split()
)

print(scores)