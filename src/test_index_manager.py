from index_manager import IndexManager


manager = IndexManager(
    "data/processed/embeddings"
)

manager.load_all()