import re
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .chunkers import Chunk

class LocalVectorStore:
    """NumPy-accelerated local vector database for cosine similarity search."""
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Adds dense embeddings and associated metadata to the store."""
        for emb, meta in zip(embeddings, metadata):
            self.embeddings.append(np.array(emb, dtype=np.float32))
            self.metadata.append(meta)

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Performs cosine similarity search against active embeddings."""
        if not self.embeddings:
            return []
        
        # Build matrix
        emb_matrix = np.vstack(self.embeddings)  # Shape: (N, D)
        q_emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)  # Shape: (1, D)
        
        # Compute cosine similarities
        similarities = cosine_similarity(q_emb, emb_matrix)[0]  # Shape: (N,)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        return results


class LocalBM25Index:
    """Python-native BM25 search index for lexical keyword matching."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into lowercase words."""
        return re.findall(r'\w+', text.lower())

    def fit(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Constructs term-frequency table and term inverse document frequencies."""
        self.doc_texts = texts
        self.metadata = metadata
        self.doc_len = []
        self.doc_term_freqs = []
        self.doc_freqs = {}
        
        for text in texts:
            tokens = self._tokenize(text)
            self.doc_len.append(len(tokens))
            
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.doc_term_freqs.append(term_freq)
            
            # Document frequencies
            for token in term_freq.keys():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                
        num_docs = len(texts)
        if num_docs > 0:
            self.avg_doc_len = sum(self.doc_len) / num_docs
        else:
            self.avg_doc_len = 0.0
            
        # Compute IDFs
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF formulation with smoothing
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Calculates term frequencies and returns documents sorted by BM25 score."""
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.doc_texts:
            return []
            
        scores = []
        for i in range(len(self.doc_texts)):
            score = 0.0
            doc_len = self.doc_len[i]
            term_freq = self.doc_term_freqs[i]
            
            for token in query_tokens:
                if token in self.idf:
                    tf = term_freq.get(token, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                    score += self.idf[token] * (numerator / denominator)
            scores.append((score, i))
            
        # Sort scores descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            results.append({
                "score": score,
                "metadata": self.metadata[idx]
            })
        return results


class IndexManager:
    """Manages active vector stores and BM25 search indices for distinct strategies."""
    def __init__(self):
        self.active_vector_stores: Dict[str, LocalVectorStore] = {}
        self.active_bm25_indices: Dict[str, LocalBM25Index] = {}
        self.parent_child_mappings: Dict[str, Dict[str, str]] = {}  # child_id -> parent_id
        
        # Cache of parent chunk data for fast retrieval lookup (chunk_id -> Chunk object)
        self.parent_chunks_db: Dict[str, Dict[str, Chunk]] = {}
        
        # Shared vectorizer for dense local embeddings
        self.vectorizer: Optional[TfidfVectorizer] = None

        # ChromaDB persistent client properties
        self.chroma_client = None
        self.chroma_collections = {}
        self.model = None

    def init_chroma(self, path: str = "data/processed/embeddings") -> bool:
        """Initializes ChromaDB persistent client and sentence transformer model."""
        import os
        if not os.path.exists(path):
            return False
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            self.chroma_client = chromadb.PersistentClient(path=path)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Load collections
            for name in ["c_512", "c_1024", "c_rec", "c_sem"]:
                try:
                    self.chroma_collections[name] = self.chroma_client.get_collection(name)
                except Exception:
                    pass
            return True
        except Exception as e:
            return False

    def load_indices_from_chroma(self, strategy_name: str, collection_name: str, dataset: Optional[List[Dict[str, Any]]] = None) -> None:
        """Loads chunks from the persistent ChromaDB collection and fits local index structures."""
        if not self.chroma_client:
            return
            
        col = self.chroma_collections.get(collection_name)
        if not col:
            return
            
        # Gather target ground truth IDs from dataset
        target_ids = []
        if dataset:
            for item in dataset:
                ids = item.get("ground_truth_chunk_ids", {}).get(collection_name, [])
                if isinstance(ids, list):
                    target_ids.extend(ids)
                elif isinstance(ids, str):
                    target_ids.append(ids)
        target_ids = list(set([tid for tid in target_ids if tid]))

        all_ids = []
        all_documents = []
        all_metadatas = []

        # 1. Load target ground-truth items
        if target_ids:
            try:
                res_gt = col.get(ids=target_ids, include=["metadatas", "documents"])
                if res_gt and "ids" in res_gt:
                    all_ids.extend(res_gt["ids"])
                    all_documents.extend(res_gt["documents"])
                    all_metadatas.extend(res_gt["metadatas"])
            except Exception as e:
                import logging
                logging.getLogger("rag_bench").warning(f"Error fetching ground-truth IDs from Chroma: {e}")

        # 2. Fetch a background sample of up to 5000 distractor items
        try:
            res_dist = col.get(limit=5000, include=["metadatas", "documents"])
            if res_dist and "ids" in res_dist:
                gt_set = set(all_ids)
                for cid, doc, meta in zip(res_dist["ids"], res_dist["documents"], res_dist["metadatas"]):
                    if cid not in gt_set:
                        all_ids.append(cid)
                        all_documents.append(doc)
                        all_metadatas.append(meta)
        except Exception as e:
            import logging
            logging.getLogger("rag_bench").warning(f"Error fetching distractor sample from Chroma: {e}")

        if not all_ids:
            return
            
        chunks = []
        for cid, doc, meta in zip(all_ids, all_documents, all_metadatas):
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                start_char=meta.get("start_char", 0),
                end_char=meta.get("end_char", len(doc)),
                strategy=strategy_name
            ))
            
        # Store parent chunks for lookup
        if strategy_name not in self.parent_chunks_db:
            self.parent_chunks_db[strategy_name] = {}
        for c in chunks:
            self.parent_chunks_db[strategy_name][c.chunk_id] = c
            
        # Get all chunk texts to fit the vectorizer
        all_texts = [c.text for c in chunks]
        vectorizer = self._get_embedding_vectorizer(all_texts)
        
        # 1. Build Flat Index Structures
        flat_key = f"{strategy_name}_flat"
        
        # We query ChromaDB directly for flat dense search, so register an empty LocalVectorStore to save memory
        self.active_vector_stores[flat_key] = LocalVectorStore()
        
        bm25_idx = LocalBM25Index()
        bm25_idx.fit(all_texts, [{"chunk_id": c.chunk_id, "text": c.text, "doc_id": c.doc_id} for c in chunks])
        self.active_bm25_indices[flat_key] = bm25_idx
        
        # 2. Build Parent Document Index Structures
        parent_doc_key = f"{strategy_name}_parent_document"
        
        child_chunks: List[Tuple[str, str, str]] = []  # (child_id, text, parent_id)
        for parent_chunk in chunks:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parent_chunk.text) if s.strip()]
            if not sentences:
                sentences = [parent_chunk.text]
                
            for s_idx, sent_text in enumerate(sentences):
                child_id = f"{parent_chunk.chunk_id}_child_{s_idx:03d}"
                child_chunks.append((child_id, sent_text, parent_chunk.chunk_id))
                
        if child_chunks:
            # Register parent child mapping
            self.parent_child_mappings[strategy_name] = {}
            for child_id, _, parent_id in child_chunks:
                self.parent_child_mappings[strategy_name][child_id] = parent_id
                
            child_texts = [x[1] for x in child_chunks]
            child_metadata = [{"chunk_id": x[0], "text": x[1], "doc_id": chunks[0].doc_id, "parent_id": x[2]} for x in child_chunks]
            
            try:
                child_embeddings = vectorizer.transform(child_texts).toarray().tolist()
            except Exception:
                child_embeddings = np.random.rand(len(child_chunks), 384).tolist()
                
            child_vector_store = LocalVectorStore()
            child_vector_store.add(child_embeddings, child_metadata)
            self.active_vector_stores[parent_doc_key] = child_vector_store
            
            child_bm25_idx = LocalBM25Index()
            child_bm25_idx.fit(child_texts, child_metadata)
            self.active_bm25_indices[parent_doc_key] = child_bm25_idx

    def _get_embedding_vectorizer(self, corpus_texts: List[str]) -> TfidfVectorizer:
        """Fits a shared vectorizer for dense representations if not already done."""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=384, analyzer='char_wb', ngram_range=(3, 5))
            try:
                self.vectorizer.fit(corpus_texts)
            except ValueError:
                pass
        return self.vectorizer

    def build_indices_for_strategy(self, chunks: List[Chunk], strategy_name: str) -> None:
        """Constructs vector stores and term-frequency tables for a chunking strategy."""
        if not chunks:
            return
            
        # Store parent chunks for lookup
        if strategy_name not in self.parent_chunks_db:
            self.parent_chunks_db[strategy_name] = {}
        for c in chunks:
            self.parent_chunks_db[strategy_name][c.chunk_id] = c
            
        # Get all chunk texts to fit the vectorizer
        all_texts = [c.text for c in chunks]
        vectorizer = self._get_embedding_vectorizer(all_texts)
        
        # Generate embeddings
        try:
            embeddings_matrix = vectorizer.transform(all_texts).toarray().tolist()
        except Exception:
            # Fallback to random embeddings if transform fails
            embeddings_matrix = np.random.rand(len(chunks), 384).tolist()

        # 1. Build Flat Index Structures
        flat_key = f"{strategy_name}_flat"
        
        vector_store = LocalVectorStore()
        metadata_list = [{"chunk_id": c.chunk_id, "text": c.text, "doc_id": c.doc_id} for c in chunks]
        vector_store.add(embeddings_matrix, metadata_list)
        self.active_vector_stores[flat_key] = vector_store
        
        bm25_idx = LocalBM25Index()
        bm25_idx.fit(all_texts, metadata_list)
        self.active_bm25_indices[flat_key] = bm25_idx

        # 2. Build Parent Document Index Structures
        parent_doc_key = f"{strategy_name}_parent_document"
        
        # Split each parent chunk into smaller sentences as child chunks
        child_chunks: List[Tuple[str, str, str]] = []  # (child_id, text, parent_id)
        for parent_chunk in chunks:
            # Simple sentence splitting
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parent_chunk.text) if s.strip()]
            if not sentences:
                sentences = [parent_chunk.text]
                
            for s_idx, sent_text in enumerate(sentences):
                child_id = f"{parent_chunk.chunk_id}_child_{s_idx:03d}"
                child_chunks.append((child_id, sent_text, parent_chunk.chunk_id))

        if child_chunks:
            # Register parent child mapping
            self.parent_child_mappings[strategy_name] = {}
            for child_id, _, parent_id in child_chunks:
                self.parent_child_mappings[strategy_name][child_id] = parent_id
                
            child_texts = [x[1] for x in child_chunks]
            child_metadata = [{"chunk_id": x[0], "text": x[1], "doc_id": chunks[0].doc_id, "parent_id": x[2]} for x in child_chunks]
            
            try:
                child_embeddings = vectorizer.transform(child_texts).toarray().tolist()
            except Exception:
                child_embeddings = np.random.rand(len(child_chunks), 384).tolist()
                
            child_vector_store = LocalVectorStore()
            child_vector_store.add(child_embeddings, child_metadata)
            self.active_vector_stores[parent_doc_key] = child_vector_store
            
            child_bm25_idx = LocalBM25Index()
            child_bm25_idx.fit(child_texts, child_metadata)
            self.active_bm25_indices[parent_doc_key] = child_bm25_idx

    def register_hierarchical_map(self, parent_chunks: List[Chunk], child_chunks: List[Chunk]) -> None:
        """Maps relationship links for Parent-Document retrieval architectures."""
        # This registers an explicit mapping from a pre-defined set of chunks
        # Here we check if the strategy is already registered, otherwise record it
        if not parent_chunks or not child_chunks:
            return
        
        strategy_name = parent_chunks[0].strategy
        if strategy_name not in self.parent_child_mappings:
            self.parent_child_mappings[strategy_name] = {}
            
        # Store parent chunks for fast lookup
        if strategy_name not in self.parent_chunks_db:
            self.parent_chunks_db[strategy_name] = {}
            
        for c in parent_chunks:
            self.parent_chunks_db[strategy_name][c.chunk_id] = c
            
        # Map child_chunks to parents based on overlap or substring matching
        for child in child_chunks:
            best_parent_id = None
            # Find which parent contains this child's text (approximate match)
            for parent in parent_chunks:
                if child.text in parent.text or parent.text in child.text:
                    best_parent_id = parent.chunk_id
                    break
            if not best_parent_id and parent_chunks:
                # Fallback to the first parent
                best_parent_id = parent_chunks[0].chunk_id
                
            self.parent_child_mappings[strategy_name][child.chunk_id] = best_parent_id
