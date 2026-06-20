from typing import List, Dict, Any, Union
from .index_manager import IndexManager

class ModularRetriever:
    """Runs core database search algorithms across active indices."""
    def __init__(self, index_manager: IndexManager):
        self.idx_mgr = index_manager

    def _get_parent_chunk(self, strategy_name: str, child_id: str) -> Dict[str, Any]:
        """Resolves child chunk ID to parent chunk metadata."""
        parent_map = self.idx_mgr.parent_child_mappings.get(strategy_name, {})
        parent_id = parent_map.get(child_id)
        if not parent_id:
            # Fallback if no parent mapping
            return None
            
        chunks_db = self.idx_mgr.parent_chunks_db.get(strategy_name, {})
        parent_chunk = chunks_db.get(parent_id)
        if not parent_chunk:
            return None
            
        return {
            "chunk_id": parent_chunk.chunk_id,
            "text": parent_chunk.text,
            "doc_id": parent_chunk.doc_id
        }

    def _resolve_results(self, raw_results: List[Dict[str, Any]], strategy_name: str, index_structure: str, top_k: int) -> List[Dict[str, Any]]:
        """Resolves raw search results, resolving child chunks to parent chunks if required."""
        if not raw_results:
            return []
            
        if index_structure == "parent_document":
            seen_parents = set()
            resolved = []
            for res in raw_results:
                meta = res["metadata"]
                child_id = meta["chunk_id"]
                parent_meta = self._get_parent_chunk(strategy_name, child_id)
                if parent_meta:
                    p_id = parent_meta["chunk_id"]
                    if p_id not in seen_parents:
                        seen_parents.add(p_id)
                        # Retain the match score
                        resolved.append({
                            "score": res["score"],
                            "metadata": parent_meta
                        })
                else:
                    # Fallback to child if parent not mapped
                    p_id = meta["chunk_id"]
                    if p_id not in seen_parents:
                        seen_parents.add(p_id)
                        resolved.append(res)
            return resolved[:top_k]
        else:
            return raw_results[:top_k]

    def search_dense(self, queries: List[str], strategy_name: str, index_structure: str, top_k: int) -> List[Dict[str, Any]]:
        """Executes vector embedding searches using cosine similarity matching."""
        # 1. Query persistent ChromaDB directly if initialized and flat structure is requested
        if self.idx_mgr.chroma_client is not None and index_structure == "flat":
            col_map = {
                "fixed_512": "c_512",
                "fixed_1024": "c_1024",
                "recursive": "c_rec",
                "semantic": "c_sem"
            }
            col_name = col_map.get(strategy_name)
            col = self.idx_mgr.chroma_collections.get(col_name)
            if col is not None:
                all_raw_results = []
                for q in queries:
                    # Encode query using SentenceTransformer model
                    q_emb = self.idx_mgr.model.encode([q])[0].tolist()
                    q_res = col.query(query_embeddings=[q_emb], n_results=top_k)
                    
                    if q_res and q_res["ids"] and len(q_res["ids"]) > 0:
                        ids = q_res["ids"][0]
                        docs = q_res["documents"][0]
                        metadatas = q_res["metadatas"][0]
                        distances = q_res["distances"][0] if "distances" in q_res and q_res["distances"] else [0.0]*len(ids)
                        
                        for cid, doc, meta, dist in zip(ids, docs, metadatas, distances):
                            # Distance metric in Chroma is typically L2 or Cosine distance.
                            # We assign a score equal to 1.0 - distance to align with similarity logic.
                            all_raw_results.append({
                                "score": float(1.0 - dist),
                                "metadata": {
                                    "chunk_id": cid,
                                    "text": doc,
                                    "doc_id": meta.get("doc_id", ""),
                                    "start_char": meta.get("start_char", 0),
                                    "end_char": meta.get("end_char", len(doc)),
                                    "strategy": strategy_name
                                }
                            })
                # Deduplicate/aggregate by chunk_id
                aggregated: Dict[str, Dict[str, Any]] = {}
                for res in all_raw_results:
                    cid = res["metadata"]["chunk_id"]
                    if cid not in aggregated or res["score"] > aggregated[cid]["score"]:
                        aggregated[cid] = res
                        
                sorted_results = sorted(aggregated.values(), key=lambda x: x["score"], reverse=True)
                return self._resolve_results(sorted_results, strategy_name, index_structure, top_k)

        # 2. Fallback to Local Vector Store
        index_key = f"{strategy_name}_{index_structure}"
        vector_store = self.idx_mgr.active_vector_stores.get(index_key)
        if not vector_store:
            return []
            
        all_raw_results = []
        for q in queries:
            if self.idx_mgr.vectorizer:
                try:
                    q_emb = self.idx_mgr.vectorizer.transform([q]).toarray()[0].tolist()
                except Exception:
                    q_emb = [0.0] * 384
            else:
                q_emb = [0.0] * 384
                
            q_res = vector_store.search(q_emb, top_k * 3)  # Search deeper to account for parent document grouping
            all_raw_results.extend(q_res)
            
        aggregated: Dict[str, Dict[str, Any]] = {}
        for res in all_raw_results:
            cid = res["metadata"]["chunk_id"]
            if cid not in aggregated or res["score"] > aggregated[cid]["score"]:
                aggregated[cid] = res
                
        sorted_results = sorted(aggregated.values(), key=lambda x: x["score"], reverse=True)
        return self._resolve_results(sorted_results, strategy_name, index_structure, top_k)

    def search_sparse(self, queries: List[str], strategy_name: str, index_structure: str, top_k: int) -> List[Dict[str, Any]]:
        """Executes standard lexical keyword lookups using BM25 frequency checks."""
        index_key = f"{strategy_name}_{index_structure}"
        bm25_idx = self.idx_mgr.active_bm25_indices.get(index_key)
        if not bm25_idx:
            return []
            
        all_raw_results = []
        for q in queries:
            q_res = bm25_idx.search(q, top_k * 3)
            all_raw_results.extend(q_res)
            
        aggregated: Dict[str, Dict[str, Any]] = {}
        for res in all_raw_results:
            cid = res["metadata"]["chunk_id"]
            if cid not in aggregated or res["score"] > aggregated[cid]["score"]:
                aggregated[cid] = res
                
        sorted_results = sorted(aggregated.values(), key=lambda x: x["score"], reverse=True)
        return self._resolve_results(sorted_results, strategy_name, index_structure, top_k)

    def fuse_hybrid_rrf(self, dense_results: List[Any], sparse_results: List[Any], k_constant: int = 60) -> List[Any]:
        """Interleaves dense and sparse rankings using the Reciprocal Rank Fusion algorithm."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        # Helper to compute and accumulate rank scores
        def accumulate_rrf(results: List[Any]):
            for rank, item in enumerate(results, start=1):
                meta = item["metadata"]
                cid = meta["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_constant + rank))
                doc_map[cid] = item
                
        accumulate_rrf(dense_results)
        accumulate_rrf(sparse_results)
        
        # Sort documents by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        fused = []
        for cid in sorted_ids:
            # We copy the original document record but assign the RRF score
            item = doc_map[cid]
            fused.append({
                "score": rrf_scores[cid],
                "metadata": item["metadata"]
            })
            
        return fused
