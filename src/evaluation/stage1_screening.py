import math
import logging
import pandas as pd
from typing import List, Dict, Any, Set
from core.index_manager import IndexManager
from core.query_transforms import QueryTransformer
from core.retrievers import ModularRetriever
from core.post_processors import PostProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Stage1ScreeningEngine:
    """Runs the high-speed screening matrix across valid pipeline permutations."""
    def __init__(self, dataset: List[Dict[str, Any]], index_manager: IndexManager, query_transformer: QueryTransformer, retriever: ModularRetriever, post_processor: PostProcessor):
        self.dataset = dataset
        self.idx_mgr = index_manager
        self.query_tf = query_transformer
        self.retriever = retriever
        self.post_proc = post_processor
        self.metrics_registry: List[Dict[str, Any]] = []

    def calculate_metrics(self, retrieved_ids: List[str], target_ids: List[str]) -> Dict[str, float]:
        """Computes Hit Rate@5, Recall@5, MRR, and NDCG@5 based on dynamic mappings."""
        # Clean inputs and keep top 5
        retrieved = retrieved_ids[:5]
        targets = set(target_ids)
        
        if not targets:
            return {
                "hit_rate_at_5": 0.0,
                "recall_at_5": 0.0,
                "mrr_at_5": 0.0,
                "ndcg_at_5": 0.0
            }
            
        # 1. Hit Rate@5
        hit = 1.0 if any(rid in targets for rid in retrieved) else 0.0
        
        # 2. Recall@5
        intersection = set(retrieved) & targets
        recall = len(intersection) / len(targets)
        
        # 3. MRR
        mrr = 0.0
        for rank, rid in enumerate(retrieved, start=1):
            if rid in targets:
                mrr = 1.0 / rank
                break
                
        # 4. NDCG@5
        dcg = 0.0
        for rank, rid in enumerate(retrieved, start=1):
            if rid in targets:
                dcg += 1.0 / math.log2(rank + 1)
                
        idcg = 0.0
        for rank in range(1, min(5, len(targets)) + 1):
            idcg += 1.0 / math.log2(rank + 1)
            
        ndcg = dcg / idcg if idcg > 0.0 else 0.0
        
        # Ensure strict bounding between 0.0 and 1.0
        return {
            "hit_rate_at_5": max(0.0, min(1.0, float(hit))),
            "recall_at_5": max(0.0, min(1.0, float(recall))),
            "mrr_at_5": max(0.0, min(1.0, float(mrr))),
            "ndcg_at_5": max(0.0, min(1.0, float(ndcg)))
        }

    def run_screening_sweep(self, pipelines: List[Dict[str, str]]) -> pd.DataFrame:
        """Loops through valid combinations, logs metrics per query, and outputs a flat table using multithreading."""
        self.metrics_registry = []
        total_runs = len(pipelines) * len(self.dataset)
        
        logger.info(f"Starting Stage 1 sweep: {len(pipelines)} pipelines across {len(self.dataset)} questions ({total_runs} runs).")

        max_workers = 1
        logger.info(f"Running sweep with {max_workers} thread workers (sequential local execution).")

        def process_pipeline(pipeline: Dict[str, str]) -> List[Dict[str, Any]]:
            chunking = pipeline["chunking"]
            pre_retrieval = pipeline["pre_retrieval"]
            retrieval = pipeline["retrieval"]
            index_structure = pipeline["index_structure"]
            post_retrieval = pipeline["post_retrieval"]
            
            pipeline_name = f"P_{chunking}_{pre_retrieval}_{retrieval}_{index_structure}_{post_retrieval}"
            pipeline_records = []
            
            for q_item in self.dataset:
                q_id = q_item["id"]
                raw_query = q_item["question"]
                
                col_map = {
                    "fixed_512": "c_512",
                    "fixed_1024": "c_1024",
                    "recursive": "c_rec",
                    "semantic": "c_sem"
                }
                col_name = col_map.get(chunking, chunking)
                target_ids = q_item.get("ground_truth_chunk_ids", {}).get(col_name, [])
                
                # Step 1: Pre-Retrieval query transform
                queries = self.query_tf.execute_transform(q_id, raw_query, pre_retrieval)
                
                # Step 2: Core Retrieval
                # Retrieve top 10 first to allow post-processing to filter down to 5
                retrieve_k = 10
                
                if retrieval == "dense_cosine":
                    retrieved_results = self.retriever.search_dense(queries, chunking, index_structure, retrieve_k)
                elif retrieval == "sparse_bm25":
                    retrieved_results = self.retriever.search_sparse(queries, chunking, index_structure, retrieve_k)
                elif retrieval == "hybrid_rrf":
                    dense_res = self.retriever.search_dense(queries, chunking, index_structure, retrieve_k)
                    sparse_res = self.retriever.search_sparse(queries, chunking, index_structure, retrieve_k)
                    retrieved_results = self.retriever.fuse_hybrid_rrf(dense_res, sparse_res)
                else:
                    retrieved_results = []
                    
                # Step 3: Post-Retrieval processing
                if post_retrieval == "cross_encoder_rerank":
                    processed_results = self.post_proc.rerank_cross_encoder(raw_query, retrieved_results, top_n=5)
                elif post_retrieval == "contextual_compression":
                    # Compress noise from candidates, keeping threshold 0.15 for cosine similarity
                    compressed = self.post_proc.compress_contextual_noise(raw_query, retrieved_results, threshold=0.15)
                    processed_results = compressed[:5]
                else:
                    processed_results = retrieved_results[:5]
                    
                # Extract chunk IDs directly
                retrieved_ids = [res["metadata"]["chunk_id"] for res in processed_results]
                
                # Calculate metrics
                metrics = self.calculate_metrics(retrieved_ids, target_ids)
                
                # Log metrics
                pipeline_records.append({
                    "pipeline_id": pipeline_name,
                    "chunking": chunking,
                    "pre_retrieval": pre_retrieval,
                    "retrieval": retrieval,
                    "index_structure": index_structure,
                    "post_retrieval": post_retrieval,
                    "question_id": q_id,
                    "hit_rate_at_5": metrics["hit_rate_at_5"],
                    "recall_at_5": metrics["recall_at_5"],
                    "mrr_at_5": metrics["mrr_at_5"],
                    "ndcg_at_5": metrics["ndcg_at_5"]
                })
            return pipeline_records

        results = []
        p_idx = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pipeline = {executor.submit(process_pipeline, p): p for p in pipelines}
            for future in as_completed(future_to_pipeline):
                results.extend(future.result())
                p_idx += 1
                if p_idx % 10 == 0 or p_idx == len(pipelines):
                    logger.info(f"Processed {p_idx}/{len(pipelines)} pipelines ({p_idx/len(pipelines)*100:.1f}%).")
                    
        self.metrics_registry = results
        df = pd.DataFrame(self.metrics_registry)
        logger.info(f"Stage 1 screening sweep completed. Created logs DataFrame of size {df.shape}.")
        return df
