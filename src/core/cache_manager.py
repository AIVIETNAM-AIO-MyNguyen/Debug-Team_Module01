import os
import re
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalCacheManager:
    """Handles disk-backed serialization of LLM pre-retrieval transformations."""
    def __init__(self, cache_path: str = "data/pre_retrieval_cache.json"):
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, List[str]]] = {}
        self.load_cache()

    def load_cache(self) -> None:
        """Loads cached transformations from the disk file if it exists."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded cache from {self.cache_path} with {len(self.cache)} entries.")
            except Exception as e:
                logger.error(f"Error loading cache from {self.cache_path}: {e}")
                self.cache = {}
        else:
            # Ensure the parent directory exists
            parent_dir = os.path.dirname(self.cache_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            self.cache = {}

    def save_cache(self) -> None:
        """Saves current memory cache state to the JSON file."""
        try:
            parent_dir = os.path.dirname(self.cache_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved cache to {self.cache_path}.")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")

    def precompute_all_transforms(self, questions: List[Dict[str, Any]], generator_llm: Any) -> None:
        """Loops through questions once, generates HyDE/rewrites, and saves to a JSON file."""
        dirty = False
        for q_item in questions:
            q_id = q_item["id"]
            question_str = q_item["question"]
            
            if q_id not in self.cache:
                self.cache[q_id] = {}
                
            # Compute rewrite if not present
            if "query_rewrite" not in self.cache[q_id]:
                logger.info(f"Generating query_rewrite for {q_id}...")
                rewrites = self._call_llm_for_rewrite(question_str, generator_llm)
                self.cache[q_id]["query_rewrite"] = rewrites
                dirty = True
                
            # Compute hyde if not present
            if "hyde" not in self.cache[q_id]:
                logger.info(f"Generating hyde for {q_id}...")
                hyde_docs = self._call_llm_for_hyde(question_str, generator_llm)
                self.cache[q_id]["hyde"] = hyde_docs
                dirty = True
                
        if dirty:
            self.save_cache()

    def get_cached_transform(self, query_id: str, transform_type: str) -> List[str]:
        """Fetches the cached query variations instantly without triggering runtime LLM calls."""
        # Standardize transform_type names
        t_type = transform_type.lower()
        if t_type == "none":
            # Will be handled by query_transforms.py or returning raw query, but standard is to return empty list or fallback
            return []
            
        if query_id in self.cache and t_type in self.cache[query_id]:
            return self.cache[query_id][t_type]
            
        # Fallback to local rule-based mock transform if not cached or generation not run
        logger.warning(f"Cache miss for {query_id} ({t_type}). Returning local mock transform.")
        if t_type == "query_rewrite":
            return [f"Detailed information about: {query_id}"]
        elif t_type == "hyde":
            return [f"Hypothetical document answering: {query_id}"]
        return []

    def _call_llm_for_rewrite(self, question: str, generator_llm: Any) -> List[str]:
        """Generates alternative search queries using the configured LLM or mock fallback."""
        if generator_llm is not None:
            try:
                # Expect generator_llm to have a generate method or be a callable
                prompt = (
                    f"Generate 2 alternative search queries for the following technical question. "
                    f"Separate them with newlines. Do not add numbers or explanations.\n\nQuestion: {question}"
                )
                response = generator_llm(prompt)
                queries = [q.strip() for q in response.split("\n") if q.strip()]
                # Strip leading dashes/numbers just in case
                cleaned_queries = []
                for q in queries:
                    q = re.sub(r'^[-\d.\s]+', '', q)
                    if q:
                        cleaned_queries.append(q)
                return cleaned_queries[:2]
            except Exception as e:
                logger.error(f"Error calling LLM for query rewrite: {e}")
                
        # Mock/Rule-based query rewrite fallback
        words = question.split()
        keywords = [w for w in words if len(w) > 3]
        keyword_str = " ".join(keywords[:5])
        return [
            f"technical details on {keyword_str}",
            f"how does {keyword_str} operate"
        ]

    def _call_llm_for_hyde(self, question: str, generator_llm: Any) -> List[str]:
        """Generates a hypothetical document answering the question using the configured LLM or mock."""
        if generator_llm is not None:
            try:
                prompt = (
                    f"Write a short, hypothetical passage (1-2 paragraphs) that answers the following technical question "
                    f"directly. Assume it is a page from a technical document.\n\nQuestion: {question}"
                )
                response = generator_llm(prompt)
                return [response.strip()]
            except Exception as e:
                logger.error(f"Error calling LLM for HyDE: {e}")
                
        # Mock HyDE fallback
        return [
            f"This document describes systems relating to the question: '{question}'. "
            f"Specifically, it provides an architectural overview and technical implementation details."
        ]
