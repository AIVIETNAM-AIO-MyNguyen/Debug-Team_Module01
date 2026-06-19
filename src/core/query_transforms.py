from typing import List
from .cache_manager import LocalCacheManager

class QueryTransformer:
    """Applies runtime pre-retrieval query adjustments using cached structures."""
    def __init__(self, cache_manager: LocalCacheManager):
        self.cache = cache_manager

    def execute_transform(self, query_id: str, raw_query: str, strategy: str) -> List[str]:
        """Returns a list of search strings based on the active strategy ('none', 'rewrite', 'hyde')."""
        strat = strategy.lower().strip()
        
        if strat == "none":
            return [raw_query]
            
        elif strat == "query_rewrite":
            cached_rewrites = self.cache.get_cached_transform(query_id, "query_rewrite")
            if cached_rewrites:
                return cached_rewrites
            return [raw_query]
            
        elif strat == "hyde":
            cached_hyde = self.cache.get_cached_transform(query_id, "hyde")
            if cached_hyde:
                return cached_hyde
            return [raw_query]
            
        else:
            # Fallback
            return [raw_query]
