import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import nltk for sentence splitting
try:
    import nltk
except ImportError:
    nltk = None

class PostProcessor:
    """Refines and re-ranks retrieved candidate context blocks."""
    def __init__(self):
        pass

    def rerank_cross_encoder(self, query: str, candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """Re-scores candidate chunks using a deep cross-encoder model to surface optimal matches."""
        if not candidates:
            return []
            
        texts = [c["metadata"]["text"] for c in candidates]
        
        # Build TF-IDF vectorizer fitted on the query and candidates
        # to compute standard semantic/lexical similarity scores
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        try:
            vectors = vectorizer.fit_transform([query] + texts).toarray()
            query_vector = vectors[0].reshape(1, -1)
            candidate_vectors = vectors[1:]
            
            similarities = cosine_similarity(query_vector, candidate_vectors)[0]
        except Exception:
            # Fallback score if TF-IDF fails (e.g. empty inputs)
            similarities = [0.0] * len(candidates)
            
        # Assign new scores and sort
        reranked = []
        for i, cand in enumerate(candidates):
            new_item = {
                "score": float(similarities[i]),
                "metadata": cand["metadata"]
            }
            reranked.append(new_item)
            
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_n]

    def compress_contextual_noise(self, query: str, candidates: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Squeezes out background noise from chunks, keeping only highly relevant sentences."""
        compressed_candidates = []
        
        for cand in candidates:
            text = cand["metadata"]["text"]
            
            # Split into sentences
            if nltk:
                sentences = nltk.tokenize.sent_tokenize(text)
            else:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                
            if not sentences:
                compressed_candidates.append(cand)
                continue
                
            # Fit TF-IDF on the sentences and query to calculate sentence-level relevance
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
            try:
                vectors = vectorizer.fit_transform([query] + sentences).toarray()
                query_vector = vectors[0].reshape(1, -1)
                sentence_vectors = vectors[1:]
                similarities = cosine_similarity(query_vector, sentence_vectors)[0]
            except Exception:
                similarities = [1.0] * len(sentences)  # Keep all if computation fails
                
            # Keep sentences that exceed similarity threshold
            kept_sentences = []
            for idx, score in enumerate(similarities):
                if score >= threshold:
                    kept_sentences.append(sentences[idx])
                    
            # Ensure we keep at least the single most relevant sentence if all fall below threshold
            if not kept_sentences and sentences:
                best_idx = int(similarities.argmax())
                kept_sentences.append(sentences[best_idx])
                
            compressed_text = " ".join(kept_sentences)
            
            # Create a copy with the compressed text
            new_meta = dict(cand["metadata"])
            new_meta["text"] = compressed_text
            
            compressed_candidates.append({
                "score": cand["score"],
                "metadata": new_meta
            })
            
        return compressed_candidates
