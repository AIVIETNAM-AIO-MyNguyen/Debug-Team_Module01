import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

# Try to import nltk for sentence/word tokenization
try:
    import nltk
    # Proactively check if punkt is available, otherwise download silently
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    nltk = None

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    strategy: str

class DocumentChunker:
    """Handles parsing and coordinate-aware token splitting of text corpora."""
    def __init__(self, overlap_percentage: float = 0.20):
        self.overlap = overlap_percentage

    def _get_words_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """Extracts words along with their start and end character positions."""
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]

    def split_fixed_window(self, text: str, token_limit: int) -> List[Chunk]:
        """Slices text into uniform token counts with fixed boundaries."""
        words_info = self._get_words_with_offsets(text)
        if not words_info:
            return []
        
        chunks = []
        step = int(token_limit * (1.0 - self.overlap))
        if step <= 0:
            step = 1

        idx = 0
        chunk_count = 0
        while idx < len(words_info):
            # Take a window of words_info
            window = words_info[idx:idx + token_limit]
            if not window:
                break
            
            start_char = window[0][1]
            end_char = window[-1][2]
            chunk_text = text[start_char:end_char]
            
            chunk_id = f"c_fixed_{token_limit}_{chunk_count:03d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id="",  # Filled by index manager or caller
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                strategy=f"fixed_{token_limit}"
            ))
            
            chunk_count += 1
            idx += step
            
            # Avoid infinite loop if step is 0 or window didn't advance
            if len(window) < token_limit:
                break
                
        return chunks

    def split_recursive(self, text: str, separators: List[str] = None) -> List[Chunk]:
        """Slices text using a structural hierarchy (paragraphs, newlines, spaces)."""
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        token_limit = 512  # Standard recursive chunk size
        overlap_tokens = int(token_limit * self.overlap)
        
        # Helper to split text recursively and return character offsets
        def _recursive_split(subtext: str, offset: int, level: int) -> List[Tuple[str, int, int]]:
            if len(subtext.split()) <= token_limit or level >= len(separators):
                return [(subtext, offset, offset + len(subtext))]
            
            separator = separators[level]
            splits = []
            if separator == "":
                # Character level split
                splits = [(c, offset + i, offset + i + 1) for i, c in enumerate(subtext)]
            else:
                # Find all occurrences of the separator
                parts = subtext.split(separator)
                curr_offset = offset
                for i, part in enumerate(parts):
                    if part:
                        splits.append((part, curr_offset, curr_offset + len(part)))
                    curr_offset += len(part)
                    if i < len(parts) - 1:
                        curr_offset += len(separator)
            
            # Recursively process the splits that are too large
            final_splits = []
            for part_text, start, end in splits:
                if len(part_text.split()) > token_limit:
                    final_splits.extend(_recursive_split(part_text, start, level + 1))
                else:
                    final_splits.append((part_text, start, end))
            return final_splits

        raw_splits = _recursive_split(text, 0, 0)
        chunks = []
        chunk_count = 0
        
        # Merge adjacent splits to fit token_limit with overlap
        current_words = []
        current_splits = []
        
        for part_text, start, end in raw_splits:
            part_words = part_text.split()
            # If adding this split exceeds token_limit, we build a chunk
            if len(current_words) + len(part_words) > token_limit:
                if current_splits:
                    # Create chunk
                    start_char = current_splits[0][1]
                    end_char = current_splits[-1][2]
                    chunk_text = text[start_char:end_char]
                    chunks.append(Chunk(
                        chunk_id=f"c_rec_{chunk_count:03d}",
                        doc_id="",
                        text=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                        strategy="recursive"
                    ))
                    chunk_count += 1
                    
                    # Backtrack to handle overlap
                    # We keep the last few splits that fit the overlap requirement
                    backtrack_splits = []
                    backtrack_words_count = 0
                    for split in reversed(current_splits):
                        s_words = len(split[0].split())
                        if backtrack_words_count + s_words <= overlap_tokens:
                            backtrack_splits.insert(0, split)
                            backtrack_words_count += s_words
                        else:
                            break
                    current_splits = backtrack_splits
                    current_words = []
                    for s in current_splits:
                        current_words.extend(s[0].split())
            
            current_splits.append((part_text, start, end))
            current_words.extend(part_words)
            
        # Add the last chunk if any remains
        if current_splits:
            start_char = current_splits[0][1]
            end_char = current_splits[-1][2]
            chunk_text = text[start_char:end_char]
            chunks.append(Chunk(
                chunk_id=f"c_rec_{chunk_count:03d}",
                doc_id="",
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                strategy="recursive"
            ))
            
        return chunks

    def split_semantic(self, text: str, embedding_model_name: str, threshold_percentile: float = 95.0) -> List[Chunk]:
        """Slices text based on embedding distance shifts between adjacent sentences."""
        # Split text into sentences
        if nltk:
            sentences = nltk.tokenize.sent_tokenize(text)
        else:
            # Fallback regex sentence tokenizer
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
        if not sentences:
            return []
            
        # Get character offsets for sentences
        sentences_with_offsets = []
        curr_idx = 0
        for sent in sentences:
            start = text.find(sent, curr_idx)
            if start == -1:
                start = text.find(sent)  # Fallback search
            if start != -1:
                sentences_with_offsets.append((sent, start, start + len(sent)))
                curr_idx = start + len(sent)
            else:
                sentences_with_offsets.append((sent, curr_idx, curr_idx + len(sent)))
                curr_idx += len(sent)

        # Generate embeddings for each sentence using a local TF-IDF model as robust fallback,
        # or load via sentence-transformers/transformers if available.
        # For simplicity and robust local execution, we use a sentence-level TF-IDF fit
        # on the document's sentences itself or character/n-gram features.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Fit vectorizer on sentences to get dense representations
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        try:
            embeddings = vectorizer.fit_transform(sentences).toarray()
        except ValueError:
            # If fitting fails (e.g. empty/short text), use one-hot vectors
            embeddings = np.eye(len(sentences))
            
        # Compute cosine similarity between adjacent sentences
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i+1].reshape(1, -1))[0][0]
            distances.append(1.0 - sim)

        # Determine threshold
        if distances:
            threshold = np.percentile(distances, threshold_percentile)
        else:
            threshold = 1.0

        # Group sentences based on threshold splits
        chunks = []
        chunk_count = 0
        current_sentences = []
        
        for i, (sent, start, end) in enumerate(sentences_with_offsets):
            current_sentences.append((sent, start, end))
            
            # If distance to next sentence is greater than threshold, split
            if i < len(distances) and distances[i] >= threshold:
                start_char = current_sentences[0][1]
                end_char = current_sentences[-1][2]
                chunk_text = text[start_char:end_char]
                chunks.append(Chunk(
                    chunk_id=f"c_sem_{chunk_count:03d}",
                    doc_id="",
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    strategy="semantic"
                ))
                chunk_count += 1
                current_sentences = []
                
        # Add remaining sentences as the last chunk
        if current_sentences:
            start_char = current_sentences[0][1]
            end_char = current_sentences[-1][2]
            chunk_text = text[start_char:end_char]
            chunks.append(Chunk(
                chunk_id=f"c_sem_{chunk_count:03d}",
                doc_id="",
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                strategy="semantic"
            ))
            
        return chunks
