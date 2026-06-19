from dataclasses import dataclass
from typing import Any, Literal
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str       
    doc_id: str         
    text: str           
    start_char: int     
    end_char: int       
    strategy: str       

# ── Chunker ───────────────────────────────────────────────────────────────────
def chunk_with_langchain(
    text: str, 
    doc_id: str, 
    size: int = 1024, 
    overlap: int = 204
) -> list[Chunk]:
    
    # Passing size, overlap, and the vital index flag into **kwargs
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
        is_separator_regex=False,
        chunk_size=size,         # Handled by **kwargs
        chunk_overlap=overlap,   # Handled by **kwargs
        add_start_index=True     # Handled by **kwargs -> tracking start_char
    )
    
    # Using the exact signature of create_documents
    lc_docs = splitter.create_documents(texts=[text], metadatas=None)
    
    chunks = []
    for doc in lc_docs:
        start_char = doc.metadata["start_index"]
        end_char = start_char + len(doc.page_content)
        
        chunks.append(Chunk(
            chunk_id="c_rec_"+ str(uuid.uuid4()),
            doc_id=doc_id,
            text=doc.page_content,
            start_char=start_char,
            end_char=end_char,
            strategy="c_rec"
        ))
        
    return chunks
