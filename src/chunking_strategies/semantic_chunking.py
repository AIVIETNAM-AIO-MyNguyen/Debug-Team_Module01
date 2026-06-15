from dataclasses import dataclass
import uuid
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings 

@dataclass
class Chunk:
    chunk_id: str       
    doc_id: str         
    text: str           
    start_char: int     
    end_char: int       
    strategy: str       

def semantic_chunk_with_ollama(text: str, doc_id: str) -> list[Chunk]:
    #NOTE: Must start Ollama bge-m3 before
    embeddings = OllamaEmbeddings(model="bge-m3") 
    
    splitter = SemanticChunker(
        embeddings=embeddings,
        add_start_index=True,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.7
    )
    
    lc_docs = splitter.create_documents(texts=[text])
    
    chunks = []
    for doc in lc_docs:
        start_char = doc.metadata.get("start_index", 0)
        end_char = start_char + len(doc.page_content)
        
        chunks.append(Chunk(
            chunk_id="c_sem_"+ str(uuid.uuid4()),
            doc_id=doc_id,
            text=doc.page_content,
            start_char=start_char,
            end_char=end_char,
            strategy="c_sem"
        ))
        
    return chunks
