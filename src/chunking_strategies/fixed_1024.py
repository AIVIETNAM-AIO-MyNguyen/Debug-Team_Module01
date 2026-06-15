from dataclasses import dataclass
import uuid

@dataclass
class Chunk:
    chunk_id: str       
    doc_id: str         
    text: str           
    start_char: int     
    end_char: int       
    strategy: str   

def chunk_1024(text: str, doc_id: str, size: int = 1024, overlap: int = 204) -> list[Chunk]:
    chunks = []
    text_len = len(text)
    start_ptr = 0

    while start_ptr < text_len:
        # Slice the window based on the maximum allowed size
        end_ptr = min(start_ptr + size, text_len)
        chunk_text = text[start_ptr:end_ptr]
        
        # Adjust window to prevent breaking words
        if end_ptr < text_len and not text[end_ptr].isspace() and not text[end_ptr-1].isspace():
            last_space = chunk_text.rfind(' ')
            if last_space > 0:
                end_ptr = start_ptr + last_space
                chunk_text = text[start_ptr:end_ptr]

        # Create the Chunk object
        chunks.append(Chunk(
            chunk_id="c_1024_"+ str(uuid.uuid4()),
            doc_id=doc_id,
            text=chunk_text,
            start_char=start_ptr,
            end_char=end_ptr,
            strategy="c_1024"
        ))

        # Advance the pointer, subtracting the overlap
        start_ptr = end_ptr - overlap
        
        # Prevent infinite loops if overlap is larger than the window step
        if start_ptr <= chunks[-1].start_char:
            start_ptr = end_ptr

    return chunks