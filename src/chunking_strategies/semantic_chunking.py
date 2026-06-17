from dataclasses import dataclass
from typing import Literal
import uuid
import torch
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load once at module level (same pattern as your ingestion script)
_st_model = SentenceTransformer(MODEL_NAME, device=DEVICE) # type: ignore
if DEVICE.type == "cuda":
    _st_model = _st_model.half()


# ── LangChain-compatible wrapper ──────────────────────────────────────────────
class STEmbeddings(Embeddings):
    """Thin LangChain Embeddings adapter around a SentenceTransformer model."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:   str
    doc_id:     str
    text:       str
    start_char: int
    end_char:   int
    strategy:   str


# ── Chunker ───────────────────────────────────────────────────────────────────
def semantic_chunk(
    text: str,
    doc_id: str,
    *,
    embeddings: Embeddings | None = None,
    breakpoint_threshold_type: Literal["percentile"] = "percentile",
    breakpoint_threshold_amount: float = 0.7,
) -> list[Chunk]:
    """
    Semantically chunk *text* using all-MiniLM-L6-v2 (default) or any
    LangChain-compatible Embeddings object passed via *embeddings*.
    """
    emb = embeddings or STEmbeddings(_st_model)

    splitter = SemanticChunker(
        embeddings=emb,
        add_start_index=True,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    chunks: list[Chunk] = []
    for doc in splitter.create_documents(texts=[text]):
        start = doc.metadata.get("start_index", 0)
        chunks.append(Chunk(
            chunk_id=  "c_sem_" + str(uuid.uuid4()),
            doc_id=    doc_id,
            text=      doc.page_content,
            start_char=start,
            end_char=  start + len(doc.page_content),
            strategy=  "c_sem",
        ))

    return chunks