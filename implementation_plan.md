# Implementation Plan - Modular RAG Benchmarking Framework

This document outlines the design and implementation strategy for building the modular, two-stage cascaded RAG benchmarking framework.

## 5-Axis Combinatorial Research Design

```mermaid
graph LR
    subgraph Axis1 ["Axis 1: Chunking (|C|=4)"]
        C1["fixed_512"]
        C2["fixed_1024"]
        C3["recursive"]
        C4["semantic"]
    end

    subgraph Axis2 ["Axis 2: Pre-Retrieval (|Pre|=3)"]
        P1["none"]
        P2["query_rewrite"]
        P3["hyde"]
    end

    subgraph Axis3 ["Axis 3: Core Retrieval (|R|=3)"]
        R1["dense_cosine"]
        R2["sparse_bm25"]
        R3["hybrid_rrf"]
    end

    subgraph Axis4 ["Axis 4: Index Structure (|Idx|=2)"]
        I1["flat"]
        I2["parent_document"]
    end

    subgraph Axis5 ["Axis 5: Post-Retrieval (|Post|=3)"]
        O1["none"]
        O2["cross_encoder_rerank"]
        O3["contextual_compression"]
    end

    Axis1 --> GRID["Combinatorial Grid: 4x3x3x2x3 = 216 configs"]
    Axis2 --> GRID
    Axis3 --> GRID
    Axis4 --> GRID
    Axis5 --> GRID
    GRID --> PRUNE["Validation Pruning: 216 -> 160"]
    PRUNE --> SWEEP["Stage 1 Screening Sweep"]
```

### Pruning Rules
| Rule | Condition | Reason |
|---|---|---|
| **1** | `hyde` + `sparse_bm25` | HyDE synthetic narratives collapse lexical BM25 matching |
| **2** | `parent_document` + `contextual_compression` | Compression strips the context padding that parent-doc indexing provides |

## Pipeline Execution Flow

```mermaid
graph TD
    subgraph Ingestion ["Ingestion & Indexing"]
        A["Raw Documents (8 source + 30 distractors)"] --> B["DocumentChunker"]
        B --> C{"Index Structure?"}
        C -- "flat" --> D["Index Chunks Directly"]
        C -- "parent_document" --> E["Child-to-Parent Mapping"]
        D --> F["IndexManager (Namespace-Isolated Stores)"]
        E --> F
    end

    subgraph Stage1 ["Stage 1: High-Throughput Screening (160 x 15 = 2400 runs)"]
        F --> G["For each valid pipeline config"]
        G --> H["Pre-Retrieval Transform (CacheManager)"]
        H --> I{"Retrieval?"}
        I -- "dense" --> J["Cosine Similarity"]
        I -- "sparse" --> K["BM25"]
        I -- "hybrid" --> L["RRF Fusion"]
        J --> M["Candidates"]
        K --> M
        L --> M
        M --> N{"Post-Retrieval?"}
        N -- "none" --> O["Top 5"]
        N -- "rerank" --> P["Cross-Encoder"] --> O
        N -- "compress" --> Q["Contextual Filter"] --> O
        O --> R["Metrics: Recall@5, MRR, NDCG@5, Hit Rate@5"]
    end

    subgraph Stats ["Statistical Analysis"]
        R --> S["stage1_screening_logs.csv"]
        S --> T["Main-Effects ANOVA (5 factors)"]
        S --> U["Interaction ANOVA (+ 10 two-way terms)"]
        T --> V["anova_main_effects.csv"]
        U --> W["anova_interactions.csv"]
        S --> X["10 Pairwise Interaction Plots"]
    end

    subgraph Stage2 ["Stage 2: Deep Generative Review (Top 5)"]
        S --> Y["Filter Top 5 Pipelines"]
        Y --> Z["Generate RAG Response (context-aware)"]
        Z --> AA["Local TF-IDF Semantic Scoring"]
        AA --> AB["Faithfulness / Relevancy / Correctness"]
        AB --> AC["stage2_deep_eval_results.csv"]
    end
```

## 1. Storage Isolation Design in `index_manager.py`

To isolate the storage of various chunking and indexing combinations during the 216-loop screening sweep, we will implement **Structured Tag Namespaces** within `IndexManager`:
- Rather than wiping a single vector collection or keyword index on every loop iteration, `IndexManager` will maintain active collections in dictionaries keyed by a combined identifier: `f"{strategy_name}_{index_structure}"` (e.g., `fixed_512_flat`, `recursive_parent_document`).
- This design ensures that:
  1. **Efficiency**: Indices are built exactly *once* per chunking strategy and index structure. Subsequent loop iterations querying the same structure access the pre-built indices instantly.
  2. **Isolation**: No data leakage occurs between different configurations.
  3. **Simplicity**: No external database process is required. We will implement a lightweight, NumPy-accelerated local vector index and a Python-native BM25 keyword index.

---

## 2. Core Modules & Implementation Strategy

We will implement the following 10 Python files under the `src/` directory:

### 1. `chunkers.py`
- Implements `DocumentChunker` with three splitting algorithms:
  - `split_fixed_window`: Slices text into fixed token-limit chunks with custom overlap.
  - `split_recursive`: Slices structurally using hierarchy (`separators=["\n\n", "\n", " ", ""]`).
  - `split_semantic`: Computes sentence-level embeddings using a local HuggingFace model (`all-MiniLM-L6-v2`), calculates cosine distances between adjacent sentences, and splits where distance shifts exceed the `threshold_percentile`.

### 2. `cache_manager.py`
- Implements `LocalCacheManager` to serialize LLM transformations (`query_rewrite` and `hyde`) once per question.
- Saves cache to `data/pre_retrieval_cache.json` using JSON.
- Provides mock generator fallbacks if no LLM key is configured.

### 3. `index_manager.py`
- Manages local BM25 and dense vector collections using namespace tags.
- Uses `sentence-transformers` (or a local HuggingFace pipeline equivalent) for vector embeddings.
- Supports `parent_document` structure by maintaining a mapping of child chunk IDs to parent chunk IDs.

### 4. `query_transforms.py`
- Handles runtime query transformations (`none`, `query_rewrite`, `hyde`) by retrieving cached transformations.

### 5. `retrievers.py`
- Implements core retrieval methods:
  - `search_dense`: Standard cosine similarity search.
  - `search_sparse`: BM25 lexical keyword lookup.
  - `fuse_hybrid_rrf`: Interleaves search results using Reciprocal Rank Fusion (RRF).

### 6. `post_processors.py`
- Implements post-retrieval refinement:
  - `rerank_cross_encoder`: Re-ranks retrieved chunks using a local cross-encoder model (or a lightweight equivalent).
  - `compress_contextual_noise`: Sentence-level filtering to retain only highly relevant sentences.

### 7. `stage1_screening.py`
- Coordinates high-throughput evaluation of the combinations.
- Computes IR metrics bounded between `0.0` and `1.0`:
  - **Hit Rate@5**: `1.0` if any target chunk ID is retrieved, else `0.0`.
  - **Recall@5**: Fraction of target chunk IDs successfully retrieved.
  - **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant chunk.
  - **NDCG@5**: Normalized Discounted Cumulative Gain.
- Evaluates configurations, prunes clashing paths, and logs performance.

### 8. `statistical_analysis.py`
- Runs a Five-Way ANOVA using `statsmodels` to identify the most significant modular axes.
- Generates interaction plots and saves them to `figures/`.

### 9. `stage2_deep_eval.py`
- Executes generative testing on the top 5 pipelines.
- Supports RAGAS-style semantic audits (Faithfulness, Answer Relevancy, Answer Correctness).

### 10. `rag_bench.py`
- Main entry point orchestrating data loading, validation, screening, statistical analysis, and deep generative review.