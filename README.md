# Modular RAG Benchmarking Framework

A high-performance, modular benchmarking framework designed to evaluate and optimize retrieval-augmented generation (RAG) pipelines. The framework conducts a **combinatorial sweep across 5 research axes** (160 pruned configurations), performs multi-factor statistical analysis (ANOVA), runs machine-learning-based sensitivity analysis, and executes deep generative quality audits — all powered by **local LLM inference** via **Ollama** (with optional **Groq** cloud acceleration).

### 🔬 5 Research Axes

The framework systematically explores every valid combination across these structural dimensions:

| # | Axis | Options | Description |
|---|------|---------|-------------|
| 1 | **Chunking Strategy** | `fixed_512`, `fixed_1024`, `recursive`, `semantic` | How source documents are split into retrievable segments — from fixed-window token splits to cosine-distance semantic boundaries. |
| 2 | **Pre-Retrieval Transform** | `none`, `query_rewrite`, `hyde` | Whether user queries are augmented before search — via LLM-generated multi-query rewrites or Hypothetical Document Embedding (HyDE). |
| 3 | **Core Retrieval Matcher** | `dense_cosine`, `sparse_bm25`, `hybrid_rrf` | The search algorithm — dense vector cosine similarity, lexical BM25 term matching, or Reciprocal Rank Fusion (RRF) hybrid. |
| 4 | **Index Structure** | `flat`, `parent_document` | Whether retrieval operates on raw chunks (flat) or maps child-chunk hits back to their larger parent document for richer context. |
| 5 | **Post-Retrieval Processing** | `none`, `cross_encoder_rerank`, `contextual_compression` | Whether retrieved results are refined — via cross-encoder reranking for precision or contextual compression to strip noise. |

> **Pruning**: Incompatible pairs (`hyde` + `sparse_bm25`, `parent_document` + `contextual_compression`) are automatically filtered, reducing the search space from 216 to **160 valid configurations**.

### 📏 Evaluation Metrics

**Stage 1 — Retrieval Quality (automated, high-throughput):**
| Metric | Description |
|--------|-------------|
| **Hit Rate@5** | Binary: 1 if any relevant chunk appears in the top-5, 0 otherwise. |
| **Recall@5** | Fraction of ground-truth relevant chunks retrieved within the top-5. |
| **MRR@5** | Mean Reciprocal Rank — reciprocal of the rank position of the first relevant result. |
| **NDCG@5** | Normalized Discounted Cumulative Gain — position-weighted relevance using logarithmic decay. |

**Stage 2 — Generative Quality (LLM-as-a-Judge, RAGAS audit on Top 5 pipelines):**
| Metric | Description |
|--------|-------------|
| **Faithfulness** | Whether generated statements are grounded in and supported by the retrieved passages. |
| **Answer Relevancy** | How directly the response addresses the original query without extraneous content. |
| **Answer Correctness** | Semantic alignment between the generated answer and the ground-truth reference answer. |

### 📊 Statistical Analysis Pipeline

The framework applies a multi-layered analysis stack to identify which RAG design choices matter most:

1. **Five-Way Main-Effects ANOVA** — Isolates the independent contribution (Eta-Squared) of each axis to `Recall@5` variance.
2. **Two-Way Interaction ANOVA** — Detects pairwise synergy/conflict effects between all 10 axis pairs (e.g., does `hybrid_rrf` perform differently with `cross_encoder_rerank` vs `none`?).
3. **Three-Pillar ANOVA** — Groups the 5 axes into 3 architectural pillars (Ingestion, Retrieval, Post-Processing) for high-level design guidance.
5. **Pairwise Interaction Plots** — Generated from the ANOVA analysis; visual diagnostic plots for every factor combination, exported to `figures/`.

---

## 🚀 Getting Started

### 1. Setup Virtual Environment & Dependencies

You can use either **pip** or **uv** to manage dependencies.

**Option A — pip (traditional):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

**Option B — uv (fast, modern):**
```powershell
# Install dependencies via uv (uses pyproject.toml + uv.lock)
uv sync
```

### 2. Configure API Token
Create a `.env` file in the root directory (see `.env.example` for reference) and add your API token:
```env
# API Authentication Token
PUTER_TOKEN=your_jwt_token_here
```

### 3. Run the Benchmarking Suite
You can customize the execution using the `--max-questions` argument:

*   **Fast Verification Run (Recommended for testing setup)**:
    Runs the benchmarking sweep on a tiny subset of 3 questions to verify end-to-end execution quickly:
    ```powershell
    .\.venv\Scripts\python src/rag_bench.py --max-questions 3
    ```
    
*   **Full Benchmark Sweep**:
    Runs the benchmarking sweep on the full dataset (default: up to 150 questions):
    ```powershell
    .\.venv\Scripts\python src/rag_bench.py
    ```

### 4. Run Custom Analysis (Optional)
After the main sweep completes, run the non-ANOVA custom analysis for Random Forest sensitivity and interaction heatmaps:
```powershell
.\.venv\Scripts\python src/analysis/custom_analysis.py
```

### 5. Run Stage 2 Deep Evaluation (with Local Models)
After completing Stage 1 and ensuring you have generated `reports/stage1_screening_logs.csv`, you can run the Stage 2 deep evaluation using local LLMs via Ollama:
1. Ensure Ollama is installed and running.
2. Download the required Qwen models:
   ```powershell
   ollama pull qwen2.5:1.5b
   ollama pull qwen2.5:7b     # Used as the judge evaluator (or any default local model configured in code)
   ```
3. Run the Stage 2 evaluation script:
   ```powershell
   .\.venv\Scripts\python src/evaluation/stage2_with_local_models.py
   ```
This will evaluate the top 5 pipeline configurations against the ground truth using RAGAS-style semantic metrics (Faithfulness, Answer Relevancy, Answer Correctness) and write the results to `reports/ragas_evaluation_checkpoint_local_90.csv`.

### 6. Run Ablation Analysis (Optional)
To perform Dunnett's statistical test and generate ablation study reports comparing configurations back to baseline controls:
```powershell
.\.venv\Scripts\python src/analysis/ablation_analysis.py
```
This will save the statistical reports to `reports/rag_ablation_dunnett_report.csv` and visualize the confidence intervals under `figures/`.


## 📈 System Outputs & Reports

After running the benchmarking suite, outputs are saved inside the `reports/` and `figures/` directories:

### Reports (`reports/`)

*   📂 **`stage1_screening_logs.csv`**: Raw metrics logs (Hit Rate, Recall, MRR, NDCG) for all valid sweep pipelines.
*   📂 **`anova_main_effects.csv`**: ANOVA statistical tables mapping factor importance on retrieval performance.
*   📂 **`anova_interactions.csv`**: Two-way interaction ANOVA table (main + pairwise effects with Eta-Squared).
*   📂 **`anova_three_pillars.csv`**: Three-pillar ANOVA grouping analysis.
*   📂 **`anova_explanation.md`**: Detailed narrative explanation of the ANOVA results and their interpretation.
*   📂 **`pillar_analysis_summary.md`**: Text summary of performance by architecture pillars.
*   📂 **`stage2_deep_eval_results.csv`**: RAGAS semantic audit scores (Faithfulness, Relevancy, Correctness) on the **Top 5** surfaced configurations.
*   📂 **`ragas_evaluation_checkpoint_local_90.csv`**: RAGAS local evaluation checkpoint file from Ollama runs.
*   📂 **`rag_ablation_dunnett_report.csv`**: Ablation study Dunnett test report comparing configurations back to baselines.
*   📂 **`custom_analysis_report.md`**: Non-ANOVA custom analysis report with Random Forest sensitivity and top-10 rankings.

### Figures (`figures/`)

*   📊 **Pairwise interaction plots**: Diagnostic interaction plots visualizing all factor combinations on `recall_at_5`.
*   📊 **`custom_feature_importance.png`**: Random Forest axis importance bar chart.
*   📊 **`custom_interaction_heatmap.png`**: Retrieval vs Post-Retrieval interaction heatmap.

---

## 🛠️ Project Structure

```
├── data/
│   ├── external/
│   │   ├── data_fetching_documents.ipynb  # Notebook to fetch raw Confluence documents
│   │   └── data_fetching_questions.ipynb  # Notebook to fetch raw Confluence questions
│   ├── processed/
│   │   ├── embeddings/                    # Persistent ChromaDB vector collections
│   │   ├── questions/
│   │   │   └── questions.jsonl            # Ground truth evaluation questions (JSONL)
│   │   └── questions.json                 # Ground truth evaluation questions (JSON fallback)
│   ├── raw/
│   │   ├── confluence_documents.parquet   # Source Confluence documents dataset
│   │   ├── confluence_questions.parquet   # Source Confluence questions dataset
│   │   ├── *.md                           # Technical domain documents & distractor docs
│   │   └── ...
│   └── pre_retrieval_cache.json           # Cached LLM transformations (HyDE, Rewrite)
├── figures/                               # Generated diagnostic plots & charts
├── models/                                # Model artifacts (placeholder)
├── notebooks/
│   ├── embedding_process/
│   │   ├── c_512_embedding.ipynb          # Fixed-512 chunking & embedding pipeline
│   │   ├── c_1024_embedding.ipynb         # Fixed-1024 chunking & embedding pipeline
│   │   ├── c_rec_embedding.ipynb          # Recursive chunking & embedding pipeline
│   │   └── c_sem_embedding.ipynb          # Semantic chunking & embedding pipeline
│   └── questions_tranform/
│       └── questions_transforming.ipynb   # Question transformation & mapping notebook
├── reports/                               # Generated analysis reports & CSV outputs
├── src/
│   ├── core/
│   │   ├── cache_manager.py               # Disk-backed LLM transformation cache
│   │   ├── chunkers.py                    # Ingestion splitting logic
│   │   ├── index_manager.py               # Chroma & local overlay index managers
│   │   ├── llm_client.py                  # Groq & Ollama LLM client wrappers
│   │   ├── post_processors.py             # Re-ranking & context compression
│   │   ├── query_transforms.py            # Cache-backed query transformation dispatcher
│   │   └── retrievers.py                  # Cosine, BM25, and RRF searchers
│   ├── evaluation/
│   │   ├── stage1_screening.py            # High-throughput screening suite
│   │   └── stage2_with_local_models.py    # Generative RAGAS audit with local models
│   ├── analysis/
│   │   ├── statistical_analysis.py        # Statsmodels ANOVA fitters & interaction plots
│   │   ├── custom_analysis.py             # Random Forest sensitivity & custom reports
│   │   └── ablation_analysis.py           # Dunnett's test on screening logs
│   └── rag_bench.py                       # Main orchestrator entry point
├── tests/
│   └── mock_utils.py                      # Mock dataset generator & LLM response simulator
├── .env.example                           # Template for environment variables
├── implementation_plan.md                 # Project implementation plan & design notes
├── inspect_mapped_questions.py            # Utility: inspect question-to-chunk mapping quality
├── map_questions_to_chroma.py             # Utility: map ground truth questions to ChromaDB chunks
├── pyproject.toml                         # Project metadata & dependency specification (uv)
├── requirements.txt                       # Dependency specification (pip)
└── README.md                              # Framework documentation
```