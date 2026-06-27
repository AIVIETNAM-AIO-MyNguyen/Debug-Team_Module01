# Non-ANOVA Custom RAG Performance Analysis Report

This report provides a comprehensive alternative to classical ANOVA to evaluate how different RAG configuration choices impact retrieval performance metrics.

## 1. RAG Component Sensitivity Analysis (Machine Learning Feature Importance)

Rather than using linear F-tests (ANOVA), we trained a **Random Forest Regressor** on the one-hot encoded pipeline configurations to predict `recall_at_5`. By summing up the importances of the categories belonging to each axis, we obtain the total variance influence of each component:

| RAG Parameter Axis | Variance Explanation Importance |
|---|---|
| **retrieval** | 25.43% |
| **chunking** | 25.37% |
| **pre_retrieval** | 20.20% |
| **post_retrieval** | 15.81% |
| **index_structure** | 13.19% |

*(Visualization saved to `figures/custom_feature_importance.png`)*

## 2. Top 10 RAG Pipeline Configurations

Ranked by a composite score (`Recall@5 * 0.7 + MRR@5 * 0.3`):

| Rank | Pipeline ID | Recall@5 | MRR@5 | NDCG@5 | Composite Score |
|---|---|---|---|---|---|
| 1 | `P_semantic_query_rewrite_sparse_bm25_parent_document_none` | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 2 | `P_semantic_query_rewrite_sparse_bm25_flat_none` | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 3 | `P_semantic_query_rewrite_sparse_bm25_parent_document_cross_encoder_rerank` | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 4 | `P_semantic_query_rewrite_sparse_bm25_flat_contextual_compression` | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 5 | `P_semantic_query_rewrite_sparse_bm25_flat_cross_encoder_rerank` | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 6 | `P_fixed_1024_hyde_hybrid_rrf_flat_none` | 1.0000 | 0.8333 | 0.8770 | **0.9500** |
| 7 | `P_fixed_1024_none_sparse_bm25_flat_contextual_compression` | 1.0000 | 0.8333 | 0.8770 | **0.9500** |
| 8 | `P_fixed_1024_hyde_hybrid_rrf_flat_contextual_compression` | 1.0000 | 0.8333 | 0.8770 | **0.9500** |
| 9 | `P_fixed_1024_query_rewrite_hybrid_rrf_flat_contextual_compression` | 1.0000 | 0.8333 | 0.8770 | **0.9500** |
| 10 | `P_fixed_512_query_rewrite_sparse_bm25_parent_document_none` | 1.0000 | 0.8333 | 0.8770 | **0.9500** |

## 3. Performance Breakdown by Individual Axes

### Axis: Chunking

| Choice | Mean Recall@5 | Std Dev Recall | Mean MRR@5 | Mean NDCG@5 |
|---|---|---|---|---|
| `fixed_1024` | 0.5750 | 0.4964 | 0.4281 | 0.4650 |
| `semantic` | 0.5500 | 0.4996 | 0.4062 | 0.4429 |
| `recursive` | 0.4333 | 0.4976 | 0.3625 | 0.3804 |
| `fixed_512` | 0.4000 | 0.4920 | 0.2703 | 0.3029 |

### Axis: Pre Retrieval

| Choice | Mean Recall@5 | Std Dev Recall | Mean MRR@5 | Mean NDCG@5 |
|---|---|---|---|---|
| `query_rewrite` | 0.6167 | 0.4876 | 0.4644 | 0.5028 |
| `none` | 0.4944 | 0.5014 | 0.3778 | 0.4072 |
| `hyde` | 0.2917 | 0.4564 | 0.2037 | 0.2261 |

### Axis: Retrieval

| Choice | Mean Recall@5 | Std Dev Recall | Mean MRR@5 | Mean NDCG@5 |
|---|---|---|---|---|
| `sparse_bm25` | 0.6917 | 0.4637 | 0.6140 | 0.6333 |
| `hybrid_rrf` | 0.5667 | 0.4969 | 0.3983 | 0.4409 |
| `dense_cosine` | 0.2778 | 0.4492 | 0.1704 | 0.1977 |

### Axis: Index Structure

| Choice | Mean Recall@5 | Std Dev Recall | Mean MRR@5 | Mean NDCG@5 |
|---|---|---|---|---|
| `flat` | 0.5799 | 0.4944 | 0.4251 | 0.4642 |
| `parent_document` | 0.3542 | 0.4795 | 0.2793 | 0.2981 |

### Axis: Post Retrieval

| Choice | Mean Recall@5 | Std Dev Recall | Mean MRR@5 | Mean NDCG@5 |
|---|---|---|---|---|
| `contextual_compression` | 0.6354 | 0.4838 | 0.4925 | 0.5289 |
| `none` | 0.5156 | 0.5011 | 0.3879 | 0.4203 |
| `cross_encoder_rerank` | 0.3906 | 0.4892 | 0.2827 | 0.3098 |

## 4. Synergy Analysis: Retrieval vs Post-Retrieval Interaction Heatmap

The table below shows the average `recall_at_5` for different combinations of Retrieval and Post-Retrieval techniques to identify potential synergy effects:

| Retrieval \ Post-Retrieval | none | cross_encoder_rerank | contextual_compression |
|---|---|---|---|
| **dense_cosine** | 0.2639 | 0.2222 | 0.4167 |
| **hybrid_rrf** | 0.6250 | 0.4167 | 0.7500 |
| **sparse_bm25** | 0.7292 | 0.6042 | 0.7917 |

*(Visualization saved to `figures/custom_interaction_heatmap.png`)*

