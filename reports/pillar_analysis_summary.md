# Pillar-level RAG Benchmarking Analysis Report

This report evaluates the RAG configurations structured across the **3 mental model pillars** (Chunking, Retrieval, and RAG Techniques/Complexity) on the metric `recall_at_5`.

## 1. 3-Pillar ANOVA & Effect Sizes (Variance Explained)

The table below shows the statistical significance and the percentage of variance explained ($\eta^2$) by each of the 3 main pillars:

| Factor (Pillar) | Sum of Squares (SS) | Deg. of Freedom (DF) | F-Statistic | p-value | Effect Size ($\eta^2$) |
|---|---|---|---|---|---|
| **C(chunking)** | 3.0341 | 3 | 8.3504 | 1.5211e-05 | **0.15%** |
| **C(retrieval)** | 129.3142 | 2 | 533.8400 | 1.0474e-224 | **6.57%** |
| **C(rag_complexity_profile)** | 15.7597 | 3 | 43.3732 | 6.7151e-28 | **0.80%** |
| **Residual** | 1820.5095 | 15031 | N/A | N/A | **92.48%** |

## 2. Pillar 1: Chunking Strategy Performance

| Chunking Strategy | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `recursive` | 0.1997 | 0.3698 | 3760 |
| `fixed_1024` | 0.1972 | 0.3686 | 3760 |
| `fixed_512` | 0.1866 | 0.3639 | 3760 |
| `semantic` | 0.1637 | 0.3450 | 3760 |

## 3. Pillar 2: Retrieval Strategy Performance

| Retrieval Strategy | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `sparse_bm25` | 0.2655 | 0.4095 | 3760 |
| `hybrid_rrf` | 0.2559 | 0.4056 | 5640 |
| `dense_cosine` | 0.0653 | 0.2233 | 5640 |

## 4. Pillar 3: RAG Techniques / Complexity Performance

This groups configurations based on composite complexity profiles (Naive RAG vs. Advanced RAG with 1, 2, or 3 active techniques).

| RAG Complexity Profile | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| **Naive RAG** | 0.2748 | 0.4158 | 1128 |
| **Advanced (1 Active)** | 0.2141 | 0.3812 | 5264 |
| **Advanced (2 Active)** | 0.1627 | 0.3428 | 6768 |
| **Advanced (3 Active)** | 0.1445 | 0.3245 | 1880 |

### Sub-Technique Performance Breakdown (Pillar 3 Components)

#### A. Pre-Retrieval Methods
| Method | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `none` | 0.2355 | 0.3919 | 5640 |
| `hyde` | 0.1845 | 0.3625 | 3760 |
| `query_rewrite` | 0.1396 | 0.3227 | 5640 |

#### B. Index Structures
| Structure | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `flat` | 0.2057 | 0.3773 | 9024 |
| `parent_document` | 0.1584 | 0.3363 | 6016 |

#### C. Post-Retrieval Refinement
| Refinement | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `contextual_compression` | 0.2156 | 0.3853 | 3008 |
| `none` | 0.1828 | 0.3602 | 6016 |
| `cross_encoder_rerank` | 0.1763 | 0.3514 | 6016 |

