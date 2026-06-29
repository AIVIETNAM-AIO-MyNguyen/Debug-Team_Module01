# Pillar-level RAG Benchmarking Analysis Report

This report evaluates the RAG configurations structured across the **3 mental model pillars** (Chunking, Retrieval, and RAG Techniques/Complexity) on the metric `recall_at_5`.

## 1. 3-Pillar ANOVA & Effect Sizes (Variance Explained)

The table below shows the statistical significance and the percentage of variance explained ($\eta^2$) by each of the 3 main pillars:

| Factor (Pillar) | Sum of Squares (SS) | Deg. of Freedom (DF) | F-Statistic | p-value | Effect Size ($\eta^2$) |
|---|---|---|---|---|---|
| **C(chunking)** | 5.3597 | 3 | 11.3081 | 2.1106e-07 | **0.29%** |
| **C(retrieval)** | 243.8403 | 2 | 771.6962 | 7.0803e-313 | **13.01%** |
| **C(rag_complexity_profile)** | 9.0583 | 3 | 19.1116 | 2.3552e-12 | **0.48%** |
| **Residual** | 1616.3937 | 10231 | N/A | N/A | **86.22%** |

## 2. Pillar 1: Chunking Strategy Performance

| Chunking Strategy | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `fixed_1024` | 0.3365 | 0.4347 | 2560 |
| `recursive` | 0.3321 | 0.4303 | 2560 |
| `fixed_512` | 0.3211 | 0.4328 | 2560 |
| `semantic` | 0.2787 | 0.4138 | 2560 |

## 3. Pillar 2: Retrieval Strategy Performance

| Retrieval Strategy | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `sparse_bm25` | 0.4742 | 0.4527 | 2560 |
| `hybrid_rrf` | 0.4114 | 0.4481 | 3840 |
| `dense_cosine` | 0.1181 | 0.2957 | 3840 |

## 4. Pillar 3: RAG Techniques / Complexity Performance

This groups configurations based on composite complexity profiles (Naive RAG vs. Advanced RAG with 1, 2, or 3 active techniques).

| RAG Complexity Profile | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| **Naive RAG** | 0.4036 | 0.4494 | 768 |
| **Advanced (1 Active)** | 0.3445 | 0.4368 | 3584 |
| **Advanced (2 Active)** | 0.2979 | 0.4216 | 4608 |
| **Advanced (3 Active)** | 0.2576 | 0.4033 | 1280 |

### Sub-Technique Performance Breakdown (Pillar 3 Components)

#### A. Pre-Retrieval Methods
| Method | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `none` | 0.3459 | 0.4328 | 3840 |
| `query_rewrite` | 0.3226 | 0.4334 | 3840 |
| `hyde` | 0.2658 | 0.4099 | 2560 |

#### B. Index Structures
| Structure | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `flat` | 0.3533 | 0.4411 | 6144 |
| `parent_document` | 0.2628 | 0.4030 | 4096 |

#### C. Post-Retrieval Refinement
| Refinement | Mean Recall@5 | Std Dev | Sample Size |
|---|---|---|---|
| `contextual_compression` | 0.3694 | 0.4474 | 2048 |
| `none` | 0.3108 | 0.4278 | 4096 |
| `cross_encoder_rerank` | 0.2973 | 0.4175 | 4096 |

