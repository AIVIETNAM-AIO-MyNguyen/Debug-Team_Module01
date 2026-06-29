# Understanding ANOVA in the RAG Benchmarking Framework

## Table of Contents

1. [What is ANOVA?](#1-what-is-anova)
2. [Why Use ANOVA in This Project?](#2-why-use-anova-in-this-project)
3. [Key Statistical Concepts](#3-key-statistical-concepts)
4. [Type 1: Five-Way Main-Effects ANOVA](#4-type-1-five-way-main-effects-anova)
5. [Type 2: Interaction ANOVA](#5-type-2-interaction-anova)
6. [Type 3: Three-Pillar ANOVA](#6-type-3-three-pillar-anova)
7. [How to Read the Output Tables](#7-how-to-read-the-output-tables)
8. [Summary & Comparison](#8-summary--comparison)

---

## 1. What is ANOVA?

**ANOVA** (Analysis of Variance) is a statistical method used to determine whether there are **statistically significant differences** between the means of **three or more groups**.

### The Core Idea

Imagine you're testing 4 different chunking strategies. Each one produces a set of `recall_at_5` scores. ANOVA asks:

> "Are the differences between these group means **real** (caused by the chunking strategy), or just **random noise** (caused by sampling variability)?"

It answers this by decomposing the **total variance** in the data into two parts:

| Variance Source | Meaning |
|---|---|
| **Between-group variance** | Differences *caused by* the factor (e.g., chunking strategy) |
| **Within-group variance** | Random variation *within* each group (noise / error) |

If the between-group variance is **much larger** than the within-group variance, the factor has a real effect.

---

## 2. Why Use ANOVA in This Project?

This benchmarking framework runs **160 valid RAG pipeline configurations**, each a unique combination of 5 design choices:

| Axis (Factor) | Options |
|---|---|
| **Chunking** | `fixed_512`, `fixed_1024`, `recursive`, `semantic` |
| **Pre-Retrieval** | `none`, `query_rewrite`, `hyde` |
| **Retrieval** | `dense_cosine`, `sparse_bm25`, `hybrid_rrf` |
| **Index Structure** | `flat`, `parent_document` |
| **Post-Retrieval** | `none`, `cross_encoder_rerank`, `contextual_compression` |

Simply looking at "which single configuration scored highest" is **not enough** because:

1. **One config winning might be a fluke** — it could have gotten lucky on certain questions.
2. **You can't see which *axis* matters** — did the winning config win because of its chunking? its retrieval? or the combination?
3. **There are 160 configs** — comparing them one-by-one is impractical.

### What ANOVA provides:

- **Factor Ranking**: Which axis explains the most variance in recall? (e.g., "Retrieval strategy accounts for 40% of performance variation")
- **Statistical Significance**: Is the effect real or random? (via p-values)
- **Interaction Detection**: Do certain factor *combinations* produce synergistic or antagonistic effects?
- **Actionable Engineering Insight**: Focus optimization effort on the axes that actually matter.

---

## 3. Key Statistical Concepts

Before diving into the three ANOVA types, here are the key terms used throughout:

### 3.1 Sum of Squares (SS)

Measures the total amount of variability. There are three types:

$$SS_{total} = \sum_{i=1}^{N}(y_i - \bar{y})^2$$

- **SS_between (SS_factor)**: Variability between group means — how much the group means differ from the overall mean.
- **SS_within (SS_residual)**: Variability within groups — how much individual observations differ from their own group mean.
- **SS_total** = SS_between + SS_within

### 3.2 Degrees of Freedom (df)

The number of independent values that can vary:

| Source | Formula |
|---|---|
| Factor with *k* levels | df = k − 1 |
| Residual (error) | df = N − number of model parameters |
| Total | df = N − 1 |

### 3.3 Mean Square (MS)

The average variance per degree of freedom:

$$MS = \frac{SS}{df}$$

### 3.4 F-Statistic

The ratio of between-group variance to within-group variance:

$$F = \frac{MS_{between}}{MS_{within}}$$

- **F ≈ 1**: The factor has no effect (between-group variance ≈ within-group variance)
- **F >> 1**: The factor has a significant effect (between-group variance >> within-group variance)

### 3.5 p-value (PR(>F))

The probability of observing an F-statistic this extreme **if the factor had no real effect** (null hypothesis).

| p-value | Interpretation |
|---|---|
| p < 0.001 | Extremely strong evidence of a real effect (***) |
| p < 0.01 | Strong evidence (**) |
| p < 0.05 | Moderate evidence (*) |
| p ≥ 0.05 | Insufficient evidence — effect may be random |

### 3.6 Eta-Squared (η²) — Effect Size

The **percentage of total variance** explained by a factor:

$$\eta^2 = \frac{SS_{factor}}{SS_{total}}$$

| η² Value | Interpretation |
|---|---|
| 0.01 (1%) | Small effect |
| 0.06 (6%) | Medium effect |
| 0.14 (14%+) | Large effect |

> **Note**: Unlike p-values (which depend on sample size), η² tells you **how much** a factor matters, not just whether it's statistically significant.

---

## 4. Type 1: Five-Way Main-Effects ANOVA

> **Implementation**: `StatisticalAnalyzer.execute_five_way_anova()` in `src/analysis/statistical_analysis.py`
> **Output**: `reports/anova_main_effects.csv`

### 4.1 Purpose

Answer the question: **"Which of the 5 RAG design axes has the largest independent effect on retrieval performance (`recall_at_5`)?"**

This treats each axis as an independent factor and measures its **main effect** — the average impact of that factor across all levels of the other factors.

### 4.2 The Statistical Model

An Ordinary Least Squares (OLS) linear model is fit with the formula:

```
recall_at_5 ~ C(chunking) + C(pre_retrieval) + C(retrieval) + C(index_structure) + C(post_retrieval)
```

Where:
- `~` means "is modeled as a function of"
- `C(...)` means "treat this variable as a **categorical** factor" (not a number)
- `+` means "additive main effects only" (no interactions)

This model assumes the effect of each factor is **independent** — the impact of chunking is the same regardless of which retrieval method is used.

### 4.3 How It's Calculated (Step by Step)

**Step 1: Fit the OLS model**
```python
model = ols("recall_at_5 ~ C(chunking) + C(pre_retrieval) + C(retrieval) + C(index_structure) + C(post_retrieval)", data=df).fit()
```
This creates dummy variables for each factor level and fits a linear regression.

**Step 2: Compute the ANOVA table using Type II Sum of Squares**
```python
anova_table = sm.stats.anova_lm(model, typ=2)
```

Type II SS tests each factor's contribution **after adjusting for all other factors**. This is appropriate when:
- The design is **unbalanced** (unequal sample sizes per cell) — which is the case here because incompatible configs were pruned (160 out of 216).
- You care about each factor's **unique** contribution.

**Step 3: Calculate eta-squared**
```python
anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
```

### 4.4 Example Output

| Source | sum_sq | df | F | PR(>F) | eta_sq |
|---|---|---|---|---|---|
| C(chunking) | 0.4521 | 3 | 18.42 | 0.0001 | 0.35 |
| C(retrieval) | 0.3104 | 2 | 15.81 | 0.0003 | 0.24 |
| C(pre_retrieval) | 0.1832 | 2 | 9.33 | 0.0015 | 0.14 |
| C(index_structure) | 0.0412 | 1 | 4.19 | 0.0420 | 0.03 |
| C(post_retrieval) | 0.0289 | 2 | 1.47 | 0.2310 | 0.02 |
| Residual | 0.2742 | 149 | — | — | 0.21 |

**Reading this example**: Chunking explains 35% of variance (η²=0.35), retrieval explains 24%, and post-retrieval explains only 2% and is not statistically significant (p=0.231). Engineering effort should focus on chunking and retrieval strategies.

### 4.5 Limitations

- Assumes **no interaction effects** — the effect of chunking is the same no matter what retrieval is used.
- Cannot detect synergistic or antagonistic pairings.
- This is why we also run Interaction ANOVA.

---

## 5. Type 2: Interaction ANOVA

> **Implementation**: `StatisticalAnalyzer.execute_interaction_anova()` in `src/analysis/statistical_analysis.py`
> **Output**: `reports/anova_interactions.csv`

### 5.1 Purpose

Answer the question: **"Do certain pairs of design choices produce effects that are greater (or worse) than the sum of their individual effects?"**

An interaction occurs when the effect of one factor **depends on** the level of another factor. For example:
- `semantic` chunking might work great with `dense_cosine` retrieval but poorly with `sparse_bm25`
- `hyde` pre-retrieval might boost `hybrid_rrf` much more than it boosts `dense_cosine`

### 5.2 The Statistical Model

The model includes all 5 main effects **plus** all 10 pairwise two-way interaction terms:

```
recall_at_5 ~ C(chunking) + C(pre_retrieval) + C(retrieval) + C(index_structure) + C(post_retrieval)
            + C(chunking):C(pre_retrieval)
            + C(chunking):C(retrieval)
            + C(chunking):C(index_structure)
            + C(chunking):C(post_retrieval)
            + C(pre_retrieval):C(retrieval)
            + C(pre_retrieval):C(index_structure)
            + C(pre_retrieval):C(post_retrieval)
            + C(retrieval):C(index_structure)
            + C(retrieval):C(post_retrieval)
            + C(index_structure):C(post_retrieval)
```

Where `:` denotes an interaction term — it captures the **combined** effect of two factors beyond what each contributes individually.

### 5.3 How Interactions Are Generated (Step by Step)

**Step 1: Identify valid factors** (must have >1 unique level)
```python
valid_factors = [f for f in factors if f in self.df.columns and self.df[f].nunique() > 1]
```

**Step 2: Generate all C(k,2) = 10 pairwise interaction terms**
```python
import itertools
interaction_terms = [f"C({a}):C({b})" for a, b in itertools.combinations(valid_factors, 2)]
```

**Step 3: Build and fit the full formula**
```python
formula = f"{metric} ~ " + " + ".join(main_terms + interaction_terms)
model = ols(formula, data=self.df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
```

### 5.4 Understanding Interaction Effects

Consider two factors: chunking (A) and retrieval (B).

**No interaction (additive effects)**:
The benefit of switching from `fixed_512` to `semantic` chunking is the same regardless of whether you use `dense_cosine` or `sparse_bm25` retrieval.

```
                 dense_cosine    sparse_bm25
fixed_512:         0.60            0.50          (difference = 0.10)
semantic:          0.75            0.65          (difference = 0.10)  ← same gap
```

**Interaction present**:
The benefit of `semantic` chunking is much larger with `dense_cosine` than with `sparse_bm25`.

```
                 dense_cosine    sparse_bm25
fixed_512:         0.60            0.50          (difference = 0.10)
semantic:          0.85            0.52          (difference = 0.33)  ← different gap!
```

### 5.5 Interaction Plots

The framework also generates **visual interaction plots** (`figures/interaction_*.png`) for all 10 factor pairs. In these plots:
- **Parallel lines** = no interaction
- **Crossing or diverging lines** = interaction is present

### 5.6 Why This Matters for RAG Engineering

If a strong interaction exists between `chunking:retrieval`, you can't optimize them independently. You need to **co-optimize** — test specific *pairings* rather than picking the best chunking and best retrieval separately.

---

## 6. Type 3: Three-Pillar ANOVA

> **Implementation**: `StatisticalAnalyzer.execute_three_pillar_anova()` in `src/analysis/statistical_analysis.py`
> **Output**: `reports/anova_three_pillars.csv` and `reports/pillar_analysis_summary.md`

### 6.1 Purpose

Answer the question: **"At a high architectural level, does retrieval quality depend more on how you chunk, how you retrieve, or how complex your RAG pipeline is?"**

This is a **simplified, strategic** view that collapses 5 fine-grained axes into 3 intuitive mental-model pillars.

### 6.2 The Three Pillars

| Pillar | Source Factor(s) | Rationale |
|---|---|---|
| **Pillar 1: Chunking** | `chunking` directly | How you split documents determines what content units enter the index |
| **Pillar 2: Retrieval** | `retrieval` directly | The core matching algorithm determines which chunks are surfaced |
| **Pillar 3: RAG Complexity** | Derived from `pre_retrieval` + `index_structure` + `post_retrieval` | Measures "how advanced" the pipeline is beyond naive RAG |

### 6.3 How the Complexity Profile Is Computed

The three sub-techniques (`pre_retrieval`, `index_structure`, `post_retrieval`) are collapsed into a single ordinal variable:

```python
def map_complexity(row):
    count = 0
    if row['pre_retrieval'] != 'none':
        count += 1
    if row['index_structure'] != 'flat':
        count += 1
    if row['post_retrieval'] != 'none':
        count += 1
    if count == 0:
        return "Naive RAG"
    else:
        return f"Advanced ({count} Active)"
```

This produces categories like:

| Profile | Meaning |
|---|---|
| `Naive RAG` | No pre-retrieval, flat index, no post-retrieval |
| `Advanced (1 Active)` | One enhancement active |
| `Advanced (2 Active)` | Two enhancements active |
| `Advanced (3 Active)` | All three enhancements active |

### 6.4 The Statistical Model

```
recall_at_5 ~ C(chunking) + C(retrieval) + C(rag_complexity_profile)
```

This is a simpler three-factor ANOVA. The same OLS + Type II SS procedure applies:

```python
model = ols(formula, data=self.df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
```

### 6.5 Why a Separate Pillar ANOVA?

| Concern | Five-Way ANOVA | Three-Pillar ANOVA |
|---|---|---|
| Granularity | 5 individual axes | 3 strategic pillars |
| Audience | Engineers tuning specific knobs | Architects making high-level design decisions |
| Degrees of freedom | High (many parameters, can overfit with small N) | Low (fewer parameters, more robust) |
| Interpretability | "pre_retrieval has η²=0.05" | "Overall pipeline complexity has η²=0.08" |
| Question answered | "Which knob matters?" | "Which architectural pillar matters?" |

### 6.6 Example Insight

If the Three-Pillar ANOVA shows:

| Pillar | η² |
|---|---|
| Chunking | 0.35 |
| Retrieval | 0.28 |
| RAG Complexity | 0.04 |

**Conclusion**: Adding pre-retrieval transforms, parent-doc indices, and post-retrieval reranking (pipeline complexity) collectively explain only 4% of performance variance. Your engineering budget is better spent on **chunking and retrieval optimization** rather than adding more pipeline stages.

---

## 7. How to Read the Output Tables

Every ANOVA table produced by this framework has the same columns:

| Column | What It Means | What to Look For |
|---|---|---|
| **Source** | The factor or interaction term | Your design axes |
| **sum_sq** | Sum of Squares — total variability attributed to this source | Higher = more variance explained |
| **df** | Degrees of freedom | (k-1) for a factor with k levels |
| **F** | F-statistic — ratio of signal to noise | Higher = stronger effect |
| **PR(>F)** | p-value — probability this effect is random | **< 0.05** = statistically significant |
| **eta_sq** | Effect size — proportion of total variance explained | **> 0.14** = large effect |

### Decision-Making Flowchart

```
For each factor in the ANOVA table:
│
├── Is p-value < 0.05?
│   ├── YES → The factor has a statistically significant effect
│   │   └── Check η²:
│   │       ├── η² > 0.14 → LARGE effect — prioritize optimizing this axis
│   │       ├── η² > 0.06 → MEDIUM effect — worth investigating
│   │       └── η² > 0.01 → SMALL effect — low priority
│   │
│   └── NO → The factor does NOT have a significant effect
│       └── Variations in this axis don't meaningfully change performance
```

---

## 8. Summary & Comparison

| Aspect | Five-Way Main-Effects | Interaction ANOVA | Three-Pillar ANOVA |
|---|---|---|---|
| **Factors** | 5 individual axes | 5 main + 10 pairwise interactions | 3 macro pillars |
| **Formula terms** | 5 | 15 | 3 |
| **Key question** | Which single axis matters most? | Do factor *combinations* matter? | Which architectural pillar matters most? |
| **Detects interactions?** | ❌ No | ✅ Yes (all 2-way pairs) | ❌ No |
| **Granularity** | Fine-grained | Fine-grained + pairwise | Strategic / high-level |
| **Risk of overfitting** | Low | Medium (many terms with limited data) | Very low |
| **Best for** | Initial factor screening | Deep-dive into synergies | Executive-level decisions |
| **Output file** | `anova_main_effects.csv` | `anova_interactions.csv` | `anova_three_pillars.csv` |

### Why All Three Are Needed

1. **Five-Way ANOVA** gives you the **first filter** — which axes deserve attention.
2. **Interaction ANOVA** reveals **hidden synergies** — you can't optimize axes in isolation if they interact.
3. **Three-Pillar ANOVA** gives the **strategic summary** — should you invest in better chunking, better retrieval, or more pipeline complexity?

Together, they provide a complete statistical picture that transforms raw benchmark numbers into **actionable engineering decisions**.

---

## Appendix: Code Reference

| Method | File | Line | Output |
|---|---|---|---|
| `execute_five_way_anova()` | `src/analysis/statistical_analysis.py` | L26–63 | `reports/anova_main_effects.csv` |
| `execute_interaction_anova()` | `src/analysis/statistical_analysis.py` | L65–100 | `reports/anova_interactions.csv` |
| `execute_three_pillar_anova()` | `src/analysis/statistical_analysis.py` | L102–147 | `reports/anova_three_pillars.csv` |
| `generate_pillar_analysis_report()` | `src/analysis/statistical_analysis.py` | L149–242 | `reports/pillar_analysis_summary.md` |
| `generate_all_pairwise_plots()` | `src/analysis/statistical_analysis.py` | L244–300 | `figures/interaction_*.png` |
| `run_statistical_analysis()` | `src/rag_bench.py` | L218–263 | Orchestrates all of the above |
