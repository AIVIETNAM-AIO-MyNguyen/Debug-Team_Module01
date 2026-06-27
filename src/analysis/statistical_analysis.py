import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
# Set backend to Agg to prevent headless display errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Runs Multi-Factor ANOVA operations on screening dataset logs."""
    def __init__(self, performance_csv_path: str):
        self.performance_csv_path = performance_csv_path
        if not os.path.exists(performance_csv_path):
            raise FileNotFoundError(f"Performance log CSV not found at {performance_csv_path}")
        self.df = pd.read_csv(performance_csv_path)
        logger.info(f"Loaded {len(self.df)} performance log records for statistical analysis.")

    def execute_five_way_anova(self) -> Optional[pd.DataFrame]:
        """Fits an Ordinary Least Squares linear model to compute a Five-Way ANOVA table with effect sizes (eta_sq)."""
        factors = ["chunking", "pre_retrieval", "retrieval", "index_structure", "post_retrieval"]
        
        # Check if we have variance in the dependent variable
        if self.df["recall_at_5"].nunique() <= 1:
            logger.warning("No variance in recall_at_5. Skipping ANOVA calculation.")
            return None
            
        # Dynamically filter factors with more than one unique value to avoid statsmodels rank errors
        valid_factors = []
        for f in factors:
            if f in self.df.columns:
                num_levels = self.df[f].nunique()
                if num_levels > 1:
                    valid_factors.append(f)
                else:
                    logger.info(f"Skipping factor '{f}' in ANOVA model (only has 1 unique level).")
                    
        if not valid_factors:
            logger.warning("No factors have multiple levels. Cannot fit ANOVA model.")
            return None
            
        # Construct formulas: recall_at_5 ~ C(f1) + C(f2) + ...
        formula = "recall_at_5 ~ " + " + ".join([f"C({f})" for f in valid_factors])
        logger.info(f"Fitting ANOVA model with formula: {formula}")
        
        try:
            model = ols(formula, data=self.df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Calculate eta-squared (percentage of variance explained)
            anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
            logger.info("Successfully calculated ANOVA table.")
            return anova_table
        except Exception as e:
            logger.error(f"Error calculating ANOVA: {e}")
            return None

    def execute_interaction_anova(self, metric: str = "recall_at_5") -> Optional[pd.DataFrame]:
        """Fits an OLS model with main effects AND 2-way interaction terms, including eta-squared."""
        factors = ["chunking", "pre_retrieval", "retrieval", "index_structure", "post_retrieval"]

        if self.df[metric].nunique() <= 1:
            logger.warning(f"No variance in {metric}. Skipping interaction ANOVA.")
            return None

        valid_factors = [f for f in factors if f in self.df.columns and self.df[f].nunique() > 1]
        if len(valid_factors) < 2:
            logger.warning("Need at least 2 multi-level factors for interaction ANOVA.")
            return None

        # Build main effects
        main_terms = [f"C({f})" for f in valid_factors]

        # Build all 2-way interaction terms
        import itertools
        interaction_terms = [
            f"C({a}):C({b})" for a, b in itertools.combinations(valid_factors, 2)
        ]

        formula = f"{metric} ~ " + " + ".join(main_terms + interaction_terms)
        logger.info(f"Fitting interaction ANOVA with formula: {formula}")

        try:
            model = ols(formula, data=self.df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Calculate eta-squared (percentage of variance explained)
            anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
            logger.info("Successfully calculated interaction ANOVA table.")
            return anova_table
        except Exception as e:
            logger.error(f"Error calculating interaction ANOVA: {e}")
            return None

    def execute_three_pillar_anova(self, metric: str = "recall_at_5") -> Optional[pd.DataFrame]:
        """Fits an OLS model for the 3 macro pillars: Chunking, Retrieval, and RAG Complexity Profile."""
        if metric not in self.df.columns:
            logger.error(f"Metric {metric} not found in dataset.")
            return None

        # Prepare RAG complexity profile if not exists
        if 'rag_complexity_profile' not in self.df.columns:
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
            self.df['rag_complexity_profile'] = self.df.apply(map_complexity, axis=1)

        # Check if we have variance in the metric
        if self.df[metric].nunique() <= 1:
            logger.warning(f"No variance in {metric}. Skipping 3-pillar ANOVA.")
            return None

        factors = ["chunking", "retrieval", "rag_complexity_profile"]
        valid_factors = [f for f in factors if self.df[f].nunique() > 1]

        if not valid_factors:
            logger.warning("No valid factors for 3-pillar ANOVA.")
            return None

        formula = f"{metric} ~ " + " + ".join([f"C({f})" for f in valid_factors])
        logger.info(f"Fitting 3-Pillar ANOVA with formula: {formula}")

        try:
            model = ols(formula, data=self.df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
            logger.info("Successfully calculated 3-Pillar ANOVA table.")
            return anova_table
        except Exception as e:
            logger.error(f"Error calculating 3-pillar ANOVA: {e}")
            return None

    def generate_pillar_analysis_report(self, output_report_path: str = "reports/pillar_analysis_summary.md", metric: str = "recall_at_5") -> None:
        """Generates aggregate tables and a Markdown summary report comparing the 3 pillars."""
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

        if 'rag_complexity_profile' not in self.df.columns:
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
            self.df['rag_complexity_profile'] = self.df.apply(map_complexity, axis=1)

        # 1. Aggregate calculations
        chunking_agg = self.df.groupby("chunking")[metric].agg(["mean", "std", "count"]).sort_values(by="mean", ascending=False)
        retrieval_agg = self.df.groupby("retrieval")[metric].agg(["mean", "std", "count"]).sort_values(by="mean", ascending=False)
        rag_comp_agg = self.df.groupby("rag_complexity_profile")[metric].agg(["mean", "std", "count"]).sort_values(by="mean", ascending=False)

        # Individual sub-techniques under Pillar 3
        pre_agg = self.df.groupby("pre_retrieval")[metric].agg(["mean", "std", "count"]).sort_values(by="mean", ascending=False)
        idx_agg = self.df.groupby("index_structure")[metric].agg(["mean", "std", "count"]).sort_values(by="mean", ascending=False)
        post_agg = self.df.groupby("post_retrieval")[metric].agg(["mean", "std", "count"]).sort_values(by="mean", ascending=False)

        # Run 3-pillar ANOVA
        three_pillar_anova = self.execute_three_pillar_anova(metric=metric)

        # Write markdown report
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write("# Pillar-level RAG Benchmarking Analysis Report\n\n")
            f.write(f"This report evaluates the RAG configurations structured across the **3 mental model pillars** (Chunking, Retrieval, and RAG Techniques/Complexity) on the metric `{metric}`.\n\n")

            if three_pillar_anova is not None:
                f.write("## 1. 3-Pillar ANOVA & Effect Sizes (Variance Explained)\n\n")
                f.write("The table below shows the statistical significance and the percentage of variance explained ($\\eta^2$) by each of the 3 main pillars:\n\n")
                f.write("| Factor (Pillar) | Sum of Squares (SS) | Deg. of Freedom (DF) | F-Statistic | p-value | Effect Size ($\\eta^2$) |\n")
                f.write("|---|---|---|---|---|---|\n")
                for index, row in three_pillar_anova.iterrows():
                    p_val = f"{row['PR(>F)']:.4e}" if not pd.isna(row['PR(>F)']) else "N/A"
                    f_stat = f"{row['F']:.4f}" if not pd.isna(row['F']) else "N/A"
                    f.write(f"| **{index}** | {row['sum_sq']:.4f} | {row['df']:.0f} | {f_stat} | {p_val} | **{row['eta_sq']*100:.2f}%** |\n")
                f.write("\n")

            f.write("## 2. Pillar 1: Chunking Strategy Performance\n\n")
            f.write("| Chunking Strategy | Mean Recall@5 | Std Dev | Sample Size |\n")
            f.write("|---|---|---|---|\n")
            for index, row in chunking_agg.iterrows():
                f.write(f"| `{index}` | {row['mean']:.4f} | {row['std']:.4f} | {row['count']:.0f} |\n")
            f.write("\n")

            f.write("## 3. Pillar 2: Retrieval Strategy Performance\n\n")
            f.write("| Retrieval Strategy | Mean Recall@5 | Std Dev | Sample Size |\n")
            f.write("|---|---|---|---|\n")
            for index, row in retrieval_agg.iterrows():
                f.write(f"| `{index}` | {row['mean']:.4f} | {row['std']:.4f} | {row['count']:.0f} |\n")
            f.write("\n")

            f.write("## 4. Pillar 3: RAG Techniques / Complexity Performance\n\n")
            f.write("This groups configurations based on composite complexity profiles (Naive RAG vs. Advanced RAG with 1, 2, or 3 active techniques).\n\n")
            f.write("| RAG Complexity Profile | Mean Recall@5 | Std Dev | Sample Size |\n")
            f.write("|---|---|---|---|\n")
            for index, row in rag_comp_agg.iterrows():
                f.write(f"| **{index}** | {row['mean']:.4f} | {row['std']:.4f} | {row['count']:.0f} |\n")
            f.write("\n")

            f.write("### Sub-Technique Performance Breakdown (Pillar 3 Components)\n\n")
            
            f.write("#### A. Pre-Retrieval Methods\n")
            f.write("| Method | Mean Recall@5 | Std Dev | Sample Size |\n")
            f.write("|---|---|---|---|\n")
            for index, row in pre_agg.iterrows():
                f.write(f"| `{index}` | {row['mean']:.4f} | {row['std']:.4f} | {row['count']:.0f} |\n")
            f.write("\n")

            f.write("#### B. Index Structures\n")
            f.write("| Structure | Mean Recall@5 | Std Dev | Sample Size |\n")
            f.write("|---|---|---|---|\n")
            for index, row in idx_agg.iterrows():
                f.write(f"| `{index}` | {row['mean']:.4f} | {row['std']:.4f} | {row['count']:.0f} |\n")
            f.write("\n")

            f.write("#### C. Post-Retrieval Refinement\n")
            f.write("| Refinement | Mean Recall@5 | Std Dev | Sample Size |\n")
            f.write("|---|---|---|---|\n")
            for index, row in post_agg.iterrows():
                f.write(f"| `{index}` | {row['mean']:.4f} | {row['std']:.4f} | {row['count']:.0f} |\n")
            f.write("\n")

        logger.info(f"Pillar analysis report generated at {output_report_path}.")

    def generate_all_pairwise_plots(self, metric: str = "recall_at_5") -> None:
        """Generates interaction plots for all 10 pairwise combinations of the 5 axes."""
        import itertools
        factors = ["chunking", "pre_retrieval", "retrieval", "index_structure", "post_retrieval"]
        valid_factors = [f for f in factors if f in self.df.columns and self.df[f].nunique() > 1]

        for a, b in itertools.combinations(valid_factors, 2):
            self.generate_interaction_plots(a, b, metric)

    def generate_interaction_plots(self, factor_x: str, factor_y: str, metric: str) -> None:
        """Plots line charts of interaction variations to highlight synergistic pairings."""
        if factor_x not in self.df.columns or factor_y not in self.df.columns or metric not in self.df.columns:
            logger.error(f"One or more columns not found in dataset: {factor_x}, {factor_y}, {metric}")
            return
            
        # Create figures directory if it doesn't exist
        os.makedirs("figures", exist_ok=True)
        
        # Ensure we have at least 2 categories in each factor to plot interactions
        if self.df[factor_x].nunique() < 2 or self.df[factor_y].nunique() < 2:
            logger.info(f"Skipping interaction plot for {factor_x} and {factor_y}: Insufficient levels.")
            return
            
        try:
            plt.figure(figsize=(10, 6))
            
            # Sort the categories to make the plot clean
            sorted_df = self.df.sort_values(by=[factor_x, factor_y])
            
            num_traces = sorted_df[factor_y].nunique()
            colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'][:num_traces]
            markers_list = ['o', 's', '^', 'D', 'x', '*', 'v', '<'][:num_traces]
            
            fig = interaction_plot(
                x=sorted_df[factor_x],
                trace=sorted_df[factor_y],
                response=sorted_df[metric],
                colors=colors_list,
                markers=markers_list,
                ms=8,
                ax=plt.gca()
            )
            
            plt.title(f"Interaction Plot of {factor_x} & {factor_y} on {metric}", fontsize=14, fontweight='bold')
            plt.xlabel(factor_x.replace('_', ' ').title(), fontsize=12)
            plt.ylabel(f"Mean {metric.replace('_', ' ').title()}", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(title=factor_y.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            filename = f"figures/interaction_{factor_x}_vs_{factor_y}_on_{metric}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved interaction plot to {filename}.")
        except Exception as e:
            logger.error(f"Failed to generate interaction plot for {factor_x} vs {factor_y}: {e}")
            plt.close()
