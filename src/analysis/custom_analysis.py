import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

def run_custom_analysis(csv_path: str = "reports/stage1_screening_logs.csv", output_md_path: str = "reports/custom_analysis_report.md"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Please run the screening sweep first.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} screening log rows from {csv_path}.")

    # 1. Top 10 Configurations
    # Calculate a composite score to rank pipelines
    df["composite_score"] = df["recall_at_5"] * 0.7 + df["mrr_at_5"] * 0.3
    
    # Aggregate by pipeline_id (mean across questions)
    pipeline_agg = df.groupby("pipeline_id").agg({
        "chunking": "first",
        "pre_retrieval": "first",
        "retrieval": "first",
        "index_structure": "first",
        "post_retrieval": "first",
        "hit_rate_at_5": "mean",
        "recall_at_5": "mean",
        "mrr_at_5": "mean",
        "ndcg_at_5": "mean",
        "composite_score": "mean"
    }).reset_index()

    top_10 = pipeline_agg.sort_values(by="composite_score", ascending=False).head(10)

    # 2. Main Effects (Axis-level analysis)
    axes = ["chunking", "pre_retrieval", "retrieval", "index_structure", "post_retrieval"]
    axis_performance = {}
    for axis in axes:
        axis_performance[axis] = df.groupby(axis).agg({
            "recall_at_5": ["mean", "std"],
            "mrr_at_5": ["mean", "std"],
            "ndcg_at_5": ["mean"]
        })
        # Flatten columns
        axis_performance[axis].columns = [f"{col[0]}_{col[1]}" for col in axis_performance[axis].columns]
        axis_performance[axis] = axis_performance[axis].sort_values(by="recall_at_5_mean", ascending=False)

    # 3. Random Forest Parameter Sensitivity Analysis (Machine Learning alternative to ANOVA)
    # Check if there is variance in recall_at_5
    if df["recall_at_5"].nunique() > 1:
        X_cats = df[axes]
        y = df["recall_at_5"]

        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False)
        X_encoded = encoder.fit_transform(X_cats)
        encoded_feature_names = encoder.get_feature_names_out(axes)

        # Train a Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_encoded, y)

        # Calculate parameter importances by grouping the one-hot encoded importances back to their main factors
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": encoded_feature_names,
            "importance": importances
        })

        # Group by original axis name (prefix before the underscore)
        feature_importance_df["axis"] = feature_importance_df["feature"].apply(
            lambda x: next(axis for axis in axes if x.startswith(axis))
        )
        axis_importance = feature_importance_df.groupby("axis")["importance"].sum().sort_values(ascending=False).reset_index()
    else:
        axis_importance = pd.DataFrame({"axis": axes, "importance": [0.0] * len(axes)})

    # 4. Interaction Analysis: Retrieval vs Post-Retrieval
    retrieval_post_pivot = df.pivot_table(
        index="retrieval",
        columns="post_retrieval",
        values="recall_at_5",
        aggfunc="mean"
    )

    # Plot 1: Feature Importance
    plt.figure(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(axis_importance)))
    plt.bar(axis_importance["axis"], axis_importance["importance"], color=colors, edgecolor="black", alpha=0.8)
    plt.title("RAG Axis Importance on Recall@5 (Random Forest Sensitivity Analysis)", fontsize=12, fontweight="bold")
    plt.ylabel("Importance Score (Sum of Feature Importances)", fontsize=10)
    plt.xlabel("RAG Pipeline Parameter Axis", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    importance_plot_path = "figures/custom_feature_importance.png"
    plt.savefig(importance_plot_path, dpi=300)
    plt.close()

    # Plot 2: Heatmap of Retrieval vs Post-Retrieval Interaction using pure matplotlib imshow
    plt.figure(figsize=(8, 6))
    data = retrieval_post_pivot.values
    x_labels = retrieval_post_pivot.columns.tolist()
    y_labels = retrieval_post_pivot.index.tolist()

    im = plt.imshow(data, cmap="YlGnBu", aspect="auto")
    
    # Show all ticks and label them with the respective list entries
    plt.xticks(np.arange(len(x_labels)), labels=x_labels, rotation=15)
    plt.yticks(np.arange(len(y_labels)), labels=y_labels)

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = data[i, j]
            # Determine text color based on cell brightness
            text_color = "white" if val > np.max(data)*0.6 else "black"
            plt.text(j, i, f"{val:.3f}",
                     ha="center", va="center", color=text_color,
                     fontweight="bold")

    plt.title("Interaction Heatmap: Retrieval vs Post-Retrieval on Recall@5", fontsize=12, fontweight="bold")
    plt.ylabel("Retrieval Method", fontsize=10)
    plt.xlabel("Post-Retrieval Processing", fontsize=10)
    plt.colorbar(im, label="Mean Recall@5")
    plt.tight_layout()
    interaction_plot_path = "figures/custom_interaction_heatmap.png"
    plt.savefig(interaction_plot_path, dpi=300)
    plt.close()

    # 5. Write Markdown Report
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# Non-ANOVA Custom RAG Performance Analysis Report\n\n")
        f.write("This report provides a comprehensive alternative to classical ANOVA to evaluate how different RAG configuration choices impact retrieval performance metrics.\n\n")
        
        f.write("## 1. RAG Component Sensitivity Analysis (Machine Learning Feature Importance)\n\n")
        f.write("Rather than using linear F-tests (ANOVA), we trained a **Random Forest Regressor** on the one-hot encoded pipeline configurations to predict `recall_at_5`. By summing up the importances of the categories belonging to each axis, we obtain the total variance influence of each component:\n\n")
        f.write("| RAG Parameter Axis | Variance Explanation Importance |\n")
        f.write("|---|---|\n")
        for _, row in axis_importance.iterrows():
            f.write(f"| **{row['axis']}** | {row['importance']*100:.2f}% |\n")
        f.write("\n")
        f.write("*(Visualization saved to `figures/custom_feature_importance.png`)*\n\n")

        f.write("## 2. Top 10 RAG Pipeline Configurations\n\n")
        f.write("Ranked by a composite score (`Recall@5 * 0.7 + MRR@5 * 0.3`):\n\n")
        f.write("| Rank | Pipeline ID | Recall@5 | MRR@5 | NDCG@5 | Composite Score |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"| {i} | `{row['pipeline_id']}` | {row['recall_at_5']:.4f} | {row['mrr_at_5']:.4f} | {row['ndcg_at_5']:.4f} | **{row['composite_score']:.4f}** |\n")
        f.write("\n")

        f.write("## 3. Performance Breakdown by Individual Axes\n\n")
        for axis, perf in axis_performance.items():
            f.write(f"### Axis: {axis.replace('_', ' ').title()}\n\n")
            f.write(f"| Choice | Mean Recall@5 | Std Dev Recall | Mean MRR@5 | Mean NDCG@5 |\n")
            f.write("|---|---|---|---|---|\n")
            for idx, row in perf.iterrows():
                f.write(f"| `{idx}` | {row['recall_at_5_mean']:.4f} | {row['recall_at_5_std']:.4f} | {row['mrr_at_5_mean']:.4f} | {row['ndcg_at_5_mean']:.4f} |\n")
            f.write("\n")

        f.write("## 4. Synergy Analysis: Retrieval vs Post-Retrieval Interaction Heatmap\n\n")
        f.write("The table below shows the average `recall_at_5` for different combinations of Retrieval and Post-Retrieval techniques to identify potential synergy effects:\n\n")
        f.write("| Retrieval \\ Post-Retrieval | none | cross_encoder_rerank | contextual_compression |\n")
        f.write("|---|---|---|---|\n")
        for idx, row in retrieval_post_pivot.iterrows():
            f.write(f"| **{idx}** | {row.get('none', 0.0):.4f} | {row.get('cross_encoder_rerank', 0.0):.4f} | {row.get('contextual_compression', 0.0):.4f} |\n")
        f.write("\n")
        f.write("*(Visualization saved to `figures/custom_interaction_heatmap.png`)*\n\n")

    print(f"Custom analysis successfully completed. Report written to {output_md_path}.")

if __name__ == "__main__":
    run_custom_analysis()
