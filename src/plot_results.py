# This file makes figures for the report, actually more graph than we need,
# So i categorize them to different values

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data_folder = "data"
    results_folder = "results"
    figure_folder = os.path.join(results_folder, "figures")
    os.makedirs(figure_folder, exist_ok=True)

    features_path = os.path.join(data_folder, "features.csv")
    model_results_path = os.path.join(results_folder, "model_results.csv")

    print("Loading files...")
    features_dataframe = pd.read_csv(features_path)
    model_results_dataframe = pd.read_csv(model_results_path)

    positive_dataframe = features_dataframe[features_dataframe["label"] == 1]
    negative_dataframe = features_dataframe[features_dataframe["label"] == 0]

    feature_columns = [
        "one_hop_head_tail",
        "two_hop_head_tail",
        "three_hop_head_tail",
        "one_hop_triple_intersection",
        "two_hop_triple_intersection",
        "three_hop_triple_intersection"
    ]

    # Figure 1, High value: Feature mean comparison, which is the most direct evidence for the paper claim

    mean_rows = []

    for feature_name in feature_columns:
        mean_rows.append({
            "feature": feature_name,
            "valid_mean": positive_dataframe[feature_name].mean(),
            "invalid_mean": negative_dataframe[feature_name].mean()
        })

    mean_dataframe = pd.DataFrame(mean_rows)

    plt.figure(figsize=(11, 5))
    x_positions = range(len(mean_dataframe))
    bar_width = 0.35

    plt.bar(
        [x - bar_width / 2 for x in x_positions],
        mean_dataframe["valid_mean"],
        width=bar_width,
        label="Valid triples"
    )
    plt.bar(
        [x + bar_width / 2 for x in x_positions],
        mean_dataframe["invalid_mean"],
        width=bar_width,
        label="Invalid triples"
    )

    plt.title("Feature Mean Comparison: Valid vs Invalid Triples")
    plt.xlabel("Feature")
    plt.ylabel("Mean value")
    plt.xticks(x_positions, mean_dataframe["feature"], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig1_feature_mean_comparison.png"), dpi=200)
    plt.close()

    # Figure 2, High value： Model AUC comparison which is showing whether the features are actually useful for classification

    short_names = model_results_dataframe["model"].str.replace("Logistic Regression - ", "LR-", regex=False)
    short_names = short_names.str.replace("MLP - ", "MLP-", regex=False)

    plt.figure(figsize=(10, 5))
    plt.plot(short_names, model_results_dataframe["auc"], marker="o")
    plt.title("Model AUC Comparison")
    plt.xlabel("Model and feature group")
    plt.ylabel("AUC")
    plt.ylim(0.88, 1.0)
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig2_model_auc_comparison.png"), dpi=200)
    plt.close()

    # Figure 3, high value: Distribution of the strongest triple feature
    # Which is to make the separation visually clear

    plt.figure(figsize=(8, 5))
    plt.hist(
        positive_dataframe["two_hop_triple_intersection"],
        bins=40,
        alpha=0.6,
        label="Valid triples"
    )
    plt.hist(
        negative_dataframe["two_hop_triple_intersection"],
        bins=40,
        alpha=0.6,
        label="Invalid triples"
    )
    plt.title("Distribution of Two-hop Triple Intersection")
    plt.xlabel("two_hop_triple_intersection")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig3_two_hop_triple_distribution.png"), dpi=200)
    plt.close()

    # Figure 4, High value: k hop triple intersection comparison,which is to explain why 1-hop is weak and 2/3-hop are more useful

    valid_triple_means = [
        positive_dataframe["one_hop_triple_intersection"].mean(),
        positive_dataframe["two_hop_triple_intersection"].mean(),
        positive_dataframe["three_hop_triple_intersection"].mean()
    ]

    invalid_triple_means = [
        negative_dataframe["one_hop_triple_intersection"].mean(),
        negative_dataframe["two_hop_triple_intersection"].mean(),
        negative_dataframe["three_hop_triple_intersection"].mean()
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(["1-hop", "2-hop", "3-hop"], valid_triple_means, marker="o", label="Valid triples")
    plt.plot(["1-hop", "2-hop", "3-hop"], invalid_triple_means, marker="o", label="Invalid triples")
    plt.title("k-hop Triple Intersection Trend")
    plt.xlabel("Hop")
    plt.ylabel("Mean intersection value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig4_khop_triple_trend.png"), dpi=200)
    plt.close()

    # Figure 5,Middle value: Heal tail vs triple-intersection comparison
    # which compares entity relation with triple feature

    selected_box_features = [
        "two_hop_head_tail",
        "two_hop_triple_intersection",
        "three_hop_head_tail",
        "three_hop_triple_intersection"
    ]

    box_data = []
    box_labels = []

    for feature_name in selected_box_features:
        box_data.append(positive_dataframe[feature_name])
        box_labels.append("valid\n" + feature_name)

        box_data.append(negative_dataframe[feature_name])
        box_labels.append("invalid\n" + feature_name)

    plt.figure(figsize=(12, 5))
    plt.boxplot(box_data, labels=box_labels, showfliers=False)
    plt.title("Head-tail Features vs Triple-intersection Features")
    plt.xlabel("Feature type")
    plt.ylabel("Value")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig5_head_tail_vs_triple_boxplot.png"), dpi=200)
    plt.close()

    # Figure 6, Medium value: Accuracy and F1 comparison, which gives extra evaluation beside AUC

    x_positions = range(len(model_results_dataframe))

    plt.figure(figsize=(10, 5))
    plt.plot(short_names, model_results_dataframe["accuracy"], marker="o", label="Accuracy")
    plt.plot(short_names, model_results_dataframe["f1"], marker="o", label="F1")
    plt.title("Accuracy and F1 Comparison")
    plt.xlabel("Model and feature group")
    plt.ylabel("Score")
    plt.ylim(0.75, 1.0)
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig6_accuracy_f1_comparison.png"), dpi=200)
    plt.close()

    # Figure 7, Medium value: Scatter plot of two useful features
    # This shows if positive and negative triples are separated in 2D feature space

    sample_dataframe = features_dataframe.sample(
        min(2000, len(features_dataframe)),
        random_state=42
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        sample_dataframe[sample_dataframe["label"] == 0]["two_hop_head_tail"],
        sample_dataframe[sample_dataframe["label"] == 0]["two_hop_triple_intersection"],
        alpha=0.5,
        label="Invalid triples"
    )
    plt.scatter(
        sample_dataframe[sample_dataframe["label"] == 1]["two_hop_head_tail"],
        sample_dataframe[sample_dataframe["label"] == 1]["two_hop_triple_intersection"],
        alpha=0.5,
        label="Valid triples"
    )
    plt.title("Two-hop Feature Space")
    plt.xlabel("two_hop_head_tail")
    plt.ylabel("two_hop_triple_intersection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig7_two_hop_feature_scatter.png"), dpi=200)
    plt.close()

    # Figure 8, Lower but still useful value: Correlation heatmap
    # This is useful for discussion because some features may repeat similar information

    correlation_dataframe = features_dataframe[feature_columns].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_dataframe, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(feature_columns)), feature_columns, rotation=35, ha="right")
    plt.yticks(range(len(feature_columns)), feature_columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "fig8_feature_correlation_heatmap.png"), dpi=200)
    plt.close()

    print("All figures are saved in:", figure_folder)


if __name__ == "__main__":
    main()