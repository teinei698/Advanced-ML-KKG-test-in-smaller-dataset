# This file trains  models with the intersection features

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_and_test_model(features_dataframe, feature_columns, model_name, model):
    X = features_dataframe[feature_columns]
    y = features_dataframe["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probability_scores = model.predict_proba(X_test)[:, 1]
    else:
        probability_scores = predictions

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probability_scores)

    #Out put with the format
    return {
        "model": model_name,
        "features": ", ".join(feature_columns),
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc
    }


def main():
    data_folder = "data"
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)

    features_path = os.path.join(data_folder, "features.csv")
    output_path = os.path.join(results_folder, "model_results.csv")

    print("Loading features...")
    features_dataframe = pd.read_csv(features_path)

    # Some feature groups for ablation study
    head_tail_features = [
        "one_hop_head_tail",
        "two_hop_head_tail",
        "three_hop_head_tail"
    ]

    triple_features = [
        "one_hop_triple_intersection",
        "two_hop_triple_intersection",
        "three_hop_triple_intersection"
    ]

    two_hop_only_features = [
        "two_hop_head_tail",
        "two_hop_triple_intersection"
    ]

    all_useful_features = [
        "one_hop_head_tail",
        "two_hop_head_tail",
        "three_hop_head_tail",
        "one_hop_triple_intersection",
        "two_hop_triple_intersection",
        "three_hop_triple_intersection",
        "relation_degree_1hop"
    ]

    feature_groups = [
        ("head_tail_only", head_tail_features),
        ("triple_intersection_only", triple_features),
        ("two_hop_only", two_hop_only_features),
        ("all_features", all_useful_features)
    ]

    results = []

    for group_name, columns in feature_groups:
        print("Training with feature group:", group_name)

        logistic_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )

        mlp_model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=500,
                random_state=42
            )
        )

        result_logistic = train_and_test_model(
            features_dataframe,
            columns,
            "Logistic Regression - " + group_name,
            logistic_model
        )

        result_mlp = train_and_test_model(
            features_dataframe,
            columns,
            "MLP - " + group_name,
            mlp_model
        )

        results.append(result_logistic)
        results.append(result_mlp)

    results_dataframe = pd.DataFrame(results)
    results_dataframe.to_csv(output_path, index=False)

    print("Suceeded, continue to step6, Saved results:", output_path)
    print(results_dataframe)


if __name__ == "__main__":
    main()