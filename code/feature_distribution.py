from itertools import groupby
import matplotlib.pyplot as plt
import numpy as np
import os

def compute_feature_score(feature_values_per_label):
    all_values = []
    for values in feature_values_per_label.values():
        all_values.extend(values)

    all_values = np.array(all_values)
    global_mean = np.mean(all_values)
    N = len(all_values)

    SW = 0
    SB = 0

    for label, values in feature_values_per_label.items():
        values = np.array(values)
        Nk = len(values)

        class_mean = np.mean(values)
        class_var = np.var(values)

        SW += (Nk / N) * class_var
        SB += (Nk / N) * (class_mean - global_mean) ** 2

    J = SB / (SW + 1e-8)
    return J, SB, SW

def compute_feature_score(feature_values_per_label):
    all_values = []
    for values in feature_values_per_label.values():
        all_values.extend(values)

    all_values = np.array(all_values)
    global_mean = np.mean(all_values)
    N = len(all_values)

    SW = 0
    SB = 0

    for label, values in feature_values_per_label.items():
        values = np.array(values)
        Nk = len(values)

        class_mean = np.mean(values)
        class_var = np.var(values)

        SW += (Nk / N) * class_var
        SB += (Nk / N) * (class_mean - global_mean) ** 2

    J = SB / (SW + 1e-8)
    return J, SB, SW


def get_feature_stats_over_labels(features_with_labels, list_of_feature_classifiers_with_names):
    features_with_labels = sorted(features_with_labels, key=lambda x: x[1])
    features_grouped_by_label = {
        label: [feature for feature, _ in group]
        for label, group in groupby(features_with_labels, lambda x: x[1])
    }

    feature_stats = {}
    feature_scores = {}

    os.makedirs("plots", exist_ok=True)

    for label, raw_features in features_grouped_by_label.items():
        feature_statistics = {}
        for feature_classifier, classifier_name in list_of_feature_classifiers_with_names:
            feature_values = [feature_classifier(feature) for feature in raw_features]
            mean = np.mean(feature_values)
            variance = np.var(feature_values)

            feature_statistics[classifier_name] = {
                "mean": mean,
                "variance": variance,
                "values": feature_values
            }

        feature_stats[label] = feature_statistics

    labels = list(feature_stats.keys())
    classifier_names = list(feature_stats[labels[0]].keys())

    for classifier_name in classifier_names:
        means = [feature_stats[label][classifier_name]["mean"] for label in labels]

        #collect raw values per label for J score
        feature_values_per_label = {
            label: feature_stats[label][classifier_name]["values"]
            for label in labels
        }

        J, SB, SW = compute_feature_score(feature_values_per_label)
        feature_scores[classifier_name] = {
            "J": J,
            "SB": SB,
            "SW": SW
        }

        plt.figure()
        plt.bar(labels, means, label=classifier_name)
        plt.title(f"Average Feature Values by Label\nJ = {J:.4f}")
        plt.ylabel("Mean Value")
        plt.legend()

        safe_name = classifier_name.replace(" ", "_").lower()
        plt.savefig(f"plots/{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    #print ranking
    print("\nFeature selection ranking (higher J is better):")
    ranked = sorted(feature_scores.items(), key=lambda x: x[1]["J"], reverse=True)

    for i, (name, scores) in enumerate(ranked, start=1):
        print(f"{i}. {name}: J = {scores['J']:.6f}, SB = {scores['SB']:.6f}, SW = {scores['SW']:.6f}")

    print("\nTop 4 features:")
    for i, (name, scores) in enumerate(ranked[:4], start=1):
        print(f"{i}. {name}")

    return feature_stats
