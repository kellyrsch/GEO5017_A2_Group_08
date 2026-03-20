from itertools import groupby
import matplotlib.pyplot as plt
import numpy as np

def get_feature_stats_over_labels(features_with_labels, list_of_feature_classifiers_with_names: list[tuple[callable, str]]):
    features_with_labels = sorted(features_with_labels, key=lambda x: x[1])
    features_grouped_by_label = groupby(features_with_labels, lambda x: x[1])
    feature_stats = {}
    
    for label, group in features_grouped_by_label:
        group = list(group)
        raw_features = [feature for feature, _ in group]
        stats_for_label = {}
        
        for feature_classifier, classifier_name in list_of_feature_classifiers_with_names:
            feature_values = [feature_classifier(feature) for feature in raw_features]
            
            if feature_values:
                mean = sum(feature_values) / len(feature_values)
                variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
            else:
                mean = 0
                variance = 0

            stats_for_label[classifier_name] = {
                "mean": mean,
                "variance": variance
            }
        feature_stats[label] = stats_for_label

    # Plotting
    labels = list(feature_stats.keys())
    if not labels:
        return feature_stats
        
    classifier_names = [name for _, name in list_of_feature_classifiers_with_names]
    
    x = np.arange(len(labels))
    width = 0.8 / len(classifier_names) if classifier_names else 0.8
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for classifier_name in classifier_names:
        means = [feature_stats[label][classifier_name]['mean'] for label in labels]
        offset = width * multiplier
        rects = ax.bar(x + offset, means, width, label=classifier_name)
        multiplier += 1

    ax.set_ylabel('Mean Value')
    ax.set_title('Average Feature Values by Label')
    ax.set_xticks(x + width * (len(classifier_names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')

    plt.show()

    return feature_stats
