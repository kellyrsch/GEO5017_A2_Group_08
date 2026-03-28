from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

def z_score_standardisation(feature_values):
    feature_values = np.array(feature_values)
    mean = np.mean(feature_values)
    std_dev = np.std(feature_values)
    
    if std_dev == 0:
        return np.zeros_like(feature_values)
    
    return (feature_values - mean) / std_dev

def evaluate_feature_set(feature_set_values_per_label: dict[str, dict[str, list[float]]]) -> tuple[np.ndarray, np.ndarray, float]:
    # Convert nested dict (class label, feature name, feature values) to {class_label: 2D_Matrix (N_k, D)} where D is the number of features and N_k is the number of samples for that class.
    # Grab the feature names from the first class to ensure our matrix columns 
    # are always stacked in the exact same order for every class.
    first_class = list(feature_set_values_per_label.keys())[0]
    feature_names = list(feature_set_values_per_label[first_class].keys())
    
    matrix_dict = {}
    
    for class_label, features_dict in feature_set_values_per_label.items():
        columns = [features_dict[feat_name] for feat_name in feature_names]
        
        # creates the (N_k, D) matrix for this class
        X_k = np.column_stack(columns) 
        matrix_dict[class_label] = X_k

    X_all = np.vstack(list(matrix_dict.values()))
    N, D = X_all.shape
    mu = np.mean(X_all, axis=0).reshape(D, 1)
    
    S_W = np.zeros((D, D))
    S_B = np.zeros((D, D))
    
    # Calculate Scatter Matrices for each class
    for X_k in matrix_dict.values():
        N_k = X_k.shape[0] # class sample size
        if N_k == 0:
            continue
            
        mu_k = np.mean(X_k, axis=0).reshape(D, 1) # mean of each feature for this class
        
        Sigma_k = np.cov(X_k, rowvar=False, bias=True) # covariance matrix of features for this class (divide by N_k, not N_k-1, since we are treating this as the population covariance for this class)
        Sigma_k = np.atleast_2d(Sigma_k) # if we are using this function on only one feature, we need to ensure Sigma_k is 2D for the matrix operations below
        
        S_W += (N_k / N) * Sigma_k  # (sum operator over k)
        
        mean_diff = mu_k - mu
        S_B += (N_k / N) * np.dot(mean_diff, mean_diff.T)
        
    trace_S_W = np.trace(S_W)
    trace_S_B = np.trace(S_B)
    
    epsilon = 1e-8 # avoid division by zero
    J = trace_S_B / (trace_S_W + epsilon)
    
    return S_W, S_B, J

def get_feature_subset(full_dict: dict[str, dict[str, list[float]]], selected_feature_names: list[str]) -> dict[str, dict[str, list[float]]]:
    subset_dict = {}
    for class_label, features_dict in full_dict.items():
        # Create a new dict for this class containing only the requested features
        subset_dict[class_label] = {
            feat: features_dict[feat] for feat in selected_feature_names
        }
    return subset_dict

def backward_search(feature_set_values_per_label: dict[str, dict[str, list[float]]], num_features_to_select: int):
    # Start with ALL available features in the basket
    first_class = list(feature_set_values_per_label.keys())[0]
    current_features = list(feature_set_values_per_label[first_class].keys())
    
    # Validation check
    if num_features_to_select >= len(current_features):
        print("Target number of features is equal to or greater than available features. Returning all.")
        return current_features, []
        
    j_score_history = []
    
    print(f"Starting SBS. Initial features: {len(current_features)}")
    
    # Calculate the J-score of the complete dataset before dropping anything
    _, _, initial_J = evaluate_feature_set(feature_set_values_per_label)
    print(f"Step 0: All features included (Initial J-score: {initial_J:.4f})")
    j_score_history.append(initial_J)
    
    # Loop until we are left with the target number of features
    while len(current_features) > num_features_to_select:
        best_j_this_step = -1.0
        feature_to_drop_this_step = None
        
        # Test removing each feature one by one
        for feature in current_features:
            features_to_test = [f for f in current_features if f != feature]
            test_dict = get_feature_subset(feature_set_values_per_label, features_to_test)
            
            # Evaluate the remaining combination
            _, _, J = evaluate_feature_set(test_dict)
            
            if J > best_j_this_step:
                best_j_this_step = J
                feature_to_drop_this_step = feature
                
        # drop the worst feature for this step
        current_features.remove(feature_to_drop_this_step)
        j_score_history.append(best_j_this_step)
        
        print(f"Dropped: '{feature_to_drop_this_step}' | Remaining: {len(current_features)} | New J-score: {best_j_this_step:.4f}")
        
    return current_features, j_score_history

def select_features_based_on_J_score(samples_with_labels: list[tuple[object, str]], feature_classifiers: list[tuple[callable, str]], desired_feature_count: int) -> list[tuple[callable, str]]:
    # Step 1: Calculate feature values for each point cloud and standardise them
    all_samples = [x[0] for x in samples_with_labels] # we can do this because we are not altering the order
    standardised_feature_values = {}
    for classifier, feature_name in feature_classifiers:
        feature_values = [classifier(s) for s in all_samples]
        standardised_values = z_score_standardisation(feature_values)
        standardised_feature_values[feature_name] = zip(standardised_values, [x[1] for x in samples_with_labels])

    # convert to {class_label: {feature_name: [feature_values]}} for J score evaluation
    feature_values_per_label = defaultdict(lambda: defaultdict(list))
    for feature_name, values_with_labels in standardised_feature_values.items():
        for value, label in values_with_labels:
            feature_values_per_label[label][feature_name].append(value)

    selected_feature_names, j_score_history = backward_search(feature_values_per_label, desired_feature_count)
    return [(classifier, name) for classifier, name in feature_classifiers if name in selected_feature_names]


def plot_feature_distribution(features_with_labels, list_of_feature_classifiers_with_names):
    if not features_with_labels or not list_of_feature_classifiers_with_names:
        return

    os.makedirs("plots", exist_ok=True)

    samples = [sample for sample, _ in features_with_labels]
    labels = [label for _, label in features_with_labels]
    unique_labels = sorted(set(labels))

    classifier_names = [name for _, name in list_of_feature_classifiers_with_names]
    normalised_values_by_classifier = {}

    for feature_classifier, classifier_name in list_of_feature_classifiers_with_names:
        raw_values = [feature_classifier(sample) for sample in samples]
        normalised_values = z_score_standardisation(raw_values)
        normalised_values_by_classifier[classifier_name] = normalised_values

    class_feature_means = {
        label: [] for label in unique_labels
    }
    for label in unique_labels:
        indices = [i for i, current_label in enumerate(labels) if current_label == label]
        for classifier_name in classifier_names:
            values = normalised_values_by_classifier[classifier_name]
            label_values = [values[i] for i in indices]
            class_feature_means[label].append(np.mean(label_values))

    split_index = (len(classifier_names) + 1) // 2
    feature_groups = [
        classifier_names[:split_index],
        classifier_names[split_index:]
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    cmap = plt.get_cmap("tab20")

    for chart_index, feature_group in enumerate(feature_groups):
        ax = axes[chart_index]
        if not feature_group:
            ax.set_visible(False)
            continue

        start_index = 0 if chart_index == 0 else split_index
        end_index = start_index + len(feature_group)

        x = np.arange(len(feature_group))
        num_classes = len(unique_labels)
        bar_width = 0.8 / max(num_classes, 1)

        for class_index, label in enumerate(unique_labels):
            offsets = x - 0.4 + (class_index + 0.5) * bar_width
            ax.bar(
                offsets,
                class_feature_means[label][start_index:end_index],
                width=bar_width,
                color=cmap(class_index % 20),
                label=label
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x, feature_group, rotation=45, ha="right")
        ax.set_ylabel("Normalised Value (Z-score)")
        ax.set_title(f"Normalised Feature Means by Class (Part {chart_index + 1})")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, title="Class", ncol=min(len(unique_labels), 4), loc="upper center")

    fig.supxlabel("Feature")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plots/normalised_feature_values_coloured.png", dpi=300, bbox_inches="tight")
    plt.close()