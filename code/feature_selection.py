from collections import defaultdict
from itertools import groupby
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
    features_with_labels = sorted(features_with_labels, key=lambda x: x[1])
    features_grouped_by_label = {
        label: [feature for feature, _ in group]
        for label, group in groupby(features_with_labels, lambda x: x[1])
    }

    feature_stats = {}

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

        # #collect raw values per label for J score
        # feature_values_per_label = {
        #     label: {
        #         classifier_name: feature_stats[label][classifier_name]["values"]
        #     }
        #     for label in labels
        # }

        # J, SB, SW = evaluate_feature_set(feature_values_per_label)
        # # convert J, SB, SW to scalars for easier comparison and plotting
        # J = float(J[0, 0])
        # SB = float(SB[0, 0])
        # SW = float(SW)

        # feature_scores[classifier_name] = {
        #     "J": J,
        #     "SB": SB,
        #     "SW": SW
        # }

        plt.figure()
        plt.bar(labels, means, label=classifier_name)
        plt.title(f"Average Feature Values by Label\nJ = {J:.4f}")
        plt.ylabel("Mean Value")
        plt.legend()

        safe_name = classifier_name.replace(" ", "_").lower()
        plt.savefig(f"plots/{safe_name}.png", dpi=300, bbox_inches="tight")
        #plt.show()