
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from statistics import mean

from rf import rf_tuning, get_rf_model
from svm import svm_tuning, get_svm_model
from data import get_data, get_data_for_sklearn
from feature_selection import plot_feature_distribution, select_features_based_on_J_score
from features.number_of_points import calculate_number_of_points
from features.convex_hull import compute_convex_hull_volume
from features.convex_hull import point_density_in_convex_hull
from features.bounding_volumes import get_axis_aligned_bbox
from features.bounding_volumes import get_oriented_bbox
from features.bounding_volumes import get_height_of_aa_bbox
from features.height_width_ratio import height_width_ratio
from features.footprint_area import footprint_area
from features.lps_features import get_linearity, get_planarity, get_scattering

features = [
    #(calculate_number_of_points, "Number of Points"),
    (compute_convex_hull_volume, "Volume of Convex Hull"),
    (point_density_in_convex_hull, "Point density in convex hull"),
    (get_axis_aligned_bbox, "Volume of axis aligned bbox"),
    (get_oriented_bbox, "Volume of oriented bbox"),
    (get_height_of_aa_bbox, "Extent in y-dimension (height) of axis aligned bbox"),
    (height_width_ratio, "Height-to-width ratio"),
    (footprint_area, "Top-down footprint area"),
    (get_linearity, "Linearity"),
    (get_planarity, "Planarity"),
    (get_scattering, "Scattering")
]

train_samples, test_samples = get_data(test_size=0.4)

#plot_feature_distribution(point_clouds_with_labels, features)
features_to_use = select_features_based_on_J_score(train_samples, features, desired_feature_count=4)

def get_model_hyperparameters(features: list[tuple[callable, str]]):
    x_train, y_train, x_test, y_test = get_data_for_sklearn(train_samples, test_samples, features)
    svm_model, _ = svm_tuning(x_train, y_train, x_test, y_test)
    rf_model, _ = rf_tuning(x_train, y_train, x_test, y_test)
    best_svm_params = {
        'kernel': svm_model.kernel,
        'C': svm_model.C
    }
    best_rf_params = {
        'n_estimators': rf_model.n_estimators,
        'max_depth': rf_model.max_depth
    }
    return best_svm_params, best_rf_params

svm_params, rf_params = get_model_hyperparameters(features_to_use)

def create_learning_curves(
    features: list[tuple[callable, str]],
    svm_params: dict,
    rf_params: dict,
    num_random_samples: int = 5,
    base_seed: int = 42,
):
    test_sizes_to_plot = [i / 100 for i in range(10, 100, 5)]
    results = []

    for test_size in test_sizes_to_plot:
        print(f"\nEvaluating models with test size = {test_size} ({num_random_samples} random splits)...")

        svm_scores = []
        rf_scores = []
        train_size_for_plot = None

        for seed in range(base_seed, base_seed + num_random_samples):
            # we take random samples to even out "getting lucky" by choosing the best samples for training. By averaging over multiple random splits, we get a more reliable estimate of model performance for each training size.
            train_samples, test_samples = get_data(test_size=test_size, seed=seed)

            if train_size_for_plot is None:
                train_size_for_plot = len(train_samples)

            x_train, y_train, x_test, y_test = get_data_for_sklearn(train_samples, test_samples, features)

            svm_model = get_svm_model(x_train, y_train, kernel=svm_params['kernel'], C=svm_params['C'])
            rf_model = get_rf_model(x_train, y_train, n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'])

            svm_scores.append(accuracy_score(y_test, svm_model.predict(x_test)))
            rf_scores.append(accuracy_score(y_test, rf_model.predict(x_test)))

        svm_mean_accuracy = mean(svm_scores)
        rf_mean_accuracy = mean(rf_scores)

        print(f"Train samples: {train_size_for_plot}")
        print(f"Average SVM Accuracy: {svm_mean_accuracy * 100:.2f}%")
        print(f"Average RF Accuracy: {rf_mean_accuracy * 100:.2f}%")

        results.append((train_size_for_plot, svm_mean_accuracy, rf_mean_accuracy))

    # Sort by absolute train size so curves go left-to-right with more training data.
    results.sort(key=lambda item: item[0])
    train_sizes_to_plot, svm_accuracies, rf_accuracies = map(list, zip(*results))

    # plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_to_plot, svm_accuracies, marker='o', label='SVM Accuracy')
    plt.plot(train_sizes_to_plot, rf_accuracies, marker='o', label='Random Forest Accuracy')
    plt.title('Learning Curves for SVM and Random Forest')
    plt.xlabel('Training sample size (absolute)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("plots/learning_curves.png")
    plt.show()

create_learning_curves(features_to_use, svm_params, rf_params, num_random_samples=20)

final_test_size = 0.4
x_train, y_train, x_test, y_test = get_data_for_sklearn(*get_data(test_size=final_test_size), features_to_use)
final_rf_model = get_rf_model(x_train, y_train, n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'])
final_svm_model = get_svm_model(x_train, y_train, kernel=svm_params['kernel'], C=svm_params['C'])

def plot_confusion_matrix(model, title: str):
    predictions = model.predict(x_test)
    class_names = sorted(set(y_test))
    cm_svm = confusion_matrix(y_test, predictions, labels=class_names)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=class_names)
    disp_svm.plot(cmap='Blues', xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()

plot_confusion_matrix(final_svm_model, "SVM Confusion Matrix")
plot_confusion_matrix(final_rf_model, "Random Forest Confusion Matrix")