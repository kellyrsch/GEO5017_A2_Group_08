from matplotlib import pyplot as plt
from statistics import mean
from sklearn.metrics import accuracy_score

from rf import get_rf_model, rf_tuning
from svm import get_svm_model, svm_tuning
from data import get_data, get_data_for_sklearn


def get_model_hyperparameters(features: list[tuple[callable, str]], train_samples: list[tuple[object, str]], test_samples: list[tuple[object, str]]):
    x_train, y_train, x_test, y_test = get_data_for_sklearn(train_samples, test_samples, features)
    svm_model, _ = svm_tuning(x_train, y_train, x_test, y_test)
    rf_model, _ = rf_tuning(x_train, y_train, x_test, y_test)
    best_svm_params = {
        'kernel': svm_model.kernel,
        'C': svm_model.C,
        'gamma': svm_model._gamma,
        'degree': svm_model.degree
    }
    best_rf_params = {
        'n_estimators': rf_model.n_estimators,
        'max_depth': rf_model.max_depth,
        'min_samples_split': rf_model.min_samples_split,
        'min_samples_leaf': rf_model.min_samples_leaf
    }
    return best_svm_params, best_rf_params

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

            svm_model = get_svm_model(x_train, y_train, kernel=svm_params['kernel'], C=svm_params['C'], gamma=svm_params['gamma'], degree=svm_params['degree'])
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
