
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from model_tuning import create_learning_curves, get_model_hyperparameters
from rf import  get_rf_model
from svm import get_svm_model
from data import get_data, get_data_for_sklearn
from feature_selection import plot_feature_distribution, select_features_based_on_J_score
from features import *

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
all_samples = train_samples + test_samples

plot_feature_distribution(all_samples, features)
features_to_use = select_features_based_on_J_score(train_samples, features, desired_feature_count=4)

svm_params, rf_params = get_model_hyperparameters(features_to_use, train_samples, test_samples)

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