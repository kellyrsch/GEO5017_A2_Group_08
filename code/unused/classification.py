import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from features.height_width_ratio import height_width_ratio
from features.footprint_area import footprint_area
from features.lps_features import lps_features


def get_linearity(points):
    return lps_features(points)[0]


def get_scattering(points):
    return lps_features(points)[2]


def run_classification(point_clouds_with_labels):

    #build dataset
    X = []
    y = []

    for pcd, label in point_clouds_with_labels:
        features = [
            height_width_ratio(pcd),
            footprint_area(pcd),
            get_linearity(pcd),
            get_scattering(pcd),
        ]
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    #encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    #split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    #scaling (SVM only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #SVM
    print("\nSVM")

    svm_params = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "degree": [2, 3]
    }

    svm = SVC()

    svm_grid = GridSearchCV(
        svm,
        svm_params,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

    svm_grid.fit(X_train_scaled, y_train)

    print("Best SVM params:", svm_grid.best_params_)

    best_svm = svm_grid.best_estimator_

    y_pred = best_svm.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    #RF
    print("\nRandom Forest")

    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)

    rf_grid = GridSearchCV(
        rf,
        rf_params,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    print("Best RF params:", rf_grid.best_params_)

    best_rf = rf_grid.best_estimator_

    y_pred = best_rf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))