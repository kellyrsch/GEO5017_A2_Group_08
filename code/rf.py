from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def get_rf_model(x_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1 # n_jobs=-1 uses all available CPU cores to speed up training
    )
    model.fit(x_train, y_train)
    return model

def rf_tuning(x_train, y_train, x_test, y_test):
    print("--- Starting Manual Random Forest Hyperparameter Tuning ---")
    # n_estimators: Number of trees
    # max_depth: Maximum depth of the tree (None means nodes expand until all leaves are pure)
    n_estimators_list = [50, 100, 200]
    max_depth_list = [10, 20, None]
    min_samples_split_list = [2, 5]
    min_samples_leaf_list = [1, 2]
    
    best_accuracy = 0
    best_params = {}
    best_rf_model = None
    tuning_results = []
    
    for n_trees in n_estimators_list:
        for depth in max_depth_list:
            for min_split in min_samples_split_list:
                for min_leaf in min_samples_leaf_list:
                    depth_str = "Unlimited" if depth is None else str(depth)
                    print(f"Testing RF with n_estimators={n_trees}, max_depth={depth_str}, min_samples_split={min_split}, min_samples_leaf={min_leaf}...")

                    model = get_rf_model(x_train, y_train, n_estimators=n_trees, max_depth=depth, min_samples_split=min_split, min_samples_leaf=min_leaf)
                    
                    predictions = model.predict(x_test)
                    accuracy = accuracy_score(y_test, predictions)
                    
                    print(f" -> Accuracy: {accuracy * 100:.2f}%")
                    
                    tuning_results.append({
                        'n_estimators': n_trees,
                        'max_depth': depth,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf,
                        'accuracy': accuracy
                    })
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'n_estimators': n_trees, 'max_depth': depth, 'min_samples_split': min_split, 'min_samples_leaf': min_leaf}
                        best_rf_model = model
                
    print("\n=== RF Tuning Complete ===")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
    
    return best_rf_model, tuning_results