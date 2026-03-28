from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def get_svm_model(x_train, y_train, kernel, C, gamma, degree):
    model = SVC(kernel=kernel, C=C, gamma=gamma or 'scale', degree=degree or 3, random_state=42)
    model.fit(x_train, y_train)
    return model

def svm_tuning(x_train, y_train, x_test, y_test):
    print("--- Starting Manual SVM Hyperparameter Tuning ---")
    
    kernels = ['linear', 'rbf', 'poly']
    C_values = [0.1, 1, 10]
    gamma_values = ['scale', 'auto', 0.01, 0.1]
    degree_values = [2, 3]
    

    best_accuracy = 0
    best_params = {}
    best_svm_model = None
    tuning_results = []
    
    for kernel in kernels:
        for C in C_values:
            for index_gamma, gamma in enumerate(gamma_values):
                if kernel == 'linear' and index_gamma > 0:
                    continue # gamma is not relevant to linear kernel
                for index_degree, degree in enumerate(degree_values):
                    if kernel != 'poly' and index_degree > 0:
                        continue # degree is only relevant to polynomial kernel
                    print(f"Testing SVM with kernel='{kernel}', C={C}, gamma={gamma if kernel != 'linear' else None}, degree={degree if kernel == 'poly' else None}...")
                    model = get_svm_model(x_train, y_train, kernel=kernel, C=C, gamma=gamma if kernel != 'linear' else None, degree=degree if kernel == 'poly' else None)
                    
                    predictions = model.predict(x_test)
                    accuracy = accuracy_score(y_test, predictions)
                    
                    print(f" -> Accuracy: {accuracy * 100:.2f}%")
                    tuning_results.append({
                        'kernel': kernel,
                        'C': C,
                        'gamma': gamma if kernel != 'linear' else None,
                        'degree': degree if kernel == 'poly' else None,
                        'accuracy': accuracy
                    })
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma if kernel != 'linear' else None, 'degree': degree if kernel == 'poly' else None}
                        best_svm_model = model
                
    print("\n=== Tuning Complete ===")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
    
    return best_svm_model, tuning_results