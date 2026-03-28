from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def get_svm_model(x_train, y_train, kernel='rbf', C=1.0):
    model = SVC(kernel=kernel, C=C, random_state=42)
    model.fit(x_train, y_train)
    return model

def svm_tuning(x_train, y_train, x_test, y_test):
    print("--- Starting Manual SVM Hyperparameter Tuning ---")
    
    kernels = ['linear', 'rbf']
    C_values = [0.1, 1, 10]
    
    best_accuracy = 0
    best_params = {}
    best_svm_model = None
    tuning_results = []
    
    for kernel in kernels:
        for C in C_values:
            print(f"Testing SVM with kernel='{kernel}', C={C}...")
            model = get_svm_model(x_train, y_train, kernel=kernel, C=C)
            
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            
            print(f" -> Accuracy: {accuracy * 100:.2f}%")
            tuning_results.append({
                'kernel': kernel,
                'C': C,
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'kernel': kernel, 'C': C}
                best_svm_model = model
                
    print("\n=== Tuning Complete ===")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
    
    return best_svm_model, tuning_results