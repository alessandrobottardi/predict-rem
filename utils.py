import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def split_data(X, Y, test_size=0.2, val_size=0.2, random_state=None):
    """Splits data into training, validation, and testing sets while preserving class distribution."""
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=test_size + val_size, random_state=random_state, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size / (test_size + val_size), random_state=random_state, stratify=Y_temp)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
def perform_grid_search(X_train, Y_train, X_val, Y_val, random_state=None):
    """Performs grid search to find the best C parameter using a separate validation set."""
    param_grid = {'C': np.logspace(-4, 4, 10)}
    best_C, best_score = None, -np.inf

    for C in param_grid['C']:
        model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='multinomial', class_weight='balanced')
        model.fit(X_train, Y_train)
        Y_val_pred = model.predict(X_val)
        score = accuracy_score(Y_val, Y_val_pred)
        
        if score > best_score:
            best_C, best_score = C, score

    return best_C, best_score
    
def train_and_evaluate(X_train, Y_train, X_test, Y_test, C):
    """Trains a logistic regression model and evaluates it using accuracy, F1-score, false positives, and false negatives."""
    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', multi_class='multinomial', class_weight='balanced')
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    test_f1 = f1_score(Y_test, Y_pred, average='weighted')  # Weighted F1 score
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    
    return test_accuracy, test_f1, conf_matrix

def run_experiment_with_labels(X, Y, i, randomize_labels=False):
    """Handles the process of splitting, training, and evaluating with a validation set."""
    if randomize_labels:
        np.random.seed(42 + i)
        Y = np.random.permutation(Y)  # Shuffle labels if specified
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale X before splitting
    
    # Split data
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X_scaled, Y, random_state=42 + i)
    
    # Perform grid search using validation set
    best_C, best_accuracy = perform_grid_search(X_train, Y_train, X_val, Y_val, random_state=42 + i)
    
    # Train on the combined training and validation set and evaluate on test set
    X_combined = np.vstack((X_train, X_val))
    Y_combined = np.hstack((Y_train, Y_val))
    test_accuracy, test_f1, conf_matrix = train_and_evaluate(X_combined, Y_combined, X_test, Y_test, best_C)
    
    return best_C, best_accuracy, test_accuracy, test_f1, conf_matrix
    
def run_experiment(X, Y, n_repeats=10):
    """Runs multiple iterations of logistic regression training and evaluation."""
    results = {
        'best_C': [], 'best_accuracy': [], 'test_accuracy': [], 'test_f1': [], 'conf_matrix': [],
        'best_C_random': [], 'best_accuracy_random': [], 'test_accuracy_random': [], 'test_f1_random': [], 'conf_matrix_random': []
    }
    
    for i in range(n_repeats):
        best_C, best_accuracy, test_accuracy, test_f1, conf_matrix = run_experiment_with_labels(X, Y, i, randomize_labels=False)
        best_C_random, best_accuracy_random, test_accuracy_random, test_f1_random, conf_matrix_random = run_experiment_with_labels(X, Y, i, randomize_labels=True)
        
        results['best_C'].append(best_C)
        results['best_accuracy'].append(best_accuracy)
        results['test_accuracy'].append(test_accuracy)
        results['test_f1'].append(test_f1)
        results['conf_matrix'].append(conf_matrix)
        
        results['best_C_random'].append(best_C_random)
        results['best_accuracy_random'].append(best_accuracy_random)
        results['test_accuracy_random'].append(test_accuracy_random)
        results['test_f1_random'].append(test_f1_random)
        results['conf_matrix_random'].append(conf_matrix_random)
    
    return results


def run_experiment_with_N_random_neurons(X, Y, n_repeats=10, N=None):
    """Run the experiment with a random subset of neurons of size N, repeated n_repeats times."""
    
    n_trials, n_neurons = X.shape
    
    if N is None:
        N = n_neurons  # If no subset size is given, use all neurons
    
    if N > n_neurons:
        raise ValueError(f"Subset size N ({N}) cannot be larger than the number of neurons ({n_neurons}).")
    
    # Initialize the results dictionary for accuracy and F1 scores
    best_accuracies = []
    test_accuracies = []
    test_f1_scores = []
    
    best_accuracies_random = []
    test_accuracies_random = []
    test_f1_scores_random = []
    
    for i in range(n_repeats):
        # Create random indices for selecting a subset of neurons
        random_indices = np.random.choice(n_neurons, size=N, replace=False)
        
        # Select the subset of neurons from X
        X_subset = X[:, random_indices]

        # Run the experiment with the subset of neurons and collect the results for normal labels
        best_C, best_accuracy, test_accuracy, test_f1, conf_matrix = run_experiment_with_labels(X_subset, Y, i, randomize_labels=False)
        best_accuracies.append(best_accuracy)
        test_accuracies.append(test_accuracy)
        test_f1_scores.append(test_f1)
        
        # Run the experiment with the subset of neurons and collect the results for randomized labels
        best_C_random, best_accuracy_random, test_accuracy_random, test_f1_random, conf_matrix_random = run_experiment_with_labels(X_subset, Y, i, randomize_labels=True)
        best_accuracies_random.append(best_accuracy_random)
        test_accuracies_random.append(test_accuracy_random)
        test_f1_scores_random.append(test_f1_random)
    
    # Calculate mean and standard deviation for each metric (normal and randomized)
    results = {
        'mean_best_accuracy': np.mean(best_accuracies),
        'std_best_accuracy': np.std(best_accuracies),
        'mean_test_accuracy': np.mean(test_accuracies),
        'std_test_accuracy': np.std(test_accuracies),
        'mean_test_f1': np.mean(test_f1_scores),
        'std_test_f1': np.std(test_f1_scores),
        'mean_best_accuracy_random': np.mean(best_accuracies_random),
        'std_best_accuracy_random': np.std(best_accuracies_random),
        'mean_test_accuracy_random': np.mean(test_accuracies_random),
        'std_test_accuracy_random': np.std(test_accuracies_random),
        'mean_test_f1_random': np.mean(test_f1_scores_random),
        'std_test_f1_random': np.std(test_f1_scores_random)
    }
    
    return results

def run_experiment_for_different_N_neurons(X, Y, N_array, n_repeats=10):
    """Run the experiment for multiple subset sizes given in N_array."""
    
    # Initialize arrays to hold mean and std for each metric
    best_accuracy = []
    test_accuracy = []
    test_f1 = []
    
    best_accuracy_random = []
    test_accuracy_random = []
    test_f1_random = []
    
    for N in N_array:
        print(f"Running experiment for subset size: {N}")
        
        # Run the experiment for each subset size
        results_for_N = run_experiment_with_N_random_neurons(X, Y, n_repeats=n_repeats, N=N)
        
        # Append the mean and std values for each metric
        best_accuracy.append([results_for_N['mean_best_accuracy'], results_for_N['std_best_accuracy']])
        test_accuracy.append([results_for_N['mean_test_accuracy'], results_for_N['std_test_accuracy']])
        test_f1.append([results_for_N['mean_test_f1'], results_for_N['std_test_f1']])
        
        best_accuracy_random.append([results_for_N['mean_best_accuracy_random'], results_for_N['std_best_accuracy_random']])
        test_accuracy_random.append([results_for_N['mean_test_accuracy_random'], results_for_N['std_test_accuracy_random']])
        test_f1_random.append([results_for_N['mean_test_f1_random'], results_for_N['std_test_f1_random']])
    
    # Convert lists to numpy arrays with shape (len(N_array), 2) for metrics
    best_accuracy = np.array(best_accuracy)
    test_accuracy = np.array(test_accuracy)
    test_f1 = np.array(test_f1)
    
    best_accuracy_random = np.array(best_accuracy_random)
    test_accuracy_random = np.array(test_accuracy_random)
    test_f1_random = np.array(test_f1_random)
    
    # Return all the arrays for each metric
    return {
        'best_accuracy': best_accuracy,  # First column: mean, second column: std
        'test_accuracy': test_accuracy,  # First column: mean, second column: std
        'test_f1': test_f1,              # First column: mean, second column: std
        'best_accuracy_random': best_accuracy_random,  # First column: mean, second column: std
        'test_accuracy_random': test_accuracy_random,  # First column: mean, second column: std
        'test_f1_random': test_f1_random   # First column: mean, second column: std
    }