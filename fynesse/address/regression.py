import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from .visualization import plot_model_results

def train_model(data, target, model='linear', alpha=0.01, cv=5):
    """
    Train and evaluate a regression model on the given data with cross-validation.

    Parameters:
        data (pd.DataFrame): The input data.
        target (str): The target column to predict.
        model (str): The type of regression model ('linear', 'lasso', 'ridge').
        alpha (float): Regularization strength for Lasso and Ridge regression.
        cv (int): Number of cross-validation folds.

    Returns:
        dict: Trained model, cross-validation scores, and mean CV score.
    """
    X = data.drop(target, axis=1)
    y = data[target]

    if model == 'linear':
        regressor = LinearRegression()
    elif model == 'lasso':
        regressor = Lasso(alpha=alpha)
    elif model == 'ridge':
        regressor = Ridge(alpha=alpha)
    else:
        raise ValueError("Invalid model type. Use 'linear', 'lasso', or 'ridge'.")

    # Cross-validation
    cv_scores = cross_val_score(regressor, X, y, cv=cv, scoring='r2')
    regressor.fit(X, y)

    return {
        'model': regressor,
        'cv_scores': cv_scores,
        'mean_cv_score': np.mean(cv_scores)
    }


def evaluate_model(model, data, target):
    """
    Evaluate a regression model on the given data.

    Parameters:
        model (sklearn.linear_model): The trained regression model.
        data (pd.DataFrame): The input data.
        target (str): The target column to predict.

    Returns:
        dict: Dictionary of evaluation metrics (R^2, MSE).
    """
    X = data.drop(target, axis=1)
    y = data[target]

    y_pred = model.predict(X)

    r2 = model.score(X, y)
    rmse = root_mean_squared_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)

    return {'R^2': r2, 'RMSE': rmse}


def find_best_model(X_train, y_train, X_test, y_test, model_class, alphas, **model_kwargs):
    """
    Find the best model and alpha using cross-validation for Ridge or Lasso regression.
    
    Parameters:
        - X_train, y_train: Training data
        - X_test, y_test: Test data
        - model_class: Model class (Ridge or Lasso)
        - alphas: List of alpha values to test
        - model_kwargs: Additional keyword arguments for the model
        
    Returns:
        - best_model: The best model instance
        - best_alpha: The best alpha value
        - best_r2: The best R^2 score
        - scores: List of R^2 scores for each alpha
    """
    best_model = None
    best_alpha = None
    best_r2 = -np.inf
    scores = []

    for alpha in alphas:
        model = model_class(alpha=alpha, **model_kwargs)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        scores.append(r2)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_alpha = alpha

    return best_model, best_alpha, best_r2, scores


def ridge_and_lasso_analysis(X_train, y_train, X_test, y_test, X_columns):
    """
    Perform Ridge and Lasso analysis, finding the best models, and plotting results.

    Parameters:
        - X_train, y_train: Training data
        - X_test, y_test: Test data
        - X_columns: List of feature names
    """
    alphas = np.logspace(-5, 0, 10)  # Range of alpha values

    # Find best Ridge model
    best_ridge_model, best_ridge_alpha, best_ridge_r2, ridge_scores = find_best_model(
        X_train, y_train, X_test, y_test, Ridge, alphas
    )

    # Find best Lasso model
    best_lasso_model, best_lasso_alpha, best_lasso_r2, lasso_scores = find_best_model(
        X_train, y_train, X_test, y_test, Lasso, alphas, max_iter=10000
    )

    # Plot results
    plot_model_results(
        alphas, ridge_scores, lasso_scores,
        best_ridge_model, best_lasso_model,
        X_columns, best_ridge_alpha, best_lasso_alpha
    )

    # Print the results
    print("Best Ridge Model:")
    print(f"Alpha: {best_ridge_alpha}")
    print(f"Test R^2: {best_ridge_r2}")

    print("\nBest Lasso Model:")
    print(f"Alpha: {best_lasso_alpha}")
    print(f"Test R^2: {best_lasso_r2}")


def linear_regression_analysis(X, y, X_train, y_train, X_test, y_test):
    """
    Perform linear regression analysis, including training, predictions, residuals,
    metrics, cross-validation, and learning curve.

    Parameters:
        - X: Full feature set
        - y: Target variable
        - X_train: Training feature set
        - y_train: Training target variable
        - X_test: Testing feature set
        - y_test: Testing target variable

    Returns:
        - linear_model: Trained linear regression model
        - y_pred_train: Predictions on training data
        - y_pred_test: Predictions on testing data
        - residuals_train: Residuals for training data
        - residuals_test: Residuals for testing data
        - train_r2: R^2 score for training data
        - test_r2: R^2 score for testing data
        - train_mse: MSE for training data
        - test_mse: MSE for testing data
        - cv_scores: Cross-validation R^2 scores
        - train_sizes: Array of training sizes for learning curve
        - train_scores_mean: Mean training scores for learning curve
        - test_scores_mean: Mean testing scores for learning curve
    """
    # Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = linear_model.predict(X_train)
    y_pred_test = linear_model.predict(X_test)

    # Residuals
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = root_mean_squared_error(y_train, y_pred_train)
    test_rmse = root_mean_squared_error(y_test, y_pred_test)

    # Cross-validation scores
    cv_scores = cross_val_score(linear_model, X, y, cv=5, scoring="r2")

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        linear_model, X, y, cv=5, scoring="r2", train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    return (
        linear_model,
        y_pred_train,
        y_pred_test,
        residuals_train,
        residuals_test,
        train_r2,
        test_r2,
        train_rmse,
        test_rmse,
        cv_scores,
        train_sizes,
        train_scores_mean,
        test_scores_mean
    )


# Define the function to train the Random Forest
def train_random_forest_data(data, target, n_estimators=100, max_depth=None, cv=5):
    """
    Train a Random Forest regression model and compute feature importance.

    Parameters:
        data (pd.DataFrame): The input data.
        target (str): The target column to predict.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): The maximum depth of the tree.
        cv (int): Number of cross-validation folds.

    Returns:
        dict: Trained model, feature importances, cross-validation scores, and mean CV score.
    """
    X = data.drop(target, axis=1)
    y = data[target]

    regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Cross-validation
    cv_scores = cross_val_score(regressor, X, y, cv=cv, scoring='r2')
    regressor.fit(X, y)

    # Feature importance
    feature_importances = pd.Series(regressor.feature_importances_, index=X.columns)

    return {
        'model': regressor,
        'cv_scores': cv_scores,
        'mean_cv_score': np.mean(cv_scores),
        'feature_importances': feature_importances
    }

def train_with_pca(X, y, n_components=None):
    """Apply PCA to the features, train a Random Forest model, and visualize PCA details."""
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Visualize explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Proportion of Variance Captured by PCA")
    plt.grid()
    plt.show()
    
    # Contribution of original features to principal components
    feature_contributions = pd.DataFrame(
        pca.components_,
        columns=X.columns,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        feature_contributions, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f", 
        cbar_kws={'label': 'Contribution'}
    )
    plt.title("Original Feature Contributions to Principal Components")
    plt.xlabel("Original Features")
    plt.ylabel("Principal Components")
    plt.show()
    
    return rf_model, train_r2, test_r2, pca

def train_random_forest(X, y):
    """Train a Random Forest model and return the model and R² scores."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    return rf_model, train_r2, test_r2