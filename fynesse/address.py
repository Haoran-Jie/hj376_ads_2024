# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import osmnx as ox
import geopandas as gpd
import seaborn as sns
import statsmodels.api as sm
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew, kurtosis




def kmeans_with_scaling(pivot_df, num_clusters=3, random_state=42, index_name="Location"):
    """
    Perform KMeans clustering with scaling on the input pivoted DataFrame.
    
    Parameters:
    - pivot_df (pd.DataFrame): A DataFrame where each row represents a location and 
                               columns represent counts of POI types.
    - num_clusters (int): The number of clusters for KMeans.
    - random_state (int): Random state for reproducibility.
    - index_name (str): Name for the index column in the resulting DataFrame.
    
    Returns:
    - pd.DataFrame: A DataFrame with the original locations and their assigned clusters.
    """
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Create a DataFrame to store the index (location) and their assigned clusters
    cluster_df = pd.DataFrame({index_name: pivot_df.index, 'Cluster': clusters})
    
    return cluster_df

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

def features_from_coordinate(lat, lon, revised_columns, refined_tags, radius=500):
    """Get features from OpenStreetMap around a given coordinate."""

    gdf = ox.features_from_point((lat, lon), tags=refined_tags, dist=radius)

    # Extract the feature counts
    feature_counts = {}
    for key, values in refined_tags.items():
        for value in values:
            column_name = f"aggregated_{key}_{value}"
            feature_counts[column_name] = (gdf[key]==value).sum()

    revised_counts = {}
    for key in revised_columns:
        revised_counts[f"aggregated_{key}"] = feature_counts[f"aggregated_{key}"]

    return revised_counts


def plot_regression_results(y_test, y_pred, residuals, title_prefix):
    """
    Generalized function to plot regression results.
    
    Parameters:
        y_test (array-like): True values for the test set.
        y_pred (array-like): Predicted values from the model.
        residuals (array-like): Residuals from the model.
        title_prefix (str): Prefix for the plot titles to distinguish between transformations.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Residual Plot
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title(f"{title_prefix} Residual Plot")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].grid(alpha=0.3)

    # Histogram/KDE of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color="blue")
    axes[1].set_title(f"{title_prefix} Histogram/KDE of Residuals")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")

    # Actual vs. Predicted Plot
    axes[2].scatter(y_test, y_pred, alpha=0.6)
    axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
    axes[2].set_title(f"{title_prefix} Actual vs Predicted")
    axes[2].set_xlabel("Actual Values")
    axes[2].set_ylabel("Predicted Values")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_transformations(y, transformations, titles, figsize=(15, 10), bins=50):
    """
    Generalized function to plot the original and transformed distributions.
    
    Parameters:
        y (array-like): Original target variable.
        transformations (dict): Dictionary of transformations with keys as transformation names
                                and values as functions to apply to `y`.
        titles (list): Titles for each subplot. The first title should correspond to the original data.
        figsize (tuple): Size of the entire figure (default: (15, 10)).
        bins (int): Number of bins for histograms (default: 50).
    """
    # Create a grid for subplots
    num_transformations = len(transformations)
    rows = (num_transformations + 1) // 3 + 1  # Add 1 row for the original distribution
    cols = min(num_transformations + 1, 3)  # At most 3 columns
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Plot the original distribution
    axes[0].hist(y, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(titles[0])

    # Apply each transformation and plot
    for i, (name, transform) in enumerate(transformations.items(), start=1):
        y_transformed = transform(y)
        axes[i].hist(y_transformed, bins=bins, color=np.random.rand(3,), edgecolor='black', alpha=0.7)
        axes[i].set_title(titles[i])

    # Turn off any extra axes
    for i in range(num_transformations + 1, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def estimate_students(latitude: float, longitude: float) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: Estimated share of students in that area (value between 0 and 1).
    """
    # Get features around the given coordinate
    features = features_from_coordinate(latitude, longitude, feature_revised["student_proportion"], radius=3000)
    
    area = np.pi * 3000**2 / 1e6 # Area of the circle with radius 3000

    # Normalize the features
    features = {f"{k}_per_km_2": v / area for k, v in features.items()}
    pca_loaded = load("l15_pca_10.joblib")
    rf_loaded = load("l15_rf_Original.joblib")

    # Apply PCA
    X = pca_loaded.transform(pd.DataFrame([features]))

    # Predict the student proportion
    student_proportion = rf_loaded.predict(X)[0]
    
    return student_proportion


def estimate_population_density(latitude: float, longitude: float) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: Estimated population density in that area.
    """
    # Get features around the given coordinate
    features = features_from_coordinate(latitude, longitude, feature_revised["population_density"],radius=3000)
    
    area = np.pi * 3000**2 / 1e6 # Area of the circle with radius 500m

    # Normalize the features
    features = {f"{k}_per_km_2": v / area for k, v in features.items()}
    pca_loaded = load("population_density_pca_10.joblib")
    rf_loaded = load("population_density_rf.joblib")

    # Apply PCA
    X = pca_loaded.transform(pd.DataFrame([features]))

    # Predict the population density
    population_density = np.exp(rf_loaded.predict(X)[0])
    
    return population_density



def plot_regression_diagnostics(
    y_train, y_test, y_pred_train, y_pred_test,
    residuals_train, residuals_test,
    train_sizes, train_scores_mean, test_scores_mean,
    feature_names, coefficients
):
    """
    Plot diagnostics for regression model performance.

    Parameters:
    - y_train: Series or array, actual target values for training data.
    - y_test: Series or array, actual target values for testing data.
    - y_pred_train: Series or array, predicted target values for training data.
    - y_pred_test: Series or array, predicted target values for testing data.
    - residuals_train: Series or array, residuals for training data.
    - residuals_test: Series or array, residuals for testing data.
    - train_sizes: Array of training set sizes for learning curve.
    - train_scores_mean: Array of mean training scores for learning curve.
    - test_scores_mean: Array of mean test scores for learning curve.
    - feature_names: List of feature names.
    - coefficients: Array of feature coefficients.
    """
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.tight_layout(pad=6.0)

    # 1. Residual Plot
    axs[0, 0].scatter(y_pred_train, residuals_train, alpha=0.7)
    axs[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axs[0, 0].set_title("Residual Plot (Training Data)")
    axs[0, 0].set_xlabel("Predicted values")
    axs[0, 0].set_ylabel("Residuals")

    # 2. Residual Distribution Plot
    sns.histplot(residuals_train, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title("Residual Distribution Plot")
    axs[0, 1].set_xlabel("Residuals")

    # 3. Y_real vs. Y_pred
    axs[1, 0].scatter(y_test, y_pred_test, alpha=0.7)
    axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                   color="red", linestyle="--", linewidth=1)
    axs[1, 0].set_title("Y_real vs. Y_pred")
    axs[1, 0].set_xlabel("Actual Values")
    axs[1, 0].set_ylabel("Predicted Values")

    # 4. Q-Q Plot
    sm.qqplot(residuals_train, line='45', ax=axs[1, 1])
    axs[1, 1].set_title("Q-Q Plot (Training Residuals)")

    # 5. Learning Curve
    axs[2, 0].plot(train_sizes, train_scores_mean, label="Training score")
    axs[2, 0].plot(train_sizes, test_scores_mean, label="Cross-validation score")
    axs[2, 0].set_title("Learning Curve")
    axs[2, 0].set_xlabel("Training set size")
    axs[2, 0].set_ylabel("R^2 score")
    axs[2, 0].legend(loc="best")

    # 6. Feature Coefficients Barchart
    axs[2, 1].barh(feature_names, coefficients)
    axs[2, 1].set_title("Feature Coefficients")
    axs[2, 1].set_xlabel("Coefficient Value")

    # Display plots
    plt.show()



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


def plot_model_results(alphas, ridge_scores, lasso_scores, ridge_model, lasso_model, X_columns, best_ridge_alpha, best_lasso_alpha):
    """
    Plot the R^2 scores and feature coefficients for Ridge and Lasso models.

    Parameters:
        - alphas: List of alpha values
        - ridge_scores, lasso_scores: R^2 scores for Ridge and Lasso
        - ridge_model, lasso_model: Best Ridge and Lasso models
        - X_columns: List of feature names
        - best_ridge_alpha, best_lasso_alpha: Best alpha values for Ridge and Lasso
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.tight_layout(pad=6.0)

    # Ridge Alpha vs. R^2
    axs[0, 0].plot(alphas, ridge_scores, marker='o')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title("Ridge Regression: Alpha vs R^2")
    axs[0, 0].set_xlabel("Alpha")
    axs[0, 0].set_ylabel("R^2 Score")
    axs[0, 0].axvline(best_ridge_alpha, color='red', linestyle='--', label=f"Best Alpha: {best_ridge_alpha:.3e}")
    axs[0, 0].legend()

    # Lasso Alpha vs. R^2
    axs[0, 1].plot(alphas, lasso_scores, marker='o')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title("Lasso Regression: Alpha vs R^2")
    axs[0, 1].set_xlabel("Alpha")
    axs[0, 1].set_ylabel("R^2 Score")
    axs[0, 1].axvline(best_lasso_alpha, color='red', linestyle='--', label=f"Best Alpha: {best_lasso_alpha:.3e}")
    axs[0, 1].legend()

    # Feature Coefficients for Best Ridge Model
    axs[1, 0].barh(X_columns, ridge_model.coef_)
    axs[1, 0].set_title(f"Feature Coefficients (Best Ridge, Alpha={best_ridge_alpha:.3e})")
    axs[1, 0].set_xlabel("Coefficient Value")

    # Feature Coefficients for Best Lasso Model
    axs[1, 1].barh(X_columns, lasso_model.coef_)
    axs[1, 1].set_title(f"Feature Coefficients (Best Lasso, Alpha={best_lasso_alpha:.3e})")
    axs[1, 1].set_xlabel("Coefficient Value")

    plt.show()


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

def plot_residuals(model, data, target):
    """
    Plot residuals for a regression model.

    Parameters:
        model (sklearn.linear_model): The trained regression model.
        data (pd.DataFrame): The input data.
        target (str): The target column to predict.
    """
    X = data.drop(target, axis=1)
    y = data[target]

    y_pred = model.predict(X)
    residuals = y - y_pred

    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid()
    plt.show()

def plot_learning_curve_subplots(models, data_dict, target, cv=5, scoring='r2'):
    """
    Plot learning curves for all years in subplots.

    Parameters:
        models (dict): Dictionary of trained models.
        data_dict (dict): Dictionary of data for each year.
        target (str): The target column to predict.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric for cross-validation.
    """
    n_years = len(data_dict)
    fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5), sharey=True)

    if n_years == 1:
        axes = [axes]

    for ax, (year, model_data) in zip(axes, models.items()):
        data = data_dict[year]
        X = data.drop(target, axis=1)
        y = data[target]

        train_sizes, train_scores, test_scores = learning_curve(
            model_data['model'], X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        ax.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
        ax.set_title(f"Learning Curve: {year}")
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring)
        ax.legend(loc="best")
        ax.grid()

    plt.tight_layout()
    plt.show()


def plot_residual_distributions(models, data_dict, target):
    """
    Plot residual distributions for all years in subplots with mean, SD, skewness, and kurtosis.

    Parameters:
        models (dict): Dictionary of trained models.
        data_dict (dict): Dictionary of data for each year.
        target (str): The target column to predict.
    """
    n_years = len(data_dict)
    n_cols = n_years  # Up to 4 columns per row
    n_rows = -(-n_years // n_cols)  # Calculate rows needed (ceiling division)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)

    # Flatten axes array if multiple rows/columns
    axes = axes.flatten() if n_years > 1 else [axes]

    for ax, (year, model_data) in zip(axes, models.items()):
        data = data_dict[year]
        X = data.drop(target, axis=1)
        y = data[target]

        y_pred = model_data['model'].predict(X)
        residuals = y - y_pred

        # Calculate statistics
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        skew_residual = skew(residuals)
        kurtosis_residual = kurtosis(residuals)

        # Plot the residual distribution
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title(f"Residuals Distribution: {year}")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.grid()

        # Add statistics as text
        stats_text = (
            f"Mean: {mean_residual:.2f}\n"
            f"SD: {std_residual:.2f}\n"
            f"Skew: {skew_residual:.2f}\n"
            f"Kurtosis: {kurtosis_residual:.2f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8))

    # Hide unused subplots if n_years < n_rows * n_cols
    for ax in axes[len(models):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def train_random_forest(data, target, n_estimators=100, max_depth=None, cv=5):
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

def plot_feature_importance_changes(feature_importance_dict, top_n=10):
    """
    Plot changes in feature importance over the years.

    Parameters:
        feature_importance_dict (dict): Dictionary of feature importances for each year.
        top_n (int): Number of top features to display.
    """
    
    for year in feature_importance_dict.keys():
        feature_importance_dict[year].index = feature_importance_dict[year].index.str[:-5]

    # Combine feature importances into a single DataFrame
    importance_df = pd.DataFrame(feature_importance_dict).T

    # Normalize feature importance values for comparison
    importance_df = importance_df.div(importance_df.sum(axis=1), axis=0)

    # Get the top N features overall
    top_features = importance_df.mean().nlargest(top_n).index

    # Plot feature importance changes for top features
    importance_df[top_features].plot(kind='line', figsize=(12, 8), marker='o')
    plt.title("Feature Importance Changes Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Normalized Importance")
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_residuals_heatmap(models, data_dict, target_column='turnout', cmap='coolwarm'):
    """
    Plot a heatmap of residuals for all years.

    Parameters:
        models (dict): A dictionary of trained models for each year.
        data_dict (dict): A dictionary of data for each year.
        target_column (str): The name of the target column in the dataset.
        cmap (str): The colormap to use for the heatmap (default is 'coolwarm').
    """
    # Create a DataFrame of residuals for each year
    residuals_df = pd.DataFrame({
        year: data[target_column] - models[year]['model'].predict(data.drop(target_column, axis=1))
        for year, data in data_dict.items()
    })
    
    # Plot the heatmap
    sns.heatmap(residuals_df, cmap=cmap, center=0, annot=False)
    plt.title("Residuals Heatmap Over Years")
    plt.xlabel("Year")
    plt.ylabel("Samples")
    plt.show()