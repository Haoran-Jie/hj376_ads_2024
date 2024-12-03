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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import osmnx as ox
import geopandas as gpd
import seaborn as sns



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