import os
import requests
import geopandas as gpd
import numpy as np
import pandas as pd
import osmnx as ox
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



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


def estimate_students(latitude: float, longitude: float, feature_revised) -> float:
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


def estimate_population_density(latitude: float, longitude: float, feature_revised) -> float:
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