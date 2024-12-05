from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

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