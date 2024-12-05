import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from shapely.geometry import box, Polygon
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch
from sklearn.neighbors import BallTree
from tqdm import tqdm

from .visualisation import calculate_and_visualise_correlations
from ..utility import calculate_half_side_degrees
from .analysis import plot_boxplot


def plot_geometry_with_buffer_and_features(geometry, ax=None, title=None, feature_type="building"):
    """
    Plots the bounding box of a geometry object with a 1km buffer, retrieves building/amenity and landuse features,
    and visualizes them with specified colors.
    
    Parameters:
        geometry (shapely.geometry.Polygon): The geometry object to process.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. Creates a new one if None.
        title (str, optional): Title for the plot.
        feature_type (str): Type of feature to visualize ("building", "amenity", or "both").
    """
    # Calculate the bounding box of the geometry
    minx, miny, maxx, maxy = geometry.bounds
    
    # Add a 1km buffer to the bounding box
    buffer_km = 1  # Buffer distance in km
    dy, dx = calculate_half_side_degrees(((miny + maxy) / 2, (minx + maxx) / 2), buffer_km)
    buffered_bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)
    
    # Extract the buffered bbox coordinates
    buffered_minx, buffered_miny, buffered_maxx, buffered_maxy = buffered_bbox.bounds

    # Get features within the buffered bbox
    bbox = (buffered_minx, buffered_miny, buffered_maxx, buffered_maxy)

    # Retrieve building/amenity features based on feature_type
    try:
        features = {}
        if feature_type in ["building", "both"]:
            features["building"] = ox.features_from_bbox(bbox=bbox, tags={"building": True})
        if feature_type in ["amenity", "both"]:
            features["amenity"] = ox.features_from_bbox(bbox=bbox, tags={"amenity": True})
    except Exception as e:
        features = {ftype: gpd.GeoDataFrame(columns=["geometry", ftype]) for ftype in ["building", "amenity"]}
    
    try:
        landuse = ox.features_from_bbox(
            bbox=bbox,
            tags={"landuse": ["meadow", "farmland", "grass", "forest"]}
        )
    except Exception as e:
        landuse = gpd.GeoDataFrame(columns=["geometry", "landuse"])  # Empty GeoDataFrame

    # Determine top categories
    top_features = {}
    for ftype, fdata in features.items():
        if not fdata.empty:
            counts = fdata[ftype].value_counts()
            top_n = 5 if feature_type != "both" else 3
            top_features[ftype] = counts.head(top_n).index

    # Assign colors for top categories (avoiding green)
    feature_colors = {}
    color_index = 0
    for ftype, categories in top_features.items():
        for cat in categories:
            feature_colors[cat] = f"C{color_index}"
            color_index += 1
            if color_index == 2:  # Skip green
                color_index += 1

    # Create the axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the buffered bbox
    bbox_graph = ox.graph_from_bbox(bbox=bbox, network_type="all")
    edges = ox.graph_to_gdfs(bbox_graph, nodes=False)
    edges.plot(ax=ax, linewidth=0.5, edgecolor="dimgray")

    # Plot the polygon boundary if the geometry is a polygon
    if isinstance(geometry, Polygon):
        mpl_polygon = MplPolygon(
            list(geometry.exterior.coords),
            closed=True,
            edgecolor="black",
            linestyle="--",
            linewidth=2,
            fill=False,
            label="Polygon Boundary"
        )
        ax.add_patch(mpl_polygon)

    # Create legend elements manually
    legend_elements = []

    # Plot the features
    for ftype, fdata in features.items():
        if not fdata.empty:
            for category in top_features.get(ftype, []):
                fdata[fdata[ftype] == category].plot(
                    ax=ax, color=feature_colors[category], label=f"{ftype.title()}: {category}", alpha=0.7
                )
                legend_elements.append(Patch(facecolor=feature_colors[category], edgecolor='black', label=category))
            # Plot the remaining features in gray
            remaining_features = ~fdata[ftype].isin(top_features.get(ftype, []))
            fdata[remaining_features].plot(ax=ax, color="lightgray", label=f"Other {ftype.title()}", alpha=0.5)
            legend_elements.append(Patch(facecolor="lightgray", edgecolor='black', label=f"Other {ftype.title()}"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The GeoDataFrame you are attempting to plot is empty. Nothing has been displayed.")
        # Plot the landuse features in green
        landuse.plot(ax=ax, color="green", label="Landuse (green areas)", alpha=0.5)
    legend_elements.append(Patch(facecolor="green", edgecolor='black', label="Landuse"))

    # Add custom legend
    legend = ax.legend(
        handles=legend_elements,
        title="Legend",
        loc="upper left",  # Position the legend inside the plot area
        bbox_to_anchor=(1, 1),  # Move the legend outside the plot
        borderaxespad=0,  # Padding between axes and legend box
        frameon=True  # Add a frame around the legend
    )

    # Optional: Adjust legend transparency
    legend.get_frame().set_alpha(0.5)

    # Set labels and limits
    ax.set_xlim([buffered_minx, buffered_maxx])
    ax.set_ylim([buffered_miny, buffered_maxy])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Features and Landuse within Buffered BBox ({feature_type.title()})")

def plot_constituency_turnout(constituencies, turnout_column, title="Constituency Turnout Rates"):
    """
    Plots constituencies with turnout rates, using color intensity to represent turnout.

    Parameters:
        constituencies (GeoDataFrame): GeoDataFrame containing constituency geometries and turnout rates.
        turnout_column (str): The column name in the GeoDataFrame containing turnout rates.
        title (str, optional): Title of the plot. Defaults to "Constituency Turnout Rates".
    """
    import matplotlib.pyplot as plt

    # Ensure turnout rates are in the GeoDataFrame
    if turnout_column not in constituencies.columns:
        raise ValueError(f"Column '{turnout_column}' not found in GeoDataFrame.")

    # Handle missing values
    if constituencies[turnout_column].isnull().any():
        constituencies[turnout_column] = constituencies[turnout_column].fillna(0)
        print("Warning: Missing values in turnout column were replaced with 0.")

    constituencies = constituencies.set_crs("EPSG:4326")

    # Plot the constituencies
    fig, ax = plt.subplots(1, 1, figsize=(12, 15))
    constituencies.boundary.plot(ax=ax, linewidth=0.3, color="black")  # Add constituency borders
    plot = constituencies.plot(
        column=turnout_column,  # Use actual turnout rates for coloring
        cmap='viridis',  # Use reversed colormap
        ax=ax,
        legend=False  # Disable default legend
    )

    # Create a custom color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=constituencies[turnout_column].min(),
                                                                     vmax=constituencies[turnout_column].max()))
    sm._A = []  # Required for ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.05)
    cbar.set_label("Turnout Rate (%)", fontsize=12)  # Add label for the color bar
    ax.set_xlabel("Longitude", fontsize=12, labelpad=10)
    ax.set_ylabel("Latitude", fontsize=12, labelpad=10)

    # Add title and adjust axes
    ax.set_title(title, fontsize=16, fontdict={"family": "sans-serif"})
    plt.tight_layout()

    plt.show()


def plot_sparse_features_boxplots(feature_data, targets, sparse_threshold=0.99, feature_id='FID', log_transform_targets=None, grid_cols=5, figsize=(20, 5)):
    """
    Generalized function to create boxplots comparing sparse features with target variables.

    Parameters:
        feature_data (pd.DataFrame): DataFrame containing feature values and target variables.
        targets (list): List of target variable names in the DataFrame.
        sparse_threshold (float): Threshold to determine sparsity (default: 0.99).
        feature_id (str): Column name representing unique identifiers for features (default: 'FID').
        log_transform_targets (list, optional): List of target variables to apply log transformation.
        grid_cols (int): Number of columns for the plot grid (default: 5).
        figsize (tuple): Base size for the plot grid (default: (20, 5)).

    Returns:
        None: Displays the boxplots.
    """
    import logging

    # Suppress matplotlib INFO messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # Step 1: Identify sparse features
    feature_sparsity = feature_data.isin([0, np.nan]).mean()
    sparse_features = feature_sparsity[feature_sparsity > sparse_threshold].index.tolist()

    if len(sparse_features) == 0:
        print("No sparse features found based on the threshold.")
        return

    sparse_feature_counts = feature_data[sparse_features + [feature_id] + targets].copy()

    # Create presence columns for each sparse feature
    for feature in sparse_features:
        sparse_feature_counts[f"{feature}_presence"] = (sparse_feature_counts[feature] > 0).astype(int)

    for target in targets:
        # Determine the layout for subplots
        num_features = len(sparse_features)
        rows = (num_features + grid_cols - 1) // grid_cols  # Calculate rows to fit all features

        # Initialize the figure
        fig, axes = plt.subplots(rows, grid_cols, figsize=(figsize[0], rows * figsize[1]))
        axes = axes.flatten()  # Flatten axes for easy indexing

        # Iterate through sparse features
        for idx, feature in enumerate(sparse_features):
            # Select the axis
            ax = axes[idx]

            # Apply log transformation if specified
            if log_transform_targets and target in log_transform_targets:
                sparse_feature_counts[f"log_{target}"] = sparse_feature_counts[target].apply(lambda x: np.log(x + 1))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sns.boxplot(
                        x=f"{feature}_presence",
                        y=f"log_{target}",
                        data=sparse_feature_counts,
                        ax=ax,
                        hue=f"{feature}_presence"
                    )
                ax.set_ylabel(f"Log({target})")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sns.boxplot(
                        x=f"{feature}_presence",
                        y=target,
                        data=sparse_feature_counts,
                        ax=ax,
                        hue=f"{feature}_presence"
                    )
                ax.set_ylabel(target)

            # Set axis labels and title
            ax.set_xlabel(f"{feature} Presence (0: Absent, 1: Present)")
            ax.set_title(feature)

        # Remove empty subplots
        for idx in range(len(sparse_features), len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout and show the plot
        fig.tight_layout()
        fig.suptitle(f"Comparison of Sparse Features with {target}", fontsize=16, y=1.02)
        plt.show()


# Function to build a KDTree and query nearby points
def get_nearby_indices_kdtree(city_df, lat, lon, radius=3):
    """Get nearby OAs within 2km using KDTree."""
    coords = np.radians(city_df[['LAT', 'LONG']].values)
    tree = BallTree(coords, metric='haversine')
    point = np.radians([[lat, lon]])
    radius_in_radians = radius / 6371  # Earth's radius in km
    indices = tree.query_radius(point, r=radius_in_radians)
    return indices[0]

def process_city(city_df, process_type, feature_revised, radius=3):
    """
    Process all OAs within a city based on the specified processing type.

    Parameters:
    - city_df (pd.DataFrame): DataFrame of OAs for a single city.
    - process_type (str): Either "L15_proportion" or "population_density".
    - radius (int): Radius in km for nearby OAs aggregation.

    Returns:
    - List of transformed rows for the city.
    """
    transformed_city_data = []
    
    for idx, row in city_df.iterrows():
        lat, lon = row['LAT'], row['LONG']
        nearby_indices = get_nearby_indices_kdtree(city_df, lat, lon, radius=radius)
        nearby_oas = city_df.iloc[nearby_indices]
        
        # Common aggregations
        total_area = nearby_oas['area'].sum()
        
        if process_type == "student_proportion":
            # Aggregate features specific to student_proportion
            total_L15_students = nearby_oas['L15_full_time_students'].sum()
            total_residents = nearby_oas['total_residents'].sum()
            
            # Recalculate L15 proportion
            new_L15_proportion = total_L15_students / total_residents if total_residents > 0 else 0
            
            # Create a new row
            new_row = row.to_dict()
            for key, value in nearby_oas[feature_revised].sum().to_dict().items():
                new_row[f"aggregated_{key}"] = value
            new_row['aggregated_L15_full_time_students'] = total_L15_students
            new_row['aggregated_total_residents'] = total_residents
            new_row['new_L15_proportion'] = new_L15_proportion
            new_row['aggregated_area'] = total_area
            
        elif process_type == "population_density":
            # Aggregate features specific to population density
            total_population = nearby_oas['population'].sum()
            
            # Recalculate population density
            new_population_density = total_population / total_area if total_area > 0 else 0
            
            # Create a new row
            new_row = row.to_dict()
            for key, value in nearby_oas[feature_revised].sum().to_dict().items():
                new_row[f"aggregated_{key}"] = value
            new_row['aggregated_population'] = total_population
            new_row['new_population_density'] = new_population_density
            new_row['aggregated_area'] = total_area
        
        else:
            raise ValueError("Invalid process_type. Must be 'L15_proportion' or 'population_density'.")
        
        # Append the transformed row
        transformed_city_data.append(new_row)
    
    return transformed_city_data

# Main function to process the entire dataset
def process_all_cities(df, process_type, feature_revised, radius=3):
    """
    Process all cities and transform the data based on the specified processing type.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing all OAs.
    - process_type (str): Either "L15_proportion" or "population_density".
    - radius (int): Radius in km for nearby OAs aggregation.

    Returns:
    - Transformed DataFrame.
    """
    transformed_data = []
    
    # Group by city for efficient processing
    city_groups = df.groupby("city")
    
    for city, city_df in tqdm(city_groups, desc=f"Processing cities for {process_type}"):
        city_df = city_df.reset_index(drop=True)  # Reset index for each city
        transformed_data.extend(process_city(city_df, process_type, feature_revised=feature_revised[process_type], radius=radius))  # Process each city and append results
    
    return pd.DataFrame(transformed_data)

def plot_polling_station_turnout(pcon_boundary_data, pcon_feature_counts, election_results):
    # Reproject boundaries and calculate area
    pcon25_boundary_tmp = pcon_boundary_data.copy()
    pcon25_boundary_tmp = pcon25_boundary_tmp.to_crs("EPSG:27700")
    pcon25_boundary_tmp['area'] = pcon25_boundary_tmp['geometry'].area

    # Combine polling station features into one column
    pcon25_features = pcon_feature_counts.copy()
    pcon25_features['polling_station'] = (
        pcon25_features['amenity_polling_station'] +
        pcon25_features['polling_station_yes'] +
        pcon25_features['polling_station_ballot_box']
    )
    pcon25_features.drop(['amenity_polling_station', 'polling_station_yes', 'polling_station_ballot_box'], axis=1, inplace=True)

    # Get the list of all features
    all_features = pcon25_features.columns.tolist()[1:]

    # Map turnout rates to the features dataframe
    pcon25_features['turnout_rate'] = pcon25_features['PCON25CD'].map(election_results.set_index('ONS_ID')['Turnout_rate'])

    # Merge with boundary data to include area
    pcon25_features = pcon25_features.merge(
        pcon25_boundary_tmp[['PCON25CD', 'area', 'geometry']],
        on='PCON25CD'
    )

    # Calculate density of features (per square kilometer)
    pcon25_features_density = pcon25_features.copy()
    for feature in all_features:
        pcon25_features_density[feature] = (
            pcon25_features_density[feature] / pcon25_features_density['area'] * 1e6
        )

    # Add binary column for polling station presence
    pcon25_features_tmp = pcon25_features.copy()
    pcon25_features_tmp['polling_station'] = pcon25_features_tmp['polling_station'].apply(lambda x: "True" if x > 0 else "False")

    plot_boxplot(
        pcon25_features_tmp,
        'polling_station',
        'turnout_rate',
        'Turnout Rate vs Polling Station',
        'Polling Station Presence',
        'Turnout Rate (%)'
    )

    if 'polling_station' in all_features:
        all_features.remove('polling_station')

    # Calculate correlations between turnout_rate and all features
    correlations = (
        pcon25_features_tmp[all_features + ['turnout_rate']]
        .corr()['turnout_rate']
        .sort_values(ascending=False)
    )
    correlations = correlations.drop(['turnout_rate'])  # Remove self-correlation

    # Create Community Engagement Index
    engagement_features = all_features.copy()

    # Normalize engagement-related features
    pcon25_features_tmp = pcon25_features.copy()
    for feature in engagement_features:
        pcon25_features_tmp[feature] = (
            pcon25_features_tmp[feature] - pcon25_features_tmp[feature].min()
        ) / (
            pcon25_features_tmp[feature].max() - pcon25_features_tmp[feature].min()
        )

    # Calculate Engagement Index using correlations as weights
    pcon25_features['engagement_index'] = 0
    for feature in engagement_features:
        pcon25_features['engagement_index'] += (
            pcon25_features_tmp[feature] * correlations[feature]
        )

    # Correlation between Engagement Index and Turnout Rate
    engagement_turnout_corr = (
        pcon25_features[['engagement_index', 'turnout_rate']]
        .corr()['turnout_rate']['engagement_index']
    )
    print(f"Correlation between Engagement Index and Turnout Rate: {engagement_turnout_corr}")

    # Plot the correlation between Engagement Index and Turnout Rate
    calculate_and_visualise_correlations(
        pcon25_features,
        [{"x_col": "engagement_index", "y_col": "turnout_rate"}],
        figsize=(10, 8)
    )

    return pcon25_features