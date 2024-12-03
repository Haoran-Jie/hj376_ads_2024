from .config import *

from . import access
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
from .access import create_connection, fetch_houses_within_box
from .utility import calculate_half_side_degrees
import logging
from tabulate import tabulate
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box, Polygon, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes sure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def filter_full_address(buildings):
    """Filters buildings with complete address information."""
    address_full_condition = buildings['addr:housenumber'].notna() & buildings['addr:street'].notna() & buildings['addr:postcode'].notna()
    return buildings[address_full_condition]

def filter_building_types(buildings, non_residential, utility):
    """Filters out non-residential and utility buildings."""
    filtered = buildings[~buildings['building'].isin(non_residential + utility)]
    print(len(buildings) - len(filtered), "buildings are filtered out by non-residential building types.")
    return filtered

def preprocess_street_name(df, street_col, house_col):
    """Preprocesses the street names and house numbers for consistent matching."""
    # Convert street name to lowercase and remove spaces
    df.loc[:, street_col] = df.loc[:, street_col].str.lower().str.replace(" ", "")
    # Ensure house number is integer
    df.loc[:, house_col] = df.loc[:, house_col].astype(str)
    return df

def match_addresses(pp_df, osm_df):
    """Matches PP and OSM data based on primary and secondary addressable object names."""
    merged_primary = pd.merge(pp_df, osm_df, left_on=['street', 'primary_addressable_object_name'], right_on=['addr:street', 'addr:housenumber'], how='inner')
    unmatched_pp = pp_df[~pp_df.index.isin(merged_primary.index)]
    merged_secondary = pd.merge(unmatched_pp, osm_df, left_on=['street', 'secondary_addressable_object_name'], right_on=['addr:street', 'addr:housenumber'], how='inner')

    # Remove duplicates from secondary matches based on primary match
    duplicates = pd.merge(merged_secondary[['addr:street', 'addr:housenumber']], merged_primary[['addr:street', 'addr:housenumber']], on=['addr:street', 'addr:housenumber'], how='inner')
    merged_secondary = merged_secondary[~merged_secondary.set_index(['addr:street', 'addr:housenumber']).index.isin(duplicates.set_index(['addr:street', 'addr:housenumber']).index)]

    return pd.concat([merged_primary, merged_secondary], ignore_index=True)

def filter_by_postcode(matched_df):
    """Filters matched DataFrame by postcode agreement."""
    original_len = len(matched_df)
    matched_df = matched_df[matched_df['postcode'] == matched_df['addr:postcode']]
    print("Number of matched buildings removed due to postcode mismatch:", original_len - len(matched_df))
    return matched_df

def filter_by_building_type(matched_df, valid_matches):
    """Filters matched DataFrame by valid building type matches based on property type."""
    original_len = len(matched_df)
    matched_df = matched_df[
        matched_df.apply(lambda x: x['building'] in valid_matches[x['property_type']] if valid_matches[x['property_type']] is not None else True, axis=1)
    ]
    print("Number of matched buildings removed due to building type mismatch:", original_len - len(matched_df))
    return matched_df


def filter_and_match(place_name, latitude, longitude, side_length_km, username, password, url):
    
    from .access import fetch_building_within_bbox
    non_residential_types = ['church', 'shed', 'dormitory', 'hotel', 'construction']
    utility_types = ['entrance', 'service']
    valid_matches = {
        'D': ['detached', 'house', 'residential'],
        'S': ['semidetached_house', 'house', 'residential'],
        'T': ['terrace', 'house', 'residential'],
        'F': ['apartments', 'house', 'residential', 'commercial'],
        'O': None  # Allow 'O' to match with any building type
    }

    buildings, area, nodes, edges = fetch_building_within_bbox(place_name, latitude, longitude, side_length_km, draw=False)
    conn = create_connection(username, password, url, 'ads_2024')
    housing_pp_df = fetch_houses_within_box((latitude, longitude), 2, conn, since_date='1995-01-01')

    full_address_buildings_df = filter_full_address(buildings)
    filtered_buildings = filter_building_types(full_address_buildings_df, non_residential_types, utility_types)
    preprocess_street_name(filtered_buildings, 'addr:street', 'addr:housenumber')
    preprocess_street_name(housing_pp_df, 'street', 'primary_addressable_object_name')
    preprocess_street_name(housing_pp_df, 'street', 'secondary_addressable_object_name')

    final_matched_df = match_addresses(housing_pp_df, filtered_buildings)
    final_matched_df = filter_by_postcode(final_matched_df)
    final_matched_df = filter_by_building_type(final_matched_df, valid_matches)

    return final_matched_df


def plot_boxplot_price_by_property_type(final_matched_df):
    """Plot the boxplot of price for each property type."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=final_matched_df, x='property_type', y='price')
    plt.yscale('log')
    plt.title("Boxplot of Price by Property Type (Log Scale)")
    plt.xlabel("Property Type")
    plt.ylabel("Price (log scale)")
    plt.show()


def plot_geodata(geodf, edges, area, xlim, ylim, condition=None, label = None, title = "Map Visualisation"):
    """
    Plot GeoDataFrame with nodes, edges, and area.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    area.plot(ax=ax, facecolor="white")
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    if condition is not None:
        geodf[condition].plot(ax=ax, color="red", alpha=0.7, markersize=10, label=label)
        geodf[~condition].plot(ax=ax, color="blue", alpha=0.7, markersize=10, label="Not " + label)
    else:
        geodf.plot(ax=ax, color="red", alpha=0.7, markersize=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.legend()

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_regularization_results(ridge_df, lasso_df, baseline_rmse, baseline_r2):
    """
    Plot the regularization results for Ridge and Lasso models.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # RMSE Plot
    axs[0].plot(ridge_df['alpha'], ridge_df['rmse'], marker='o', label='Ridge RMSE', color='b')
    axs[0].plot(lasso_df['alpha'], lasso_df['rmse'], marker='o', label='Lasso RMSE', color='r')
    axs[0].axhline(baseline_rmse, color='g', linestyle='--', label='Baseline RMSE')
    axs[0].set_xscale('log')
    axs[0].set_ylabel('RMSE')
    axs[0].set_xlabel('Alpha (Regularization Strength)')
    axs[0].legend()
    axs[0].grid(True)

    # R² Plot
    axs[1].plot(ridge_df['alpha'], ridge_df['r2'], marker='o', label='Ridge R²', color='b')
    axs[1].plot(lasso_df['alpha'], lasso_df['r2'], marker='o', label='Lasso R²', color='r')
    axs[1].axhline(baseline_r2, color='g', linestyle='--', label='Baseline R²')
    axs[1].set_xscale('log')
    axs[1].set_ylabel('R²')
    axs[1].set_xlabel('Alpha (Regularization Strength)')
    axs[1].legend()
    axs[1].grid(True)

    plt.show()

def calculate_corr(y, y_pred):
    return np.corrcoef(y, y_pred)[0, 1]


def plot_coefficients(coefficients_df):
    """
    Plot the coefficients for each feature across age groups.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in coefficients_df.columns:
        ax.plot(range(100), coefficients_df[col], label=col)

    ax.set_xlabel("Age Group")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Model Coefficients Across Age Groups")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def remove_duplicate_columns(df, dataset_name="DataFrame"):
    """Remove duplicate columns from the DataFrame."""
    if df.columns.duplicated().any():
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"{dataset_name}: Duplicate columns identified for removal: {duplicate_cols}")
        deduplicated_df = df.loc[:, ~df.columns.duplicated()]
        print(f"{dataset_name}: Removed duplicate columns. New shape: {deduplicated_df.shape}")
        return deduplicated_df
    else:
        print(f"{dataset_name}: No duplicate columns identified.")
        return df

def remove_empty_columns(df, dataset_name="DataFrame"):
    """Remove columns with all zero values."""
    empty_columns = df.columns[(df == 0).all()]
    if empty_columns.empty:
        print(f"{dataset_name}: No empty columns identified.")
        return df, empty_columns
    print(f"{dataset_name}: Empty columns identified for removal: {empty_columns.tolist()}")
    cleaned_df = df.drop(columns=empty_columns)
    print(f"{dataset_name}: Shape after removing empty columns: {cleaned_df.shape}")
    return cleaned_df, empty_columns

def update_non_empty_tags(refined_tags, df):
    """Update refined_tags to include only tags with non-zero values."""
    non_empty_tags = {
        tag: [
            value for value in refined_tags[tag]
            if f"{tag}_{value}" in df.columns and df[f"{tag}_{value}"].sum() > 0
        ]
        for tag in refined_tags
    }
    return non_empty_tags

def find_rows_with_all_zeros(df, feature_columns, dataset_name="DataFrame"):
    """Identify rows with all zero values in specific columns."""
    rows_with_all_zeros = df.index[(df[feature_columns] == 0).all(axis=1)]
    print(f"{dataset_name}: Rows with all zeros in feature columns: {len(rows_with_all_zeros)} out of {df.shape[0]}")
    return rows_with_all_zeros

def check_columns_types(df, columns, expected_types, dataset_name="DataFrame"):
    """Check the data types of columns in the DataFrame."""
    print(f"{dataset_name}: Checking data types for {len(columns)} columns...")
    issues = []
    
    for column, expected_type in zip(columns, expected_types):
        actual_type = df[column].dtype
        if actual_type != expected_type:
            issues.append([column, str(expected_type), str(actual_type)])
    
    if issues:
        logger.warning(f"{dataset_name}: Found {len(issues)} type mismatches:")
        print(tabulate(issues, headers=["Column", "Expected Type", "Actual Type"], tablefmt="pretty"))
    else:
        print(f"{dataset_name}: All columns have expected data types.")

def calculate_and_visualise_correlations(
    data, analyses, grid_size=None, figsize=(16, 10)
):
    import math
    """
    Generalized function to calculate and visualize correlations in subplots.

    Parameters:
        data: pandas DataFrame containing the data.
        analyses: List of dictionaries, each specifying:
            - x_col: Column for the x-axis.
            - y_col: Column for the y-axis.
            - filtering_condition (optional): Filtering condition for the data.
            - x_log (optional): Whether to use a log scale for the x-axis.
            - y_log (optional): Whether to use a log scale for the y-axis.
            - hue_col (optional): Column for hue in scatterplots.
        grid_size (optional): Tuple specifying (rows, cols) for the subplot grid. Automatically calculated if not provided.
        figsize: Tuple specifying the overall figure size.
    """
    # Determine grid size automatically if not specified
    num_analyses = len(analyses)
    if grid_size is None:
        cols = math.ceil(math.sqrt(num_analyses))
        rows = math.ceil(num_analyses / cols)
    else:
        rows, cols = grid_size

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten in case of a grid

    for i, analysis in enumerate(analyses):
        if i >= len(axes):  # Avoid index out of range in case of excess analyses
            break

        # Extract parameters for the analysis
        x_col = analysis["x_col"]
        y_col = analysis["y_col"]
        filtering_condition = analysis.get("filtering_condition", None)
        x_log = analysis.get("x_log", False)
        y_log = analysis.get("y_log", False)
        hue_col = analysis.get("hue_col", None)

        # Apply filtering if specified
        filtered_data = data
        if filtering_condition is not None:
            filtered_data = filtered_data[filtering_condition]

        # Calculate correlation
        if not filtered_data.empty:
            corr = filtered_data[x_col].corr(filtered_data[y_col])
        else:
            corr = float("nan")
            print(f"No data left after filtering for analysis {i+1}.")

        # Plot scatter and regression
        sns.scatterplot(
            data=filtered_data, x=x_col, y=y_col, hue=hue_col, alpha=0.6, ax=axes[i]
        )
        sns.regplot(
            data=filtered_data,
            x=x_col,
            y=y_col,
            scatter=False,
            ax=axes[i],
            color="blue",
        )

        # Log scale adjustments
        if x_log:
            axes[i].set_xscale("log")
        if y_log:
            axes[i].set_yscale("log")

        # Titles and labels
        axes[i].set_title(f"{y_col} vs {x_col}\nCorrelation: {corr:.2f}")
        axes[i].set_xlabel(x_col + " (log scale)" if x_log else x_col)
        axes[i].set_ylabel(y_col + " (log scale)" if y_log else y_col)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.show()


        
        
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


def load_credentials(yaml_file = "../credentials.yaml"):
    with open(yaml_file) as file:
        credentials = yaml.safe_load(file)
    return credentials['username'], credentials['password'], credentials['url'], credentials['port']

def read_sql_ignoring_warnings(query, con, *args, **kwargs):
    """Wrapper for pandas.read_sql that suppresses UserWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_sql(query, con, *args, **kwargs)
    

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
    logging.getLogger('matplotlib').disabled = True
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


def analyze_correlation(features, feature_counts, title, threshold=0.8):
    """
    Analyze feature correlations, plot a heatmap, and identify highly correlated feature pairs.

    Parameters:
        features (list): List of feature names to analyze.
        feature_counts (pd.DataFrame): DataFrame containing feature counts.
        title (str): Title for the heatmap.
        threshold (float): Correlation threshold to identify highly correlated pairs.
    """
    # Subset the DataFrame for the selected features
    revised_feature_counts = feature_counts[features]

    # Calculate the correlation matrix
    correlation_matrix = revised_feature_counts.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title(title)
    plt.show()

    # Identify highly correlated feature pairs
    highly_correlated_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):  # Only look at one side of the matrix
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                highly_correlated_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))

    # Display highly correlated pairs
    print(f"Highly Correlated Feature Pairs for {title}:")
    for pair in highly_correlated_pairs:
        print(f"{pair[0]} and {pair[1]} have a correlation of {pair[2]:.2f}")

    return highly_correlated_pairs


def train_and_visualize_feature_importance(X, y, target_name, top_n=20):
    """
    Train a Random Forest Regressor and visualize feature importances.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        target_name (str): Name of the target variable for titles.
        top_n (int): Number of top features to visualize.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(top_n), hue = 'Feature')
    plt.title(f'Top {top_n} Feature Importances for {target_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

    return feature_importances


def plot_histograms_and_boxplots(data_list, titles, xlabels, hist_color='skyblue', box_color='skyblue', figsize=(15, 5)):
    """
    Plot histograms and boxplots for multiple datasets with statistical annotations.
    
    Parameters:
        data_list (list of pd.Series): List of data arrays or pandas Series to plot.
        titles (list of str): Titles for each plot.
        xlabels (list of str): Labels for the x-axis for each plot.
        hist_color (str, optional): Color for histogram bars. Default is 'skyblue'.
        box_color (str, optional): Color for boxplot boxes. Default is 'skyblue'.
        figsize (tuple, optional): Size of the figure. Default is (15, 5).
    """
    if len(data_list) != len(titles) or len(data_list) != len(xlabels):
        raise ValueError("Length of data_list, titles, and xlabels must match.")

    # Prepare data for histograms
    fig, axs = plt.subplots(1, len(data_list), figsize=figsize)
    
    for i, data in enumerate(data_list):
        axs[i].hist(data, bins=50, color=hist_color, edgecolor='black')

        # calculate skewness and kurtosis and annotate in the top-right corner
        skewness = data.skew()
        kurtosis = data.kurtosis()
        stats_text = f"Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}"
        axs[i].text(
            0.95, 0.95, stats_text,
            transform=axs[i].transAxes,
            fontsize=9, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

        axs[i].set_yscale('log')
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_ylabel('Frequency (log scale)')
        axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()

    # Prepare data for boxplots
    fig, axs = plt.subplots(1, len(data_list), figsize=figsize)

    for i, data in enumerate(data_list):
        # Plot boxplot
        box = axs[i].boxplot(
            data, vert=False, patch_artist=True,
            boxprops=dict(facecolor=box_color, color='black'),
            flierprops=dict(marker='o', color='#6d6875', markersize=5)
        )
        axs[i].set_xscale("log")

        # Calculate statistics
        data_clean = data.dropna() if isinstance(data, pd.Series) else data[~np.isnan(data)]
        q1, median, q3 = np.percentile(data_clean, [25, 50, 75])
        min_val = np.min(data_clean)
        max_val = np.max(data_clean)
        
        # Annotate statistics in the top-right corner
        stats_text = (f"Min: {min_val:.2f}\n"
                      f"Q1: {q1:.2f}\n"
                      f"Median: {median:.2f}\n"
                      f"Q3: {q3:.2f}\n"
                      f"Max: {max_val:.2f}")
        axs[i].text(
            0.95, 0.95, stats_text, transform=axs[i].transAxes,
            fontsize=9, ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

        # Customize appearance
        axs[i].set_xlabel(xlabels[i])
        axs[i].set_title(f"Boxplot of {xlabels[i]}")

    plt.tight_layout()
    plt.show()


def merge_dataframes(oa_features, ns_sec, population_density):
    """Merge feature, socio-economic, and population density data."""
    merged_df = pd.merge(
        oa_features,
        ns_sec[['geography_code', 'L15_proportion']],
        left_on="OA21CD",
        right_on="geography_code",
        how="left"
    ).drop(columns=['FID', 'GlobalID', 'geom', 'geometry_wkt', 'geography_code'])

    merged_df = pd.merge(
        merged_df,
        population_density[['geography_code', 'population_density_per_sq_km']],
        left_on="OA21CD",
        right_on="geography_code",
        how="left"
    ).drop(columns=['geography_code'])

    return merged_df


def compute_pairwise_correlations(df, feature_columns, target_column):
    """Compute pairwise correlations of features with a target column."""
    correlations = df[feature_columns + [target_column]].corr()[target_column]
    return correlations.sort_values(ascending=False)


def visualize_feature_correlations(correlations, title, color):
    """Visualize the correlation of features with a target variable."""
    plt.figure(figsize=(13, 7))
    correlations.drop(correlations.index[0]).plot(kind="bar", color=color)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Correlation")
    plt.show()


def plot_feature_target_pairplots(df, features, target, title, composite_features = None, ncols=3, log_y=False):
    """
    Create subplots with pairplots of each feature against the target variable.

    Parameters:
        df (pd.DataFrame): The DataFrame containing features and target variable.
        features (list): List of feature columns to plot.
        target (str): Target column name.
        title (str): Title of the plot grid.
        ncols (int): Number of columns in the grid of subplots.
    """
    if composite_features is None:
        composite_features = {}
    n_features = len(features) + len(composite_features)
    nrows = (n_features + ncols - 1) // ncols  # Calculate rows to fit all features

    df_tmp = df.copy()
    for composite_feature in composite_features.keys():
        df_tmp[composite_feature] = df_tmp[composite_features[composite_feature]].sum(axis=1)

    features += list(composite_features.keys())

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()  # Flatten axes for easy indexing

    for idx, feature in enumerate(features):
        df_tmp_1 = df_tmp[df_tmp[feature] > 0]
        # remove the top 1% of the feature values for better visualization
        if df_tmp_1[feature].dtype == 'float64':
            df_tmp_1 = df_tmp_1[df_tmp_1[feature] < df_tmp_1[feature].quantile(0.99)]
        sns.scatterplot(x=df_tmp_1[feature], y=df_tmp_1[target], ax=axes[idx], alpha=0.6)
        axes[idx].set_title(f"{feature} vs {target}")
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel(target)
        if log_y:
            axes[idx].set_yscale('log')

    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout and title
    fig.suptitle(title, fontsize=16, y=1.003)
    fig.tight_layout()
    plt.show()


def compute_correlations(df, feature_groups, target_column, area_column, residents_column="total_residents"):

    df_clean = df.loc[(df[feature_groups].sum(axis=1) > 0) & (df[target_column] > 0)]
    correlations = {}
    for group in feature_groups:
        feature_norm = df_clean[group] / df_clean[feature_groups].sum(axis=1)
        feature_density = df_clean[group] / df_clean[area_column]
        feature_population = df_clean[group] / df_clean[residents_column]

        correlation_norm = feature_norm.corr(df_clean[target_column])
        correlation_density = feature_density.corr(df_clean[target_column])
        correlation_population = feature_population.corr(df_clean[target_column])
        correlations[f"{group}_norm"] = correlation_norm
        correlations[f"{group}_density"] = correlation_density
        correlations[f"{group}_population"] = correlation_population
    return pd.Series(correlations).sort_values(ascending=False)


def plot_top_feature_correlations(correlations_dict, levels, title_suffix, ylabel="Correlation", top_n=5):
    """
    Plots the top feature correlations for multiple levels as subplots.

    Parameters:
    - correlations_dict (dict): A dictionary with level names as keys and correlation Series as values.
    - levels (list): List of levels to include in the plot, e.g., ['oa', 'lsoa', 'city'].
    - title_suffix (str): Suffix for subplot titles, e.g., "with L15 Proportion".
    - ylabel (str): Label for the y-axis. Default is "Correlation".
    - top_n (int): Number of top correlations to plot for each level. Default is 5.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt

    # Set up subplots
    fig, axs = plt.subplots(1, len(levels), figsize=(6 * len(levels), 6), sharey=True)

    for i, level in enumerate(levels):
        # Extract correlations for the level
        correlations = correlations_dict.get(level)
        if correlations is not None:
            # Plot the top N correlations
            correlations.head(top_n).plot(kind='bar', ax=axs[i], color='skyblue', edgecolor='black')
            axs[i].set_title(f"Top {top_n} Feature Correlations {title_suffix} ({level.upper()} Level)")
            axs[i].set_xlabel('Feature Group')
            axs[i].set_ylabel(ylabel if i == 0 else "")
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right')

            # Add grid
            axs[i].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
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


def plot_correlation_improvement(
    original_correlations, transformed_correlations, top_n=5, title="", ylabel="Correlation Improvement"
):
    """
    Plot comparison of features with the largest improvement in correlation after transformation.

    Parameters:
    - original_correlations (pd.Series): Correlations before transformation.
    - transformed_correlations (pd.Series): Correlations after transformation.
    - top_n (int): Number of features with the largest improvement to consider.
    - title (str): Title of the plot.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None (Displays the plot).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate the improvement in correlation
    correlation_improvement = transformed_correlations - original_correlations

    # Find the top N features with the largest improvement
    top_features = correlation_improvement.abs().nlargest(top_n).index

    # Extract relevant data
    comparison_df = pd.DataFrame({
        "Before Transformation": original_correlations.loc[top_features],
        "After Transformation": transformed_correlations.loc[top_features],
        "Improvement": correlation_improvement.loc[top_features]
    }).sort_values("Improvement", ascending=False)

    # Plot the comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar width for grouping
    bar_width = 0.4
    x = np.arange(len(comparison_df))

    # Plot bars
    ax.bar(x - bar_width / 2, comparison_df["Before Transformation"], bar_width, label="Before Transformation", color="skyblue", edgecolor="black")
    ax.bar(x + bar_width / 2, comparison_df["After Transformation"], bar_width, label="After Transformation", color="orange", edgecolor="black")

    # Add labels and title
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha="right")
    ax.legend()

    # Add grid
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()