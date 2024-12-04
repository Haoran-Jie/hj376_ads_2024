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
from scipy.stats import skew, kurtosis, probplot
import scipy.stats as stats
from scipy.stats import linregress
from scipy.interpolate import interp1d, CubicSpline
from .constants import *

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


def plot_turnout_distribution(
    election_results,
    turnout_column="Turnout_rate",
    party_column="First_party",
    party_colors=None,
    bins=np.arange(40, 76, 1),
    title="Distribution of Turnout Rate by Winning Party",
    figsize=(12, 8)
):
    """
    Plot a stacked histogram with KDE and statistics for election turnout rates.
    
    Args:
        election_results (pd.DataFrame): DataFrame containing election data.
        turnout_column (str): Column name for turnout rates.
        party_column (str): Column name for party affiliation.
        party_colors (dict): Dictionary mapping party names to colors.
        bins (np.array): Array of bin edges for the histogram.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure.
    
    Returns:
        None
    """
    if party_colors is None:
        party_colors = {"Con": "#2475C0", "Lab": "#DA302F", "LD": "#F7A938", "Other": "#8E8D8D"}

    # Categorize winning party
    election_results["Winning_party"] = election_results[party_column].apply(
        lambda x: "Other" if x not in ["Con", "Lab", "LD"] else x
    )
    
    # Define party categories
    party_categories = ["Lab", "Con", "LD", "Other"]
    stacked_data = [
        election_results.loc[election_results["Winning_party"] == party, turnout_column]
        for party in party_categories
    ]

    # Calculate statistics
    mean = election_results[turnout_column].mean()
    std = election_results[turnout_column].std()
    skewness = skew(election_results[turnout_column])
    kurt = kurtosis(election_results[turnout_column])

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot the stacked histogram
    counts, _, _ = ax1.hist(
        stacked_data,
        bins=bins,
        stacked=True,
        color=[party_colors[party] for party in party_categories],
        edgecolor="black",
        alpha=0.8,
        label=party_categories,
    )

    ax1.set_xlabel("Turnout Rate (%)", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    ax1.tick_params(axis="y")

    # Add KDE plot on secondary axis
    ax2 = ax1.twinx()
    sns.kdeplot(
        election_results[turnout_column],
        color="black",
        linewidth=2,
        label="KDE",
        ax=ax2
    )

    ax2.set_ylabel("Density", fontsize=14, color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Add title
    plt.title(title, fontsize=16, fontweight="bold")

    # Annotate statistics
    stats_text = f"Mean: {mean:.2f}%\nStd Dev: {std:.2f}%\nSkewness: {skewness:.2f}\nKurtosis: {kurt:.2f}"
    ax1.annotate(stats_text, xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
                 fontsize=12)

    # Add legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, title="Legend", fontsize=12, title_fontsize=14)

    # Add grid and layout adjustments
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_qq(data, title="QQ Plot", xlabel="Theoretical Quantiles", ylabel="Ordered Values", 
            dot_color="#2475C0", line_color="#DA302F"):
    """
    Creates a QQ plot for the given data with improved formatting and customization options.
    
    Parameters:
    - data (array-like): The data to be plotted.
    - title (str): The title of the plot.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - dot_color (str): Color of the dots in the QQ plot.
    - line_color (str): Color of the diagonal reference line in the QQ plot.

    Returns:
    - None: Displays the QQ plot.
    """
    # Create the QQ plot
    plt.figure(figsize=(10, 6))
    qq = probplot(data, dist="norm")
    theoretical_quantiles, ordered_values = qq[0]
    slope, intercept, _ = qq[1]

    # Scatter plot for the data points
    plt.scatter(theoretical_quantiles, ordered_values, color=dot_color, label="Data Points", alpha=0.9)

    # Plot the diagonal line
    plt.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color=line_color, 
             label="Reference Line", linewidth=2)

    # Customize the title and labels
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Add a grid for better readability
    plt.grid(linestyle='--', alpha=0.6)

    # Add a legend
    plt.legend(fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_turnout_and_electorate(election_data, title="Turnout and Total Electorate at General Elections", 
                                turnout_color="#e07a5f", electorate_color="#3d405b", background_color="#f4f1de"):
    """
    Plots turnout rates as a bar chart and total electorate as a line chart with dual y-axes.

    Parameters:
    - election_data (pd.DataFrame): Data containing 'election', 'turnout_rate', and 'total_electorate' columns.
    - title (str): Title of the plot.
    - turnout_color (str): Color of the turnout bar chart.
    - electorate_color (str): Color of the electorate line chart.
    - background_color (str): Background color of the chart.

    Returns:
    - None: Displays the plot.
    """
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Set background color
    ax1.set_facecolor(background_color)

    # Plot the turnout rate as a bar chart
    ax1.bar(election_data['election'], election_data['turnout_rate'], color=turnout_color, label='Turnout Rate (%)')
    ax1.set_title(title, fontsize=16, pad=15)
    ax1.set_xlabel('Election Year', fontsize=12)
    ax1.set_ylabel('Turnout Rate (%)', fontsize=12, color=turnout_color)
    ax1.tick_params(axis='y', labelcolor=turnout_color)

    # Add grid lines
    ax1.grid(axis='y', linestyle='--', color='grey', alpha=0.7)

    # Adjust x-axis ticks
    ax1.set_xticks(range(len(election_data['election'])))
    ax1.set_xticklabels(election_data['election'], rotation=45, fontsize=10)

    # Add a secondary y-axis for total electorate
    ax2 = ax1.twinx()
    ax2.plot(election_data['election'], election_data['total_electorate'], color=electorate_color, 
             marker='o', label='Total Electorate')
    ax2.set_ylabel('Total Electorate', fontsize=12, color=electorate_color)
    ax2.tick_params(axis='y', labelcolor=electorate_color)

    # Adding a legend
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, fontsize=10)

    # Adding a note
    plt.figtext(0.5, -0.03, "Note: 1974F and 1974O are the two general elections held February and October in 1974", 
                ha='center', fontsize=10, wrap=True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



def plot_contention_distribution(data, contention_column, majority_column, valid_votes_column, 
                                  bins=20, kde=True, color="skyblue", 
                                  percentiles=[0.33, 0.67], 
                                  title="Distribution of Contentiousness", 
                                  xlabel="Contentiousness", ylabel="Frequency"):
    """
    Plots the distribution of contentiousness with KDE and highlights specified percentiles.

    Parameters:
    - data (pd.DataFrame): The dataset containing election results.
    - contention_column (str): Column name to store the calculated contentiousness.
    - majority_column (str): Column name for majority votes.
    - valid_votes_column (str): Column name for valid votes.
    - bins (int): Number of bins for the histogram.
    - kde (bool): Whether to include a KDE plot.
    - color (str): Color of the histogram.
    - percentiles (list of floats): Percentiles to highlight (e.g., [0.33, 0.67]).
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None: Displays the plot.
    """
    # Calculate contentiousness
    data[contention_column] = 1 - data[majority_column] / data[valid_votes_column]

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data[contention_column],
        bins=bins,
        kde=kde,
        color=color,
        edgecolor="black",
        alpha=0.8
    )

    # Calculate and add percentile lines
    percentile_values = data[contention_column].quantile(percentiles)
    for i, perc in enumerate(percentiles):
        color_line = "red" if i == 0 else "green"
        plt.axvline(percentile_values.iloc[i], color=color_line, linestyle="--", 
                    label=f"{int(perc * 100)}th percentile")

    # Title and labels
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()



def analyze_turnout_by_contention(election_data, contention_column, turnout_column, 
                                  constituency_column, percentiles, 
                                  bins=[0, 0.33, 0.67, 1], 
                                  bin_labels=['Low', 'Medium', 'High'], 
                                  palette="viridis", figsize=(8, 6)):
    """
    Analyze turnout rates by contentiousness levels and perform One-Way ANOVA.

    Parameters:
    - election_data (pd.DataFrame): Dataset containing election results.
    - contention_column (str): Column name for contentiousness data.
    - turnout_column (str): Column name for turnout rates.
    - constituency_column (str): Column name for constituency names.
    - percentiles (pd.Series or list): Percentiles for binning contentiousness.
    - bins (list): Bin edges for grouping contentiousness levels.
    - bin_labels (list): Labels for each contentiousness group.
    - palette (str): Seaborn palette for boxplot.
    - figsize (tuple): Size of the figure.

    Returns:
    - None: Displays the boxplot and prints ANOVA results.
    """
    # Create a copy of relevant data
    contention_data = election_data[[constituency_column, contention_column, turnout_column]].copy()

    # Categorize contentiousness into bins
    contention_data['group'] = pd.cut(contention_data[contention_column], bins=bins, labels=bin_labels)
    
    anova_result = stats.f_oneway(
        contention_data.loc[contention_data['group'] == 'High', turnout_column],
        contention_data.loc[contention_data['group'] == 'Medium', turnout_column],
        contention_data.loc[contention_data['group'] == 'Low', turnout_column]
    )
    print('ANOVA p-value:', anova_result.pvalue)
    if anova_result.pvalue < 0.05:
        print('The differences in Turnout Rate between groups are statistically significant.')
    else:
        print('The differences in Turnout Rate between groups are not statistically significant.')
    
    # Plot the boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(
        x='group',
        y=turnout_column,
        data=contention_data,
        hue='group',
        palette=palette,
    )

    # annotate the anova p-value on the plot, as a bbox on the top right corner
    plt.figtext(0.96, 0.9, f"ANOVA p-value: {anova_result.pvalue:.2e}", ha='right', va='top', 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    plt.title('Turnout Rates by Contentiousness Group', fontsize=16, fontweight='bold')
    plt.xlabel('Contentiousness Group', fontsize=14)
    plt.ylabel('Turnout Rate (%)', fontsize=14)
    plt.tight_layout()
    plt.show()


def analyze_turnout_by_party(election_data, party_column, turnout_column, party_order=None, palette="viridis", figsize=(10, 6)):
    """
    Analyze turnout rates by winning party and perform One-Way ANOVA.

    Parameters:
    - election_data (pd.DataFrame): Dataset containing election results.
    - party_column (str): Column name for winning party data.
    - turnout_column (str): Column name for turnout rates.
    - party_order (list): Order of parties for the boxplot.
    - palette (str): Seaborn palette for boxplot.
    - figsize (tuple): Size of the figure.

    Returns:
    - None: Displays the boxplot and prints ANOVA results.
    """
    # Categorize "Winning_party" as specified
    election_data["Winning_party"] = election_data[party_column].apply(
        lambda x: "Other" if x not in ["Con", "Lab", "LD"] else x
    )
    
    # Determine party groups
    parties = election_data["Winning_party"].unique()
    if party_order is None:
        party_order = ["Con", "Lab", "LD", "Other"]
    
    # Perform ANOVA
    turnout_by_party = [
        election_data.loc[election_data["Winning_party"] == party, turnout_column]
        for party in party_order if party in parties
    ]
    anova_result = stats.f_oneway(*turnout_by_party)
    print("ANOVA p-value:", anova_result.pvalue)
    if anova_result.pvalue < 0.05:
        print("The differences in Turnout Rate between parties are statistically significant.")
    else:
        print("The differences in Turnout Rate between parties are not statistically significant.")
    
    # Plot the boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(
        x="Winning_party",
        y=turnout_column,
        data=election_data,
        order=party_order,
        hue = "Winning_party",
        palette=palette
    )

    # Annotate the ANOVA p-value on the plot
    plt.figtext(
        0.96, 0.9, f"ANOVA p-value: {anova_result.pvalue:.2e}", ha="right", va="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=1")
    )
    
    # Add title and labels
    plt.title("Turnout Rate by Winning Party", fontsize=16, fontweight="bold")
    plt.xlabel("Winning Party", fontsize=14)
    plt.ylabel("Turnout Rate (%)", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()




def test_linearity(df, col_prefix, years=[2001, 2011, 2021], threshold=0.7):
    """
    Test the linearity of a variable across multiple years.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col_prefix (str): The column prefix to test for linearity.
        years (list): A list of years to test for linearity.
        threshold (float): The R-squared threshold for linearity.

    Returns:
        str: Either 'linear' or 'cubic' depending on the overall linearity of the column.
    """
    linear_count = 0
    total_count = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Testing Linearity for {col_prefix}"):
        y = [row[f"{col_prefix}_{year}"] for year in years]
        x = years
        # Fit a linear regression
        _, _, r_value, _, _ = linregress(x, y)
        if abs(r_value) >= threshold:
            linear_count += 1
        total_count += 1
    # Decide method based on proportion of linear trends
    return 'linear' if linear_count / total_count > 0.7 else 'cubic'


def interpolate_data(df, columns, years=[2001, 2011, 2021], target_years=[2010, 2015, 2017, 2019, 2024], methods=None):
    """
    Interpolate data for multiple target years using the specified interpolation methods.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        columns (list): List of column prefixes to interpolate (e.g., ['white_british', 'asian_total']).
        years (list): The years corresponding to the data columns (default: [2001, 2011, 2021]).
        target_years (list): List of years to interpolate (e.g., [2010, 2015, 2017, 2019]).
        methods (dict): Dictionary mapping column prefixes to interpolation methods ('linear' or 'cubic').

    Returns:
        pd.DataFrame: The original DataFrame with new columns containing interpolated values for each target year.
    """
    if methods is None:
        raise ValueError("Methods dictionary must be provided for interpolation.")

    interpolated_results = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Interpolating Data"):
        row_result = {'OA11CD': row['OA11CD']}  # Identifier for each OA

        for col_prefix in columns:
            # Extract the values for interpolation
            data_values = [row[f"{col_prefix}_{year}"] for year in years]

            # Check for missing values
            if any(pd.isnull(data_values)):
                for target_year in target_years:
                    row_result[f"{col_prefix}_{target_year}"] = np.nan  # Cannot interpolate with missing data
            else:
                # Determine interpolation method
                method = methods.get(col_prefix, 'linear')  # Default to 'linear' if not specified

                # Perform interpolation
                if method == 'linear':
                    interpolator = interp1d(years, data_values, kind='linear', fill_value="extrapolate")
                elif method == 'cubic':
                    interpolator = CubicSpline(years, data_values)
                else:
                    raise ValueError(f"Invalid method for {col_prefix}. Use 'linear' or 'cubic'.")

                # Interpolate for all target years
                for target_year in target_years:
                    row_result[f"{col_prefix}_{target_year}"] = interpolator(target_year)

        interpolated_results.append(row_result)

    # Convert results to a DataFrame
    interpolated_df = pd.DataFrame(interpolated_results)

    # Merge the interpolated values back to the original DataFrame
    result_df = pd.merge(df, interpolated_df, on='OA11CD')
    return result_df


def process_interpolation_oa(df_direct, df_approx, columns, years=[2001, 2011, 2021], target_years=[2010, 2015, 2017, 2019, 2024], threshold=0.7):
    """
    Full pipeline to determine interpolation methods using df_direct and perform interpolation on df_approx.

    Parameters:
        df_direct (pd.DataFrame): DataFrame for direct matching (unchanged OAs).
        df_approx (pd.DataFrame): DataFrame for approximated matching (aggregated OAs).
        columns (list): List of column prefixes to process (e.g., ['white_british', 'asian_total']).
        years (list): The years corresponding to the data columns (default: [2001, 2011, 2021]).
        target_years (list): List of years to interpolate (e.g., [2010, 2015, 2017, 2019]).
        threshold (float): The R-squared threshold for linearity.

    Returns:
        pd.DataFrame: The DataFrame with interpolated values for target years.
    """
    methods = {}

    # Step 1: Determine interpolation methods using df_direct
    for col_prefix in tqdm(columns, desc="Determining Interpolation Methods"):
        methods[col_prefix] = test_linearity(df_direct, col_prefix, years=years, threshold=threshold)
        print(f"{col_prefix}: {methods[col_prefix]} interpolation chosen")

    # Step 2: Perform interpolation using df_approx
    interpolated_df = interpolate_data(df_approx, columns, years=years, target_years=target_years, methods=methods)

    return interpolated_df

def plot_boxplot(data, x_col, y_col, title, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=x_col,
        y=y_col,
        data=data,
        hue = x_col,
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


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


def merge_total_population(interpolated_df, total_df, total_col_name):
    """
    Merge total population data with the interpolated DataFrame.

    Parameters:
        interpolated_df (pd.DataFrame): Interpolated data.
        total_df (pd.DataFrame): DataFrame containing total population values.
        total_col_name (str): Column name for the total population.

    Returns:
        pd.DataFrame: Updated DataFrame with total population merged.
    """
    return interpolated_df.merge(
        total_df[['geography_code', total_col_name]], 
        left_on='OA11CD', 
        right_on='geography_code', 
        how='left'
    )

def calculate_counts(interpolated_df, columns, years, total_col_name):
    """
    Calculate counts from proportions using total population.

    Parameters:
        interpolated_df (pd.DataFrame): Interpolated data.
        columns (list): Column prefixes to process.
        years (list): Target years.
        total_col_name (str): Column name for the total population.

    Returns:
        pd.DataFrame: Updated DataFrame with counts calculated.
    """
    for col in columns:
        for year in years:
            interpolated_df[f"{col}_{year}_count"] = (
                interpolated_df[f"{col}_{year}"] * interpolated_df[total_col_name]
            )
    return interpolated_df

def aggregate_to_pcon(interpolated_df, columns, years, total_col_name, pcon_map, pcon_col='PCON11CD'):
    """
    Aggregate counts and compute proportions by parliamentary constituency.

    Parameters:
        interpolated_df (pd.DataFrame): Data with counts calculated.
        columns (list): Column prefixes to process.
        years (list): Target years.
        total_col_name (str): Column name for the total population.
        pcon_map (pd.DataFrame): Mapping of OA to PCON.

    Returns:
        pd.DataFrame: Aggregated data by PCON.
    """
    # Merge OA to PCON mapping
    interpolated_df = interpolated_df.merge(pcon_map, on='OA11CD', how='left')

    # Group by PCON and aggregate counts
    grouped_df = interpolated_df.groupby(pcon_col).agg(
        {
            total_col_name: 'sum',
            **{f"{col}_{year}_count": 'sum' for col in columns for year in years}
        }
    ).reset_index()

    # Compute proportions
    for col in columns:
        for year in years:
            grouped_df[f"frac_{col}_{year}"] = (
                grouped_df[f"{col}_{year}_count"] / grouped_df[total_col_name]
            )
    return grouped_df

def process_pipeline_pcon(result_dfs, total_dfs, pcon_map, config, years=[2010, 2015, 2017, 2019], pcon_col='PCON11CD'):
    """
    Process all pipelines (ethnic group, household composition, deprivation, qualification).

    Parameters:
        result_dfs (dict): Dictionary of interpolated result DataFrames for each pipeline.
        total_dfs (dict): Dictionary of total population DataFrames for each pipeline.
        pcon_map (pd.DataFrame): Mapping of OA11CD to PCON11CD.
        config (dict): Configuration dictionary with column prefixes and total column names.
        years (list): Target years for interpolation.

    Returns:
        dict: Dictionary of processed DataFrames for each pipeline aggregated by PCON.
    """
    final_results = {}

    for pipeline, params in config.items():
        print(f"Processing {pipeline} pipeline...")

        # Step 1: Copy interpolated data
        interpolated_df = result_dfs[pipeline].copy()

        # Step 2: Merge total population
        interpolated_df = merge_total_population(
            interpolated_df, 
            total_dfs[pipeline], 
            params['total_col']
        )

        # Step 3: Calculate counts
        interpolated_df = calculate_counts(
            interpolated_df, 
            params['columns'], 
            years, 
            params['total_col']
        )

        # Step 4: Aggregate to PCON and compute proportions
        final_results[pipeline] = aggregate_to_pcon(
            interpolated_df, 
            params['columns'], 
            years, 
            params['total_col'], 
            pcon_map,
            pcon_col=pcon_col
        )
    
    print("___ Processing complete. ___")
    
    return final_results

def get_merged_census_results(final_results):
    # Access results for individual pipelines
    ethnic_group_results = final_results['ethnic_group']
    household_results = final_results['household_composition']
    deprivation_results = final_results['deprivation']
    qualification_results = final_results['qualification']
    economic_activity_results = final_results['economic_activity']

    merged_census_results = pd.concat([ethnic_group_results, household_results, deprivation_results, qualification_results, economic_activity_results], axis=1)
    merged_census_results = remove_duplicate_columns(merged_census_results, "interpolated_census_result")
    return merged_census_results

def get_yearly_census_results(merged_census_results, election_results, years=[2010, 2015, 2017, 2019]):
    M = {}
    for year in years:
        election_results_history_now = election_results[election_results['election'] == year]
        election_results_history_merged = election_results_history_now.merge(merged_census_results, left_on='constituency_id', right_on='PCON11CD', how='inner')
        # sort by constituency_id
        election_results_history_merged = election_results_history_merged.sort_values('constituency_id')
        feature_columns = []
        for key, cols in COLUMN_PREFIXES.items():
            for col in cols:
                feature_columns.append(f"frac_{col}_{year}")
        M[year] = election_results_history_merged[feature_columns + ['turnout']]
    
    return M
            

def plot_turnout_comparison(regional_turnout_2024, regional_turnout_2019, xlim=(50, 75), dot_size=100):
    """
    Plot a comparison of turnout rates for 2019 and 2024 general elections by region.

    Parameters:
        regional_turnout_2024 (pd.DataFrame): DataFrame with regions as index and 2024 turnout rates.
        regional_turnout_2019 (pd.DataFrame): DataFrame with regions as index and 2019 turnout rates.
        xlim (tuple): Tuple specifying the limits of the x-axis (default: (50, 75)).
        dot_size (int): Size of the scatter plot dots (default: 100).
    """
    # Prepare data
    regions = regional_turnout_2024.index  # List of regions
    turnout_2024 = regional_turnout_2024['turnout_rate']
    turnout_2019 = regional_turnout_2019['turnout_rate']

    # Sorting data for consistent ordering
    sorted_indices = turnout_2024.argsort()
    regions = regions[sorted_indices]
    turnout_2024 = turnout_2024.iloc[sorted_indices]
    turnout_2019 = turnout_2019.iloc[sorted_indices]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bars connecting scatter points
    for i, region in enumerate(regions):
        # use dotted lines to connect the points
        ax.plot([turnout_2019.iloc[i], turnout_2024.iloc[i]], [region, region], color='#cdb4db', linewidth=1, linestyle='--')


    # Scatter points for 2019
    ax.scatter(turnout_2019, regions, color='#a2d2ff', label='2019 general election', zorder=3, s=dot_size)

    # Scatter points for 2024
    ax.scatter(turnout_2024, regions, color='#ffc8dd', label='2024 general election', zorder=3, s=dot_size)

    # Remove unwanted spines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add labels, title, and legend
    ax.set_xlabel("Turnout Rate (%)")
    ax.set_ylabel("Region")
    ax.set_title("Turnout by Region and Country (2019 vs 2024)")
    ax.legend()

    # Add grid and style adjustments
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(xlim)

    # set background color
    ax.set_facecolor('#f5ebe0')

    # Show plot
    plt.tight_layout()
    plt.show()