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


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


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
        logger.info(f"{dataset_name}: Duplicate columns identified for removal: {duplicate_cols}")
        deduplicated_df = df.loc[:, ~df.columns.duplicated()]
        logger.info(f"{dataset_name}: Removed duplicate columns. New shape: {deduplicated_df.shape}")
        return deduplicated_df
    else:
        logger.info(f"{dataset_name}: No duplicate columns identified.")
        return df

def remove_empty_columns(df, dataset_name="DataFrame"):
    """Remove columns with all zero values."""
    empty_columns = df.columns[(df == 0).all()]
    if empty_columns.empty:
        logger.info(f"{dataset_name}: No empty columns identified.")
        return df, empty_columns
    logger.info(f"{dataset_name}: Empty columns identified for removal: {empty_columns.tolist()}")
    cleaned_df = df.drop(columns=empty_columns)
    logger.info(f"{dataset_name}: Shape after removing empty columns: {cleaned_df.shape}")
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
    logger.info(f"{dataset_name}: Rows with all zeros in feature columns: {len(rows_with_all_zeros)} out of {df.shape[0]}")
    return rows_with_all_zeros

def check_columns_types(df, columns, expected_types, dataset_name="DataFrame"):
    """Check the data types of columns in the DataFrame."""
    logger.info(f"{dataset_name}: Checking data types for {len(columns)} columns...")
    issues = []
    
    for column, expected_type in zip(columns, expected_types):
        actual_type = df[column].dtype
        if actual_type != expected_type:
            issues.append([column, str(expected_type), str(actual_type)])
    
    if issues:
        logger.warning(f"{dataset_name}: Found {len(issues)} type mismatches:")
        print(tabulate(issues, headers=["Column", "Expected Type", "Actual Type"], tablefmt="pretty"))
    else:
        logger.info(f"{dataset_name}: All columns have expected data types.")

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


        
        
def plot_geometry_with_buffer_and_features(geometry, ax=None, title=None):
    """
    Plots the bounding box of a geometry object with a 1km buffer, retrieves building and landuse features,
    and visualizes them with specified colors.
    
    Parameters:
        geometry (shapely.geometry.Polygon): The geometry object to process.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. Creates a new one if None.
        title (str, optional): Title for the plot.
    """
    # Calculate the bounding box of the geometry
    minx, miny, maxx, maxy = geometry.bounds
    
    # Add a 1km buffer to the bounding box using fynesse
    buffer_km = 1  # Buffer distance in km
    dy, dx = calculate_half_side_degrees(((miny + maxy) / 2, (minx + maxx) / 2), buffer_km)
    buffered_bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)
    
    # Extract the buffered bbox coordinates
    buffered_minx, buffered_miny, buffered_maxx, buffered_maxy = buffered_bbox.bounds

    # Get building and landuse features within the buffered bbox
    bbox = (buffered_minx, buffered_miny, buffered_maxx, buffered_maxy)
    buildings = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    landuse = ox.features_from_bbox(
        bbox=bbox,
        tags={"landuse": ["meadow", "farmland", "grass", "forest"]}
    )

    # Count the occurrences of each building type
    building_counts = buildings["building"].value_counts()

    # Identify the top 5 building types
    top_building_types = building_counts.head(5).index

    # Assign colors for the top 5 building types (avoiding green)
    building_colors = {}
    for i, btype in enumerate(top_building_types):
        building_colors[btype] = f"C{i + 1}"  # Skip C0 (green) to avoid conflict

    # Assign green color for landuse types
    landuse_color = "green"

    # Create the axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the buffered bbox
    bbox_graph = ox.graph_from_bbox(buffered_maxy, buffered_miny, buffered_maxx, buffered_minx, network_type="all")
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
            fill=False,  # Don't fill the polygon
            label="Polygon Boundary"
        )
        ax.add_patch(mpl_polygon)

    # Create legend elements manually
    legend_elements = []

    # Plot the buildings
    for btype in top_building_types:
        buildings[buildings["building"] == btype].plot(
            ax=ax, color=building_colors[btype], label=f"Building: {btype}", alpha=0.7
        )
        legend_elements.append(Patch(facecolor=building_colors[btype], edgecolor='black', label=btype))

    # Plot the remaining buildings in gray
    remaining_buildings = ~buildings["building"].isin(top_building_types)
    buildings[remaining_buildings].plot(ax=ax, color="lightgray", label="Other Buildings", alpha=0.5)
    legend_elements.append(Patch(facecolor="lightgray", edgecolor='black', label="Other Buildings"))

    # Plot the landuse features in green
    landuse.plot(ax=ax, color=landuse_color, label="Landuse (green areas)", alpha=0.5)
    legend_elements.append(Patch(facecolor=landuse_color, edgecolor='black', label="Landuse"))

    # Add custom legend
    ax.legend(handles=legend_elements, title="Legend")

    # Set labels and limits
    ax.set_xlim([buffered_minx, buffered_maxx])
    ax.set_ylim([buffered_miny, buffered_maxy])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Buildings and Landuse within Buffered BBox")