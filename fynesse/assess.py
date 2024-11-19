from .config import *

from . import access
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from .access import create_connection, fetch_houses_within_box

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


def calculate_and_visualise_correlation(final_matched_df, x_col, y_col, x_log=False, y_log=False, hue_col=None, filtering_condition=None):
    """Calculate and visualise the correlation between two columns in the DataFrame."""

    corr = final_matched_df[x_col].corr(final_matched_df[y_col])
    print(f"Correlation between {x_col} and {y_col}: {corr:.2f}")

    if filtering_condition is not None:
        final_matched_df = final_matched_df[filtering_condition]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_matched_df, x=x_col, y=y_col, hue=hue_col, alpha=0.6)
    sns.regplot(data=final_matched_df, x=x_col, y=y_col, scatter=False)
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.title(f"Scatter Plot of {y_col} vs. {x_col} (Log Scale)\nCorrelation: {corr:.2f}")
    plt.xlabel(x_col + " (log scale)" if x_log else x_col)
    plt.ylabel(y_col + " (log scale)" if y_log else y_col)
    if hue_col is not None:
        plt.legend(title=hue_col)
    plt.show()


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