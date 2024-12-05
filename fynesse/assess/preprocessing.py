import pandas as pd
import numpy as np
from tabulate import tabulate
from ..access import fetch_houses_within_box, create_connection
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


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
    
    from ..access import fetch_building_within_bbox
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