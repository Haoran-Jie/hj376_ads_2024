import os
import requests
import pandas as pd
import geopandas as gpd
from .file_io import download_file

def download_census_data(code, base_dir=''):
    import zipfile
    import io
    url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
    extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Files already exist at: {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")

def load_census_data(code, level='msoa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

def download_and_save_census_data(category, level, output_filename):
    """
    Download and save census data for a given category and level.
    
    Args:
        category (str): Census data category (e.g., 'TS062').
        level (str): Geography level (e.g., 'oa').
        output_filename (str): Name of the file to save the data.
    """
    print(f"Downloading {category} Census Data...")
    download_census_data(category)
    census_data = load_census_data(category, level)
    census_data.to_csv(output_filename, index=False)
    print(f"Saved {category} data to {output_filename}")


def download_and_process_geojson(url, output_geojson_file, output_csv_file, epsg_code=27700):
    """
    Download and process a GeoJSON file to calculate area and save it as a CSV.

    Args:
        url (str): URL of the GeoJSON file.
        output_geojson_file (str): Path to save the downloaded GeoJSON file.
        output_csv_file (str): Path to save the processed CSV file.
        epsg_code (int): EPSG code for area calculation (default: 27700).
    """
    print(f"Downloading GeoJSON file from {url}...")
    download_file(url, output_geojson_file)

    if not os.path.exists(output_csv_file):
        print(f"Processing GeoJSON file: {output_geojson_file}")
        geo_data = gpd.read_file(output_geojson_file)
        geo_data["geometry_wkt"] = geo_data["geometry"].apply(lambda x: x.wkt)

        # Convert CRS and calculate area
        geo_data.to_crs(epsg=epsg_code, inplace=True)
        geo_data['area'] = geo_data['geometry'].area
        geo_data = geo_data.drop(columns=["geometry"])
        
        # Convert back to geographic CRS for saving
        geo_data.to_crs(epsg=4326, inplace=True)
        geo_data.to_csv(output_csv_file, index=False)
        print(f"Processed and saved to {output_csv_file}")
    else:
        print(f"Processed GeoJSON already exists: {output_csv_file}")


def download_2011_census_data(category, basedir="./"):
    """
    Download 2011 Census Data for a specific category and OA geography.
    
    Args:
    category (str): The category of census data to download.
    oa_geography (str): The OA geography code.
    
    Returns:
    None
    """
    oa_geography = {
        'East': '2013265926TYPE299',
        'East_Midlands': '2013265924TYPE299',
        'London': '2013265927TYPE299',
        'North_East': '2013265921TYPE299',
        'North_West': '2013265922TYPE299',
        'South_East': '2013265928TYPE299',
        'South_West': '2013265929TYPE299',
        'Wales': '2013265930TYPE299',
        'West_Midlands': '2013265925TYPE299',
        'Yorkshire_Humber': '2013265923TYPE299'
    }
    nm_codes = {
        'age_distribution': 'NM_145_1', 
        'ethnicity': 'NM_608_1',
        'qualification': 'NM_623_1',
        'household_composition': 'NM_605_1',
        'economic_activity': 'NM_624_1',
        'deprivation': 'NM_519_1'
    }
    for region, oa_code in oa_geography.items():
        url = f"https://www.nomisweb.co.uk/api/v01/dataset/{nm_codes[category]}.bulk.csv?time=latest&measures=20100&rural_urban=total&geography={oa_code}"
        download_file(url, f"{basedir}{category}_{region}.csv")