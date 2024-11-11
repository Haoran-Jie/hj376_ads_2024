from .config import *

import requests
import pymysql
import csv
import time
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import osmnx as ox

from .utility import calculate_half_side_degrees
from .assess import plot_geodata
"""
import httplib2
import oauth2
import tables
import mongodb
import sqlite
"""
# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
  print("Hello from the data science library!")

def download_price_paid_data(year_from: int, year_to: int):
    """
    Download UK house price data for given year range
    :param year_from: starting year
    :param year_to: ending year
    """
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user: str, password: str, host: str, database: str, port:int = 3306)-> pymysql.connections.Connection:
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn: pymysql.connections.Connection, year: int):
  """
  Upload the data to the database
  :param conn: Connection object
  :param year: year for which data is to be uploaded
  """

  csv_file_path = os.path.abspath(f'./output_file_{year}.csv')
  cur = conn.cursor()

  if not os.path.exists(csv_file_path):

    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"
    
    print('Selecting data for year: ' + str(year))
    cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)
      # Write the data rows
      csv_writer.writerows(rows)
  else:
    print('CSV file already exists')

  print('Storing data for year: ' + str(year))
  response = cur.execute(f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  print('Data stored for year: ' + str(year))
  cur.close()



def fetch_houses_within_box(location: tuple[float], side_length_km: float, conn: pymysql.connections.Connection, since_date: str = '2020-01-01', inflation_correction = True) -> pd.DataFrame:
    """
    Fetch houses data from the database within a square box centered around a location.

    Parameters:
    location (tuple): A tuple containing the latitude and longitude of the center of the box.
    side_length_km (float): The side length of the square box in kilometers.
    conn (pymysql.connections.Connection): A connection to the database.

    Returns:
    DataFrame: A Pandas DataFrame containing the fetched houses data.
    """
    latitude, longitude = location

    # Calculate the half side length in degrees
    half_side_length_lat, half_side_length_lon = calculate_half_side_degrees(location, side_length_km)

    sql_query = f"""
    SELECT pp.*
    FROM pp_data pp
    JOIN postcode_data pc ON pp.postcode = pc.postcode
    WHERE
        pc.latitude BETWEEN {latitude - half_side_length_lat} AND {latitude + half_side_length_lat}
        AND pc.longitude BETWEEN {longitude - half_side_length_lon} AND {longitude + half_side_length_lon}
        AND pp.date_of_transfer >= '{since_date}';
    """
    
    """
    UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy. 
    houses_df = pd.read_sql(sql_query, conn)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        houses_df = pd.read_sql(sql_query, conn)

    # inflation_correction
    """Year	 Inflation	Multiplier
        2024		    1.00
        2023	9.7%	1.10
        2022	11.6%	1.22
        2021	4.1%	1.27
        2020	1.5%	1.29
        2019	2.6%	1.33
        2018	3.3%	1.37
        2017	3.6%	1.42
        2016	1.8%	1.45
        2015	1.0%	1.46
        2014	2.4%	1.50
        2013	3.0%	1.54
        2012	3.2%	1.59
        2011	5.2%	1.67
        2010	4.6%	1.75
        2009	−0.5%	1.74
        2008	4.0%	1.81
        2007	4.3%	1.89
        2006	3.2%	1.95
        2005	2.8%	2.00
        2004	3.0%	2.06
        2003	2.9%	2.12
        2002	1.7%	2.16
        2001	1.8%	2.20
        2000	3.0%	2.26
        1999	1.5%	2.30
        1998	3.4%	2.38
        1997	3.1%	2.45
        1996	2.4%	2.51
        1995	3.5%	2.60
    """
    multiplier_map = {
        2024: 1.00,
        2023: 1.10,
        2022: 1.22,
        2021: 1.27,
        2020: 1.29,
        2019: 1.33,
        2018: 1.37,
        2017: 1.42,
        2016: 1.45,
        2015: 1.46,
        2014: 1.50,
        2013: 1.54,
        2012: 1.59,
        2011: 1.67,
        2010: 1.75,
        2009: 1.74,
        2008: 1.81,
        2007: 1.89,
        2006: 1.95,
        2005: 2.00,
        2004: 2.06,
        2003: 2.12,
        2002: 2.16,
        2001: 2.20,
        2000: 2.26,
        1999: 2.30,
        1998: 2.38,
        1997: 2.45,
        1996: 2.51,
        1995: 2.60
    }
    if inflation_correction:
        # Extract the year from the date_of_transfer column
        houses_df['year'] = pd.to_datetime(houses_df['date_of_transfer']).dt.year
        houses_df['price'] = houses_df.apply(lambda row: row['price'] * multiplier_map.get(row['year'], 1.0), axis=1)

    return houses_df

def fetch_building_within_bbox(place_name, latitude, longitude, side_length_km, draw =True):
    half_side_length_lat, half_side_length_lon = calculate_half_side_degrees((latitude, longitude), side_length_km)
    north, south, east, west = latitude + half_side_length_lat, latitude - half_side_length_lat, longitude + half_side_length_lon, longitude - half_side_length_lon
    bbox = (north, south, east, west)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
    
    # Preprocesse OSM building data
    buildings = buildings[buildings['geometry'].notna()]
    buildings = buildings.drop_duplicates(subset='geometry')
    address_full_condition = buildings['addr:housenumber'].notna() & buildings['addr:street'].notna() & buildings['addr:postcode'].notna()
    
    address_columns = [col for col in buildings if str(col).startswith('addr:')]
    buildings = buildings[address_columns + ['geometry', 'building']]

    # Calculate areas: first convert to metric CRS then convert back
    """
    Originally it is in geographic CRS, which is good for latitude and longitude (plotting) but not for area calculation, as the unit is degree squared rather than meters squared.
    ESPG: 4326 is the geographic CRS, and ESPG: 32630 is the metric CRS.
    https://epsg.io/32630
    """
    buildings = buildings.to_crs(epsg=32630)
    buildings['area'] = buildings.geometry.area
    buildings = buildings.to_crs(epsg=4326)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    area = ox.geocode_to_gdf(place_name)
    if draw:
        plot_geodata(buildings, edges, area, (west, east), (south, north), address_full_condition, "With Full Address", place_name + " Buildings with Full Address")

    return buildings, area, nodes, edges

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    pois = ox.features_from_point((latitude, longitude), tags=tags, dist=distance_km*1000)
    poi_counts = {}
    for tag in tags.keys():
        if tag in pois.columns:
            poi_counts[tag] = pois[tag].notnull().sum()
        else:
            poi_counts[tag] = 0

    return poi_counts

def count_specific_pois_near_coordinates(latitude: float, longitude: float, tags: dict, subtags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags (or tags+subtag) and values are the counts of POIs for each tag/(tags+subtag).
    """
    pois = ox.features_from_point((latitude, longitude), tags=(tags | subtags), dist=distance_km*1000)
    poi_counts = {}
    for tag in tags.keys():
        if tag in pois.columns:
            poi_counts[tag] = pois[tag].notnull().sum()
        else:
            poi_counts[tag] = 0
    for tag in subtags.keys():
        if tag in pois.columns:
            for subtag in subtags[tag]:
                poi_counts[tag + ":" + subtag] = len(pois[pois[tag] == subtag])

    return poi_counts

def multi_location_poi_counts(locations_dict: dict, tags: dict, subtags: dict = None) -> pd.DataFrame:
    """
    Count Points of Interest (POIs) near multiple locations.
    Args:
        locations_dict (dict): A dictionary where keys are the location names and values are tuples of (latitude, longitude).
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
    Returns:
        pd.DataFrame: A dataframe where each row contains the counts of POIs for each tag at a given location.
    """
    combined_df = pd.DataFrame()
    if subtags is None:
        for location, (latitude, longitude) in locations_dict.items():
            poi_counts = count_pois_near_coordinates(latitude, longitude, tags)
            poi_counts_df = pd.DataFrame(list(poi_counts.items()), columns=['POI Type', 'Count'])
            poi_counts_df['Location'] = location
            combined_df = pd.concat([combined_df, poi_counts_df], ignore_index=True)
    else:
        for location, (latitude, longitude) in locations_dict.items():
            poi_counts = count_specific_pois_near_coordinates(latitude, longitude, tags, subtags)
            poi_counts_df = pd.DataFrame(list(poi_counts.items()), columns=['POI Type', 'Count'])
            poi_counts_df['Location'] = location
            combined_df = pd.concat([combined_df, poi_counts_df], ignore_index=True)

    return combined_df