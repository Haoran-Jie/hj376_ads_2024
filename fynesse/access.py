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



def fetch_houses_within_box(location: tuple[float], side_length_km: float, conn: pymysql.connections.Connection, since_date: str = '2020-01-01') -> pd.DataFrame:
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
    return houses_df

def fetch_building_within_bbox(place_name, latitude, longitude, side_length_km, draw=True):
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
        fig, ax = plt.subplots(figsize=(10, 10))
        area.plot(ax=ax, facecolor="white")
        edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
        ax.set_xlim([west, east])
        ax.set_ylim([south, north])
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")

        buildings[address_full_condition].plot(ax=ax, color="red", alpha=0.7, markersize=10, label="Full Address")
        buildings[~address_full_condition].plot(ax=ax, color="blue", alpha=0.7, markersize=10, label="Partial or None Address")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.legend()
        
        plt.tight_layout()
        plt.show()

    return buildings, area, nodes, edges