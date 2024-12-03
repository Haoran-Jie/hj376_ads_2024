import pymysql
import pandas as pd
import requests
import warnings
from ..utility import calculate_half_side_degrees



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