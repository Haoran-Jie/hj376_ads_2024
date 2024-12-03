import pymysql
import yaml
import pandas as pd
import warnings
from typing import List




def execute_query(connection: pymysql.connections.Connection, query: str):
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
        connection.commit()
        print(f"Query executed successfully:\n{query}")
    except Exception as e:
        print(f"Error executing query:\n{query}\nError: {e}")


def load_csv_to_table(connection: pymysql.connections.Connection, table_name: str, file_path: str, columns: str, ignore_rows: int = 1):
    query = f"""
    LOAD DATA LOCAL INFILE '{file_path}'
    INTO TABLE {table_name}
    FIELDS TERMINATED BY ','
    OPTIONALLY ENCLOSED BY '"'
    LINES TERMINATED BY '\\n'
    IGNORE {ignore_rows} ROWS
    ({columns});
    """
    execute_query(connection, query)

def load_csv_to_table_with_geometry_conversion(connection: pymysql.connections.Connection, table_name: str, file_path: str, columns: str, ignore_rows: int = 1):

    # Step 1: Load CSV data into the table
    load_csv_to_table(connection, table_name, file_path, columns, ignore_rows)

    # Step 2: Convert WKT to GEOMETRY in the `geom` column
    update_query = f"""
    UPDATE {table_name}
    SET geom = ST_GeomFromText(geometry_wkt)
    WHERE geometry_wkt IS NOT NULL;
    """
    execute_query(connection, update_query)

def create_table(connection: pymysql.connections.Connection, create_query: str):
    execute_query(connection, create_query)

def create_index(connection: pymysql.connections.Connection, table_name: str, column_name: str | List[str]):
    if isinstance(column_name, list):
        column_name_index = "_".join(column_name)
        column_name = ["`" + column + "`" for column in column_name]
        column_name = ", ".join(column_name)
    else:
        column_name_index = column_name
    query = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name_index} ON {table_name} ({column_name});"
    execute_query(connection, query)

def create_spatial_index(connection: pymysql.connections.Connection, table_name: str, column_name: str):
    modify_not_null_query = f"ALTER TABLE {table_name} MODIFY COLUMN {column_name} GEOMETRY NOT NULL;"
    create_spatial_index_query = f"ALTER TABLE {table_name} ADD SPATIAL INDEX({column_name});"

    execute_query(connection, modify_not_null_query)
    execute_query(connection, create_spatial_index_query)


def load_credentials(yaml_file = "../credentials.yaml"):
    with open(yaml_file) as file:
        credentials = yaml.safe_load(file)
    return credentials['username'], credentials['password'], credentials['url'], credentials['port']


def read_sql_ignoring_warnings(query, con, *args, **kwargs):
    """Wrapper for pandas.read_sql that suppresses UserWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_sql(query, con, *args, **kwargs)
    


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