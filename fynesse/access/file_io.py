import os
import requests
import geopandas as gpd


def download_file(url, output_file, file_type="binary", chunk_size=1024):
    """
    Downloads a file from the given URL and saves it to the specified output file.

    Args:
        url (str): The URL of the file to download.
        output_file (str): Path to save the downloaded file.
        file_type (str): Type of file ("binary", "stream") to determine the download method.
        chunk_size (int): Size of chunks to download when using streaming mode (in bytes).

    Returns:
        bool: True if the download succeeds, False otherwise.
    """
    output_file = os.path.abspath(output_file)
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return True

    try:
        if file_type == "stream":
            print("Downloading file in streaming mode...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(output_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk: 
                        file.write(chunk)
        else:
            print("Downloading file in non-streaming mode...")
            response = requests.get(url)
            response.raise_for_status()

            with open(output_file, "wb") as file:
                file.write(response.content)

        print(f"File downloaded successfully and saved as {output_file}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return False
    

def geojson_to_csv_with_wkt(geojson_path: str, csv_path: str):
    """
    Converts a GeoJSON file to a CSV file with a WKT column.

    Args:
    geojson_path (str): The path to the GeoJSON file.
    csv_path (str): The path to save the CSV file.
    """
    # Load the GeoDataFrame
    gdf = gpd.read_file(geojson_path)

    # Extract the WKT geometries
    gdf["wkt"] = gdf["geometry"].apply(lambda x: x.wkt)

    # Drop the geometry column
    gdf.drop(columns=["geometry"], inplace=True)

    # Save the DataFrame as a CSV file
    gdf.to_csv(csv_path, index=False)