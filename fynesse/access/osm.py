import os
import subprocess
import warnings
import geopandas as gpd
import pandas as pd
import osmium as osm
import shapely
shapely.use_pygeos = True
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm
import osmnx as ox
from geopandas.tools import sjoin
from ..utility import calculate_half_side_degrees


class OSMFeatureHandlerWithProgress(osm.SimpleHandler):
    def __init__(self, tags, total_elements):
        super().__init__()
        self.tags_dict = {key: set(values) for key, values in tags.items()}
        self.key_set = set(key for key in self.tags_dict.keys() if not tags[key])
        self.features = []
        self.total_elements = total_elements
        self.counter = 0
        self.progress_bar = tqdm(total=total_elements, desc="Processing OSM Data", mininterval=1)

    def is_matching(self, osm_tags):
        keys = osm_tags.keys() & self.tags_dict.keys()
        return any((osm_tags[k] in self.tags_dict[k]) or (k in self.key_set) for k in keys)

    def process_element(self, element, element_type):
        tags = dict(element.tags)
        if not self.is_matching(tags):
            return

        if element_type == "node":
            self.features.append({
                "id": element.id,
                "type": "node",
                "tags": tags,
                "geometry": Point(element.location.lon, element.location.lat),
            })
        elif element_type == "way":
            coords = []
            for n in element.nodes:
                coords.append((n.location.lon, n.location.lat))
            if coords:
                try:
                    self.features.append({
                        "id": element.id,
                        "type": "way",
                        "tags": tags,
                        "geometry": Polygon(coords),
                    })
                except ValueError:
                    # Handle invalid polygon geometry
                    pass
        elif element_type == "relation":
            multipolygon = []
            for member in element.members:
                # Ensure we only process valid "outer" or "inner" members of type Way
                if member.role in ["outer", "inner"] and isinstance(member, osm.osm.Way):
                    coords = []
                    for n in member.nodes:
                        try:
                            coords.append((n.location.lon, n.location.lat))
                        except osm.InvalidLocationError:
                            # Skip invalid nodes within the way
                            print(f"Skipping invalid node in relation {element.id}, member {member.ref}")
                            continue
                    if coords:
                        try:
                            polygon = Polygon(coords)
                            multipolygon.append(polygon)
                        except ValueError:
                            # Handle invalid geometry for the polygon
                            print(f"Invalid geometry for relation {element.id}, member {member.ref}")
                            continue
            if multipolygon:
                try:
                    self.features.append({
                        "id": element.id,
                        "type": "relation",
                        "tags": tags,
                        "geometry": MultiPolygon(multipolygon),
                    })
                except ValueError:
                    # Handle invalid MultiPolygon geometries
                    print(f"Invalid MultiPolygon geometry for relation {element.id}")

    def node(self, n):
        self.counter += 1
        if self.counter % 100 == 0:
            self.progress_bar.update(100)
        self.process_element(n, "node")

    def way(self, w):
        self.counter += 1
        if self.counter % 100 == 0:
            self.progress_bar.update(100)
        self.process_element(w, "way")

    def relation(self, r):
        self.counter += 1
        if self.counter % 100 == 0:
            self.progress_bar.update(100)
        self.process_element(r, "relation")

    def close(self):
        self.progress_bar.update(self.counter % 100)  # Finish remaining
        self.progress_bar.close()

def store_osm_data(osm_features):
    """Convert the OSM features to a GeoDataFrame, separate tags, and store as csv for later database operations."""
    gdf = gpd.GeoDataFrame(
        osm_features,
        columns=["id", "type", "tags", "geometry"],
        crs="EPSG:4326"
    )

    # Ensure the geometry column is treated correctly
    gdf.set_geometry("geometry", inplace=True)
    gdf['geometry_wkt'] = gdf['geometry'].apply(lambda x: x.wkt)
    gdf.drop(columns=['geometry'], inplace=True)

    tags_list = []
    for feature in osm_features:
        feature_id = feature["id"]
        tags = feature["tags"]
        for key, value in tags.items():
            tags_list.append({"osm_feature_id": feature_id, "key": key, "value": value})

    # Convert to DataFrame
    tags_df = pd.DataFrame(tags_list)

    # Save GeoDataFrame (gdf) to CSV
    gdf_csv_path = "osm_features.csv"
    gdf.to_csv(gdf_csv_path, index=False)

    # Save tags DataFrame (tags_df) to CSV
    tags_csv_path = "osm_tags.csv"
    tags_df.to_csv(tags_csv_path, index=False)

    return gdf, tags_df


def get_osm_features_counts(oa_geo: gpd.GeoDataFrame, features: list, tags: dict):
    """Calculate feature counts for each OA using spatial join."""
    
    print("Calculating feature counts for each Output Area (OA)...")
    
    # Convert features to GeoDataFrame
    features_gdf = gpd.GeoDataFrame(
        features,
        geometry=[f["geometry"] for f in features],
        crs="EPSG:4326"
    )

    # add spatial index

    _ = features_gdf.sindex
    _ = oa_geo.sindex


    # Perform spatial join
    joined = sjoin(features_gdf, oa_geo, predicate='intersects')
    
    # Create separate columns for each tag key-value pair
    # Collect all new columns into a dictionary first
    new_columns = {
        f"{key}_{value}": joined["tags"].apply(lambda x: int(x.get(key) == value))
        for key, values in tags.items() for value in values
    }

    # Create a DataFrame from the new columns
    new_columns_df = pd.DataFrame(new_columns)

    # Concatenate with the original DataFrame
    joined = pd.concat([joined, new_columns_df], axis=1)


    # Group by OA and sum the counts for each tag
    counts = joined.groupby("FID").sum(numeric_only=True)
    
    counts.reset_index(inplace=True)
    counts = counts[["FID"] + [f"{key}_{value}" for key, values in tags.items() for value in values]]
    oa_geo = oa_geo.merge(counts, how='left', on='FID', suffixes=('', '_r'))

    # Fill NaN values in count columns with 0
    oa_geo = oa_geo.fillna(0)

    return oa_geo


def get_osm_features_from_pbf(pbf_file, tags):
    """Load features from a .osm.pbf file."""

    print("Extracting features from the PBF file...")
    total_elements_minimal = 783609 + 2745785 + 14833
    total_elements_refined = 1627408 + 6329649 + 17764
    handler = OSMFeatureHandlerWithProgress(tags, total_elements_refined)
    handler.apply_file(pbf_file)
    handler.close()
    return handler.features


def run_osmium_commands(input_pbf, output_pbf, tags, tags_node_only=None):
    """Run osmium commands to preprocess the PBF file."""
    print("Preprocessing the PBF file...")

    # Step 1: Filter by tags
    print("Filtering PBF file by tags...")
    filtered_pbf = "filtered.pbf"
    tag_filters = [f"{key}={value}" for key, values in tags.items() for value in values if len(values) > 0] + [f"{key}" for key, values in tags.items() if len(values) == 0]
    if tags_node_only:
        tag_filters += [f"n/{key}={value}" for key, values in tags_node_only.items() for value in values]
    
    subprocess.run(
        ["osmium", "tags-filter", input_pbf, *tag_filters, "-o", filtered_pbf, "--overwrite"],
    )

    print("Adding locations to ways...")
    subprocess.run(
        ["osmium", "add-locations-to-ways", "--keep-untagged-nodes", "-o", output_pbf, filtered_pbf, "--overwrite"],
    )
    
    os.remove(filtered_pbf)


def fetch_building_within_bbox(place_name, latitude, longitude, side_length_km, draw =True):
    from ..assess import plot_geodata

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
    
    # check if place_name is OSM_id to get area
    if place_name[1:].isnumeric():
        area = ox.geocode_to_gdf(place_name, by_osmid=True)
    else:
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

def calculate_and_store_feature_counts(oa_geo: gpd.GeoDataFrame, osm_features: list, tags: dict, output_file: str):
    """Calculate feature counts for each OA using spatial join and store the results in a CSV file."""
    # Calculate feature counts
    oa_geo = get_osm_features_counts(oa_geo, osm_features, tags)
    
    # Convert the GeoJSON file to a CSV file for later database loading, select only the FID and the tag columns
    oa_geo = oa_geo[["FID"] + [f"{key}_{value}" for key, values in tags.items() for value in values]]
    oa_geo.to_csv(output_file, index=False)