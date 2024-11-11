import numpy as np

def calculate_half_side_degrees(location, side_length_km):
    """
    Calculate half of the side length in degrees for latitude and longitude 
    based on a given location and side length in kilometers.

    Parameters:
    location (tuple): Tuple of (latitude, longitude) for the central point.
    side_length_km (float): Side length of the box in kilometers.

    Returns:
    tuple: Half side length in degrees for latitude and longitude.
    """
    # Unpack the location
    latitude, _ = location

    # Constants for converting degrees to km
    one_deg_lat_in_km = 110.574
    one_deg_lon_in_km = 111.320 * np.cos(np.radians(latitude))

    # Convert half side length from km to degrees
    half_side_lat_deg = (side_length_km / 2) / one_deg_lat_in_km
    half_side_lon_deg = (side_length_km / 2) / one_deg_lon_in_km

    return half_side_lat_deg, half_side_lon_deg