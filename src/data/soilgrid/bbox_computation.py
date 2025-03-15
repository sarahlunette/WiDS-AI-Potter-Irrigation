import numpy as np
def compute_bbox(lat, lon, size_km=0.005):

    """
    Compute a bounding box for a field given its center coordinates and size.
    
    Args:
        lat (float): Latitude of the center.
        lon (float): Longitude of the center.
        size_km (float): Field size (assumed square), default 0.5 km (500m).

    Returns:
        tuple: Bounding box (min_lon, min_lat, max_lon, max_lat).
    """
    lat_offset = size_km / 111.32  # Convert km to degrees latitude
    lon_offset = size_km / (85 * abs(np.cos(np.radians(lat))))  # Convert km to degrees longitude

    min_lat = lat - lat_offset
    max_lat = lat + lat_offset
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset

    return (min_lon, min_lat, max_lon, max_lat)

