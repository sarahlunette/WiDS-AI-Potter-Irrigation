from pyproj import Proj, transform

def transform_coordinates(lat, lon):
    if lat is None or lon is None:
        raise ValueError("Latitude and Longitude cannot be None")
    
    # Définir les systèmes de projection (EPSG:4326 et EPSG:32628 pour UTM Zone 28N comme exemple)
    wgs84 = Proj(init='epsg:4326')  # EPSG:4326 (Latitude, Longitude)
    utm = Proj(init='epsg:32628')  # EPSG:32628 (UTM Zone 28N pour un système local plus précis)

    # Transformation des coordonnées géographiques en EPSG:32628
    x, y = transform(wgs84, utm, lon, lat)

    return x, y

def return_bbox(lat, lon, zone_size = 1000):
    x, y = transform_coordinates(lat, lon)

    # Calcul des coins de la bbox
    minx = x - zone_size / 2
    miny = y - zone_size / 2
    maxx = x + zone_size / 2
    maxy = y + zone_size / 2

    # Affichage de la bbox
    bbox = (minx, miny, maxx, maxy)

    return bbox
