from shapely.geometry import box
import numpy as np
from math import ceil

# Function to split geometry into smaller subregions
def split_geometry(geometry, x_num_parts, y_num_parts) -> list:
    '''
    For running our models on the subset, we must convert the Google Earth Engine image to an array that can be used by the scikit-learn library. 
    Unfortunately, there is no direct way to do this. 
    Instead, we must download the image from Earth Engine as a raster file and then read it back in as an array. 
    Even though we are only working with a subset of the country, Google's download limits still prevent images of this size from being downloaded. 
    To get around this, the function split_geometry() was created. This function takes in a geometry and splits it into smaller portions that can be downloaded separately.
    '''
    bounds = geometry.bounds().getInfo()
    # Extracting bounding coordinates
    xmin = bounds['coordinates'][0][0][0]
    ymin = bounds['coordinates'][0][0][1]
    xmax = bounds['coordinates'][0][2][0]
    ymax = bounds['coordinates'][0][2][1]
    width = (xmax - xmin) / x_num_parts
    height = (ymax - ymin) / y_num_parts

    subgeometries = []
    for i in range(x_num_parts):
        for j in range(y_num_parts):
            subgeometry = box(xmin + i * width, ymin + j * height,
                              xmin + (i + 1) * width, ymin + (j + 1) * height)
            subgeometries.append(subgeometry)

    return subgeometries


def haversine(lat1=None, lon1=None, lat2=None, lon2=None, coords=None) -> float:
    '''
    The haversine formula calculates the shortest distance between two points on a sphere using their latitudes and longitudes measured along the surface.
    '''
    if coords is not None:

        if len(coords) != 4:
            raise ValueError("If 'coords' is used, it must contain exactly 4 values.")
        
        if any(v is not None for v in [lat1, lon1, lat2, lon2]):
            raise Warning("Both 'coords' and lat/lon values were provided. Using 'coords'.")

        lat1, lon1, lat2, lon2 = coords
    
    if any(v is None for v in [lat1, lon1, lat2, lon2]) and coords is None:
        raise ValueError("All coordinates must be provided if 'coords' is not used.")

    R = 6378137  # radius of Earth in meters
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    meters = R * c  # output distance in meters
    return meters


def calc_segment_count(img_height, img_width, desired_clip_height, desired_clip_width) -> tuple:
    '''
    The function calc_segment_count() calculates the number of vertical and horizontal segments that an image should be split into based on the desired height and width of the segments.
    '''
    return (ceil(img_height / desired_clip_height), ceil(img_width / desired_clip_width))
