from shapely.geometry import box

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


# Function to split geometry into smaller subregions but by size of the parts
def split_geometry_by_size(geometry, x_num_parts, y_num_parts) -> list:

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