# Imports
import os
import sys

import ee
import geemap

import pandas as pd
import json

from tqdm import tqdm
from shapely.geometry import mapping

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, os.path.join('src'))

from split_images import split_geometry



def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask).divide(10000)

class HiddenPrints:
    """A class for hiding prints to the console."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class AnalysisImage:
    """A class for extracting satellite images from Google Earth Engine for further analysis."""

    def __init__(self,
                    presets = None,
                    image = None,
                    year = 2023,
                    left = None,
                    right = None,
                    top = None,
                    bottom = None,
                    feature_bands = None
                 ):
        
        if presets is not None:
            preset_values = self.set_presets(presets)
            self.image = preset_values['image'] if image is None else image
            self.left = preset_values['left'] if left is None else left
            self.right = preset_values['right'] if right is None else right
            self.top = preset_values['top'] if top is None else top
            self.bottom = preset_values['bottom'] if bottom is None else bottom
            self.feature_bands = preset_values['feature_bands'] if feature_bands is None else feature_bands
        elif any([image is None, left is None, right is None, top is None, bottom is None, feature_bands is None]):
            raise ValueError("If presets is None, all other arguments must be provided.")

        self.year = year
        year_start = f'{self.year}-01-01'
        year_end = f'{self.year}-12-31'

        dataset = (
            ee.ImageCollection(self.image)
            .filterDate(year_start, year_end)
            # Pre-filter to get less cloudy granules.
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
            .map(mask_s2_clouds)
        )

        self.geometry_sd = ee.Geometry.Polygon(
            [[[self.left, self.bottom],
            [self.right, self.bottom],
            [self.right, self.top],
            [self.left, self.top]]])


        self.sd_cutout = dataset.median().clip(self.geometry_sd).select(self.feature_bands)

        self.center = [(self.top + self.bottom) / 2, (self.left + self.right) / 2]


    def set_presets(self, presets):
        """Sets the image to a preset location."""

        if presets.lower() == "tegucigalpa":
            return {
                'image': 'COPERNICUS/S2_SR_HARMONIZED',
                'left': -87.2702053,
                'right': -87.1455428,
                'top': 14.1214563,
                'bottom': 14.0312229,
                'feature_bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            }
        
        raise ValueError("Invalid preset")


    def plot_image(self, bands=None, lon = None, lat = None, zoom=13):
        """Plots the saved satellite image."""

        if bands is None:
            bands = ['B4', 'B3', 'B2']

        if lon is None:
            lon = self.center[1]
        
        if lat is None:
            lat = self.center[0]

        visualization = {
            'min': 0.0,
            'max': 0.3,
            'bands': bands,
        }

        m = geemap.Map()
        m.set_center(lon, lat, zoom)
        m.add_layer(self.sd_cutout, visualization, 'RGB')
        return m

    def get_bounds(self, side=None):
        """Returns the bounds of the image."""
        
        if side == "ceiling":
            return [self.top, self.left, self.top, self.right]
        
        if side == "floor":
            return [self.bottom, self.left, self.bottom, self.right]
        
        if side == "left":
            return [self.top, self.left, self.bottom, self.left]
        
        if side == "right":
            return [self.top, self.right, self.bottom, self.right]
        
        return [self.left, self.right, self.top, self.bottom]

    def extract_clips_to_folder(self, folder_name, file_name, x_num_parts, y_num_parts, img_scale, continue_preexisting=False):
        """Extracts the image clips to a folder."""

        # Create a directory for temporary files if it doesn't exist
        temp_dir = folder_name
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        output_files = []

        output_path = os.path.join(temp_dir, f"{file_name}_0.tif")
        if not os.path.exists(output_path) or continue_preexisting:

            # Create dataframe to store image coordinates
            image_coords = pd.DataFrame(columns=["file_name", "x", "y"])

            # Split geometry into smaller subregions (to bypass Earth Engine export limit)
            subgeometries = split_geometry(self.geometry_sd, x_num_parts=x_num_parts, y_num_parts=y_num_parts)

            # Export and download each subregion
            for i, subgeometry in enumerate(tqdm(subgeometries)):
                output_filename = os.path.join(temp_dir, f"{file_name}_{i}.tif")

                if continue_preexisting and os.path.exists(output_filename):
                    continue

                # Convert Shapely geometry to GeoJSON
                geojson_geometry = json.dumps(mapping(subgeometry))

                with HiddenPrints():
                    geemap.ee_export_image(self.sd_cutout, filename=output_filename, region=geojson_geometry, scale=img_scale)

                new_row = pd.DataFrame([{"file_name": output_filename, "x": subgeometry.centroid.x, "y": subgeometry.centroid.y}])

                # Concatenate the new row to the existing DataFrame
                image_coords = pd.concat([image_coords, new_row], ignore_index=True)


            # Read and merge downloaded images
            output_files = [os.path.join(temp_dir, f"{file_name}_{i}.tif") for i in range(len(subgeometries))]


        else:
            print(f"Found existing {os.path.join(temp_dir, f'{file_name}.tif')} files. Use that instead <3.")

        return output_files


