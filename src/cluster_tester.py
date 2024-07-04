import os, sys

import geemap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import numpy as np

# Import precision_score and recall_score from sklearn.metrics
from sklearn.metrics import precision_score, recall_score

from analysis_image import AnalysisImage


class ClusterTester:
    """ Class used to test clustering results from ELOISA object. """

    def __init__(self, cluster_location_info, analysis_image, validation_data):
        """.git\

        Args:
            cluster_location_info (dict): Used get_cluster_location_info() method from ELOISA object.
            analysis_image (AnalysisImage): An AnalysisImage object of the area of interest.
            validation_data (gdp.Geodataframe): A geodataframe with validation data (for example, ground truth slum area data).
        
        
        """
        
        self.cluster_location_info = cluster_location_info
        self.analysis_image = analysis_image
        self.validation_data = validation_data

        self.sampling_points = None
    
    def plot_clusters(self, sampling_points=None, show_clusters=True, show_image=True, show_validation_data=True, zoom=13):
        """Plot clusters, image and validation data."""

        visualization = {
            'min': 0.0,
            'max': 0.3,
            'bands': ['B4', 'B3', 'B2'],
        }

        m = geemap.Map()
        m.set_center(self.analysis_image.center[1], self.analysis_image.center[0], zoom)

        if show_image:
            m.add_layer(self.analysis_image.sd_cutout, visualization, 'RGB')

        if show_clusters:
            # Generate a colormap with x distinct colors
            num_clusters = len(self.cluster_location_info['cluster'].unique())
            cmap = plt.get_cmap('tab20')  # You can choose other colormaps like 'tab20', 'tab20c', etc.
            colors = cmap(np.linspace(0, 1, num_clusters))

            # Create a dictionary mapping each cluster number to a color
            colormap = {i + 1: mcolors.rgb2hex(colors[i]) for i in range(num_clusters)}

        
            m.add_gdf(self.cluster_location_info, layer_name='Clusters', fill_colors=list(colormap.values()), style={'fillOpacity': 0.75, 'opacity': 0},
                hover_style={'fillOpacity': 0.25, 'opacity': 1})

        if show_validation_data:
            m.add_gdf(self.validation_data, layer_name='Validation Data', fill_colors='red', style={'fillOpacity': 0.75, 'opacity': 0})

        if sampling_points is not None:
            m.add_gdf(sampling_points, layer_name='Sampling Points', fill_colors='blue', style={'fillOpacity': 0.75, 'opacity': 0})

        return m

    def create_sampling_points(self, n_points):
        """Create points for random sampling"""

        # Create a set of sample points within the area of interest
        x = np.random.uniform(self.analysis_image.left, self.analysis_image.right, n_points)
        y = np.random.uniform(self.analysis_image.bottom, self.analysis_image.top, n_points)
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs='EPSG:4326')

        # Assign the cluster number to each point based on the cluster it falls within
        points['cluster'] = points.apply(lambda x: next((index for index, row in self.cluster_location_info.iterrows() if row.geometry.contains(x.geometry)), np.nan), axis=1)

        # Convert the cluster column to an integer type
        points['cluster'] = points['cluster'].astype('Int64')

        # Remove any points that do not fall within a cluster
        points = points.dropna(subset=['cluster'])

        # Assign whether each point falls within a validation area
        points['within_target'] = points.apply(lambda x: self.validation_data.contains(x.geometry).any(), axis=1)

        self.sampling_points = points
        return points

    def calc_sampling_metrics(self):
        """Calculate precision and recall for the sampling points"""

        if self.sampling_points is None:
            raise ValueError("Sampling points have not been created. Run create_sampling_points() first.")


        unique_clusters = self.sampling_points.drop(columns='geometry')['cluster'].unique()

        # Create a new dataframe to hold the transformed data
        points_pivot = pd.DataFrame()

        # Retain the "within_target" column
        points_pivot['within_target'] = self.sampling_points['within_target']

        # Add new columns for each cluster with True/False values
        for cluster in unique_clusters:
            points_pivot[cluster] = (self.sampling_points['cluster'] == cluster)


        # For each cluster, calculate the precision and recall, and store them in a dataframe
        precision_recall_df = pd.DataFrame(columns=["cluster", "precision", "recall"])

        for cluster in unique_clusters:
            precision = precision_score(points_pivot['within_target'], points_pivot[cluster])
            recall = recall_score(points_pivot['within_target'], points_pivot[cluster])

            f1 = 2 * (precision * recall) / (precision + recall)

            precision_recall_df = pd.concat([precision_recall_df, pd.DataFrame({"cluster": cluster, "precision": precision, "recall": recall, "F1": f1}, index=[0])], ignore_index=True)

        return precision_recall_df