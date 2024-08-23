import os, sys

import geemap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import branca

from tqdm import tqdm

from itertools import combinations
from shapely.geometry import mapping

# Import precision_score and recall_score from sklearn.metrics
from sklearn.metrics import precision_score, recall_score

from analysis_image import AnalysisImage


class ClusterTester:
    """ Class used to test clustering results from ELOISA object. """
    def __init__(self, cluster_location_info, analysis_image, validation_data, cluster_order=None, num_clusters=None, palette="viridis", colormap=None):
        """
        Args:
            cluster_location_info (dict): Used get_cluster_location_info() method from ELOISA object.
            analysis_image (AnalysisImage): An AnalysisImage object of the area of interest.
            validation_data (gdp.Geodataframe): A geodataframe with validation data (for example, ground truth slum area data).
            cluster_order (dict or list of lists, optional): A dictionary specifying the relative positions of clusters for color mapping or a list of lists for group coloring.
        """
        
        self.cluster_location_info = cluster_location_info
        self.analysis_image = analysis_image
        self.validation_data = validation_data
        self.cluster_order = cluster_order
        self.sampling_points = None
        self.cluster_metrics = None
        self.combination_metrics = None
        self.points_pivot = pd.DataFrame()
        # Generate a colormap with distinct colors
        if num_clusters is None:
            num_clusters = len(self.cluster_location_info['cluster'].unique())
        cmap = plt.get_cmap(palette)  # Choose an appropriate colormap


        if colormap is None:
            if isinstance(self.cluster_order, dict):
                # Remove any values outside +/- 2 standard deviations from the mean
                cluster_order_values = list(self.cluster_order.values())
                mean = np.mean(cluster_order_values)
                std = np.std(cluster_order_values)
                cluster_order_values = [value for value in cluster_order_values if value >= mean - 2 * std and value <= mean + 2 * std]
                
                # Reassign the cluster order values to the filtered values
                self.cluster_order = {k: v for k, v in self.cluster_order.items() if v in cluster_order_values}
                # Normalize the values to be in the range [0, 1]
                min_value = min(self.cluster_order.values())
                max_value = max(self.cluster_order.values())
                normalized_order = {k: (v - min_value) / (max_value - min_value) for k, v in sorted(self.cluster_order.items())}
                # Create a colormap mapping each cluster number to a color based on normalized values
                self.colormap = {cluster: mcolors.rgb2hex(cmap(normalized_order[cluster])) for cluster in normalized_order}

            elif isinstance(self.cluster_order, list):

                colors = cmap(np.linspace(0, 1, len(self.cluster_order)))
                self.colormap = {}
                for i, cluster_list in enumerate(self.cluster_order):
                    color = mcolors.rgb2hex(colors[i])
                    for cluster in cluster_list:
                        self.colormap[cluster] = color

            else:
                colors = cmap(np.linspace(0, 1, num_clusters))
                self.colormap = {i + 1: mcolors.rgb2hex(colors[i]) for i in range(num_clusters)}
        else:
            self.colormap = colormap

    def plot_clusters(self, sampling_points=None, show_clusters=True, show_image=True, show_validation_data=True, zoom=13, screenshot_mode=False):
        """Plot clusters, image, and validation data."""
        m = folium.Map(location=self.analysis_image.center, zoom_start=zoom)
        
        # Add satellite and other base layers
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer(
            tiles='https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=False,
            control=True,
            subdomains=['mt0', 'mt1', 'mt2', 'mt3']
        ).add_to(m)
        
        if show_clusters:
            cluster_layer = folium.FeatureGroup(name='Clusters')

            if screenshot_mode:
                cluster_layer_screenshot = folium.FeatureGroup(name='Edges')

            for _, row in self.cluster_location_info.iterrows():
                cluster_id = row['cluster']
                color = self.colormap.get(cluster_id, '#000000')  # Default to black if cluster_id not found

                if screenshot_mode:
                    folium.GeoJson(
                        mapping(row['geometry']),
                        style_function=lambda feature, color=color: {
                            'fillColor': color,
                            'color': color,
                            'weight': 1,
                            'fillOpacity': 0,
                            'stroke': True,
                        }
                    ).add_to(cluster_layer_screenshot)

                folium.GeoJson(
                    mapping(row['geometry']),
                    style_function=lambda feature, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 0,
                        'fillOpacity': 0.6,
                    },
                    tooltip=folium.Tooltip(f'Cluster: {cluster_id}')
                ).add_to(cluster_layer)
                
            cluster_layer_screenshot.add_to(m)
            cluster_layer.add_to(m)
            
        
        if show_validation_data:
            validation_layer = folium.FeatureGroup(name='Validation Data')
            folium.GeoJson(
                self.validation_data,
                style_function=lambda feature: {
                    'fillColor': 'black',
                    'color': 'black',
                    'weight': 0,
                    'fillOpacity': 0.6,
                }
            ).add_to(validation_layer)
            validation_layer.add_to(m)
        
        if sampling_points is not None:
            sampling_layer = folium.FeatureGroup(name='Sampling Points')
            folium.GeoJson(
                sampling_points,
                style_function=lambda feature: {
                    'fillColor': 'blue',
                    'color': 'blue',
                    'weight': 0,
                    'fillOpacity': 0.75,
                }
            ).add_to(sampling_layer)
            sampling_layer.add_to(m)
        
        # Create a legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: auto; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; opacity: 0.8;">
        <h4> Legend</h4>
        '''
        
        for cluster_id, color in self.colormap.items():
            legend_html += f'<i style="background:{color};width:10px;height:10px;display:inline-block;"></i> Cluster {cluster_id}<br>'
        
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(m)
        
        return m

    def create_sampling_points(self, n_points=None, points=None):
        """Create points for random sampling"""

        if points is not None and n_points is None:
            points = points.to_crs('EPSG:4326')
        elif n_points is not None and points is None:
            # Create a set of sample points within the area of interest
            x = np.random.uniform(self.analysis_image.left, self.analysis_image.right, n_points)
            y = np.random.uniform(self.analysis_image.bottom, self.analysis_image.top, n_points)
            points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs='EPSG:4326')
        else:
            raise ValueError("Either n_points or points must be provided, but not both.")

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

        # Retain the "within_target" column
        self.points_pivot['within_target'] = self.sampling_points['within_target'] # pylint: disable=unsubscriptable-object

        # Add new columns for each cluster with True/False values
        for cluster in unique_clusters:
            self.points_pivot[cluster] = (self.sampling_points['cluster'] == cluster) # pylint: disable=unsubscriptable-object


        # For each cluster, calculate the precision and recall, and store them in a dataframe
        precision_recall_df = pd.DataFrame(columns=["cluster", "precision", "recall"])

        for cluster in unique_clusters:
            precision = precision_score(self.points_pivot['within_target'], self.points_pivot[cluster])
            recall = recall_score(self.points_pivot['within_target'], self.points_pivot[cluster])

            f1 = 2 * (precision * recall) / (precision + recall)

            precision_recall_df = pd.concat([precision_recall_df, pd.DataFrame({"cluster": cluster, "precision": precision, "recall": recall, "F1": f1}, index=[0])], ignore_index=True)

        self.cluster_metrics = precision_recall_df
        return precision_recall_df

    def group_clusters_and_calc_metrics(self, precision_min): #TODO: Does this work properly?
        """Test groups of clusters and calculate metrics."""

        if self.cluster_metrics is None:
            raise ValueError("Cluster metrics have not been calculated. Run calc_cluster_metrics() first.")

        # Create a subset of the clusters with a Precision score higher than precision_min
        high_precision_clusters = self.cluster_metrics[self.cluster_metrics['precision'] >= precision_min]['cluster'].tolist() # pylint: disable=unsubscriptable-object

        # Create a new empty dataframe
        combinations_df = pd.DataFrame(columns=["combination"])

        # Create a list of all possible combinations of clusters

        for i in range(1, len(high_precision_clusters) + 1):
            for combination in combinations(high_precision_clusters, i):
                combination = list(combination)

                if len(combination) > 5:
                    break

                combinations_df = pd.concat([combinations_df, pd.DataFrame({"combination": [combination]})], ignore_index=True)


        # For each combination of clusters, calculate the precision and recall
        for i, row in tqdm(combinations_df.iterrows(), total=combinations_df.shape[0]):
            combination = row['combination']


            precision = precision_score(self.points_pivot['within_target'], self.points_pivot[combination].any(axis=1))
            recall = recall_score(self.points_pivot['within_target'], self.points_pivot[combination].any(axis=1))

            
            combinations_df.loc[i, "precision"] = precision
            combinations_df.loc[i, "recall"] = recall

        # Sort the combinations by precision
        combinations_df = combinations_df.sort_values(by="precision", ascending=False)

        combinations_df["F1"] = 2 * (combinations_df["precision"] * combinations_df["recall"]) / (combinations_df["precision"] + combinations_df["recall"])

        self.combination_metrics = combinations_df
        return combinations_df

    def plot_top_cluster_combos(self, metric, n_combos=5, zoom=13, show_validation_data=True):
        """Plot the top n_combos cluster combinations by metric score."""
        
        if self.combination_metrics is None:
            raise ValueError("Cluster metrics have not been calculated. Run group_clusters_and_calc_metrics() first.")

        if metric not in ["precision", "recall", "F1"]:
            raise ValueError("Metric must be either 'precision', 'recall', or 'F1'.")

        m = geemap.Map()
        m.set_center(self.analysis_image.center[1], self.analysis_image.center[0], zoom)

        if show_validation_data:
            m.add_gdf(self.validation_data, layer_name='Validation Data', fill_colors='red', style={'fillOpacity': 0.75, 'opacity': 0})

        cluster_colors = list(self.colormap.values())

        for i, row in self.combination_metrics.sort_values(by=metric, ascending=False).head(n_combos).iterrows():
            combination = row['combination']
            cluster_gdf_subset = self.cluster_location_info[self.cluster_location_info.index.isin(combination)]
            m.add_gdf(cluster_gdf_subset, layer_name=f'Combination {i+1}', fill_colors=cluster_colors, style={'fillOpacity': 0.75, 'opacity': 0})

        return m