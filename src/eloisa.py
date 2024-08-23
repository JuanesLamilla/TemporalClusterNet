# type: ignore

import os, sys

from collections import defaultdict, Counter

from copy import deepcopy
from pprint import pprint

import rasterio
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, OPTICS, DBSCAN

import keras
# from keras.preprocessing import image # pylint: disable=import-error 
from keras.preprocessing import image as keras_image # pylint: disable=import-error 

from tqdm import tqdm

import matplotlib.pyplot as plt

from threading import Lock

import sqlite3
import pickle

import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier

import geopandas as gpd

import concurrent.futures


sys.path.insert(1, os.path.join('src'))
import feature_extraction as fe # pylint: disable=wrong-import-position
from subspace_clustering.cluster.selfrepresentation import ElasticNetSubspaceClustering 

class Eloisa:
    """
    Earth Landscape Observation and Intelligent Satellite Analysis (ELOISA) is a class used for 
    feature extraction, clustering, and unsupervised classification of satellite image segments.

    Args:
        name (str): The name of the Eloisa object.
        seed (int): The seed for the random number generator.
        db_folder (str): The folder where the Eloisa database will be stored.
        image_shape (tuple): The shape of the images in the Eloisa object.

    Attributes:
        _data (dict): A nested dictionary containing the data stored in the Eloisa object.
            Contains the following keys:
                - year (int): The year of the data.
                    - image_list (list): A list of images.
                    - model_name (str): The name of the model used for feature extraction.
                    - features (pd.DataFrame): The extracted features.
                    - features_reduced (pd.DataFrame): The reduced features.
                    - cluster (pd.Series): The cluster labels.

        _seed (int): The seed for the random number generator.
        image_shape (tuple): The shape of the images in the Eloisa object.
        db_path (str): The path to the Eloisa database.
        _database (sqlite3.Connection): The connection to the Eloisa database.
        image_names (dict): A dictionary containing the names of the images stored in the Eloisa object.
    """



    def __init__(self, name, seed, db_folder, image_shape=(600, 600, 3)):
        self._data = defaultdict(dict)
        self._seed = seed
        self.image_shape = image_shape
        self.image_names = {}
        self._lock = Lock()

        self.db_path = os.path.join(db_folder, f"{name}.db")

        if os.path.isfile(self.db_path):
            print("The database already exists. Connecting to it.")
            self._database = sqlite3.connect(self.db_path)

            self.load_all_features_from_db()

        else:
            print("The database does not yet exist. Creating it.")
            self._database = sqlite3.connect(self.db_path)
            self._database.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER,
                    image_clip TEXT,               
                    image BLOB,
                    bands_used TEXT,
                    model_name TEXT,
                    features TEXT,
                    features_reduced TEXT,
                    cluster INTEGER
                )
                '''
            )
            self._database.commit()

    def __getitem__(self, idx):
        return self._data[idx]
        

    def close_db(self):
        self._database.close()

    def open_db(self):
        self._database = sqlite3.connect(self.db_path)

    def head(self):
        printable_data = deepcopy(self._data)

        for year in printable_data:
            printable_data[year]["image_list"] = str(len(printable_data[year]["image_list"][0])) + " images"

        pprint(dict(printable_data))
        del printable_data

    def get_data(self, year=None):
        """Returns the data stored in the Eloisa object."""
        if year:
            return self._data[year]
        return self._data

    def vacuum_db(self):
        """Optimizes the database."""
        self._database.execute("VACUUM")
        self._database.commit()
    
    
    def import_images_by_year(self, folder_path, year, bands):
        """Imports images from a folder and stores them in the Eloisa object."""
        image_list = []
        self.image_names[year] = []

        for img_path in tqdm(os.listdir(folder_path)):
            with rasterio.open(os.path.join(folder_path, img_path)) as src:

                img = src.read(bands)  # Read the first three channels (RGB)
                img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)
                
                # Normalize to 0-255 and convert to uint8
                img = (img - img.min()) / (img.max() - img.min()) * 255
                img = img.astype(np.uint8)

                # Convert to Pillow image
                pil_img = image.array_to_img(img, scale=False)
                pil_img = pil_img.resize((self.image_shape[0], self.image_shape[1]))  # Resize the image to the desired size
                
                image_list.append(pil_img)

                self.image_names[year].append(img_path)

                self._database.execute('''
                    INSERT OR REPLACE INTO images (year, image_clip, image, bands_used) VALUES (?, ?, ?, ?)''', (year, img_path, os.path.join(folder_path, img_path), json.dumps(bands)))

        self._data[year]["image_list"] = image_list
        
        self._database.commit()

    def import_images_from_db(self, year):
        """Imports images from the Eloisa database."""
        cursor = self._database.cursor()
        cursor.execute(f'''
            SELECT image FROM images WHERE year = {year}
        ''')

        image_list = []
        for row in cursor.fetchall():
            img = pickle.loads(row[0])
            image_list.append(img)

        self._data[year]["image_list"] = image_list

    def show_clip_by_year(self, years: list, index: int, model=None, fontsize=12):
        """Displays the image clip for the specified years and index."""

        fig, axes = plt.subplots(1, len(years), figsize=(20, 10))
        for i, ax in enumerate(axes.flat):
            
            

            ax.imshow(self._data[years[i]]["image_list"][index])
            ax.axis("off")

            if model is not None and self._data[years[i]][model.__name__ + "_cluster"] is not None:

                ax.set_title(f"{years[i]} (Cluster: {self._data[years[i]][model.__name__ + "_cluster"][index]})", fontsize=fontsize)

            else:
                ax.set_title(f"{years[i]}")

        plt.tight_layout()
        plt.show()
    
    def show_images_sample(self, year, rows=2, cols=5):
        """Displays a sample of images from the specified year."""
        # Show the images in image_list_testing
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(self._data[year]["image_list"][i-1])
            ax.axis("off")
            ax.set_title(f"Image {i+1}")

        plt.tight_layout()
        plt.show()

    def extract_features(self, year, model, preprocess_input):
        """Extracts features from the images using a pre-trained model."""

        
        if model.__name__ not in self._data[year]:
            self._data[year][model.__name__] = fe.extract(model, preprocess_input, self._data[year]["image_list"], self.image_shape)
        else:
            print(f"Features for {model.__name__} already exist in Eloisa data for year {year}.")

    def extract_features_multiyear(self, years, model, preprocess_input):
        """Extracts features from the images using a pre-trained model across multiple years."""

        # Collect images from all specified years
        combined_image_list = []
        year_image_counts = {}
        
        for year in years:
            if model.__name__ in self._data[year]:
                print(f"Features for {model.__name__} already exist in Eloisa data for year {year}.")
                return  # Exit if features already exist for any year
            combined_image_list.extend(self._data[year]["image_list"])
            year_image_counts[year] = len(self._data[year]["image_list"])

        # Extract features for the combined image list
        combined_features = fe.extract(model, preprocess_input, combined_image_list, self.image_shape)

        # Split the combined features back into their respective years
        start_idx = 0
        for year in years:
            end_idx = start_idx + year_image_counts[year]
            self._data[year][model.__name__] = combined_features[start_idx:end_idx]
            start_idx = end_idx


    def update_database(self):
        """Updates the database with up-to-date data from the Eloisa object."""
        for year in self._data:
            for model_name in self._data[year]:
                if model_name == "image_list" or model_name[-8:] == "_cluster" or model_name[-8:] == "_reduced":
                    continue

                features = self._data[year][model_name]

                
                j = 0
                for i, row in features.iterrows():

                    # Convert features to json
                    features_as_json = json.dumps(row.values.flatten().tolist())
                    features_reduced_as_json = json.dumps(self._data[year][model_name + "_reduced"][j].values.flatten().tolist() 
                                                          if model_name + "_reduced" in self._data[year] else None)
                    cluster_int = int(self._data[year][model_name + "_cluster"][j]) if model_name + "_cluster" in self._data[year] else None

                    self._database.execute('''UPDATE images
                        SET features = ?, features_reduced = ?, model_name = ?, cluster = ?
                        WHERE year = ? AND image = ?''', 
                        (features_as_json, features_reduced_as_json, model_name, cluster_int, year, self.image_names[year][j]))
                    
                    j += 1

        self._database.commit()

    # def load_all_features_from_db(self):
    #     """Loads all features from the Eloisa database."""
    #     cursor = self._database.cursor()
    #     cursor.execute('''
    #         SELECT year, image_clip, image, bands_used, model_name, features, features_reduced, cluster
    #         FROM images
    #     ''')

    #     for i, row in enumerate(tqdm(cursor.fetchall())):
    #         year, image_clip, image, bands_used, model_name, features, features_reduced, cluster = row

    #         if year not in self._data:
    #             self._data[year] = {}

    #         if 'image_list' not in self._data[year]:
    #             self._data[year]['image_list'] = []
    #             self.image_names[year] = []
            
    #         bands_used_list = json.loads(bands_used)

    #         with rasterio.open(image) as src:

    #             img = src.read(bands_used_list)  # Read the first three channels (RGB)
    #             img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)
                
    #             # Normalize to 0-255 and convert to uint8
    #             img = (img - img.min()) / (img.max() - img.min()) * 255
    #             img = img.astype(np.uint8)

    #             # Convert to Pillow image
    #             pil_img = keras.preprocessing.image.array_to_img(img, scale=False)
    #             pil_img = pil_img.resize((self.image_shape[0], self.image_shape[1]))  # Resize the image to the desired size

    #             self.image_names[year].append(image)

    #             self._data[year]["image_list"].insert(i, pil_img)

    #         if features is not None and features != 'null':
    #             if model_name not in self._data[year]:
    #                 self._data[year][model_name] = pd.DataFrame(columns=np.arange(0, len(json.loads(features))))

    #             self._data[year][model_name].loc[i] = json.loads(features)


    #         if features_reduced is not None and features_reduced != 'null':

    #             if model_name + '_reduced' not in self._data[year]:
    #                 self._data[year][model_name + '_reduced'] = pd.DataFrame(columns=np.arange(0, len(json.loads(features_reduced))))

    #             self._data[year][model_name + '_reduced'].loc[i] = json.loads(features_reduced)

    #         if cluster is not None and cluster != 'null':

    #             if model_name + "_cluster" not in self._data[year]:
    #                 self._data[year][model_name + "_cluster"] = []
                
    #             self._data[year][model_name + "_cluster"].insert(i, cluster)

    def process_row(self, row, i):
        year, image_clip, image, bands_used, model_name, features, features_reduced, cluster = row

        with self._lock:  # Ensure thread-safe access to shared data structures
            if year not in self._data:
                self._data[year] = {}

            if 'image_list' not in self._data[year]:
                self._data[year]['image_list'] = []
                self.image_names[year] = []

        bands_used_list = json.loads(bands_used)

        with rasterio.open(image) as src:
            img = src.read(bands_used_list)  # Read the specified bands
            img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)

            # Normalize to 0-255 and convert to uint8
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)

            # Convert to Pillow image
            pil_img = keras_image.array_to_img(img, scale=False)
            pil_img = pil_img.resize((self.image_shape[0], self.image_shape[1]))  # Resize the image to the desired size

            with self._lock:  # Ensure thread-safe access to shared data structures
                self.image_names[year].append(image)
                self._data[year]["image_list"].append(pil_img)

        if features is not None and features != 'null':
            features_data = json.loads(features)
            with self._lock:  # Ensure thread-safe access to shared data structures
                if model_name not in self._data[year]:
                    self._data[year][model_name] = pd.DataFrame(columns=np.arange(0, len(features_data)))

                self._data[year][model_name].loc[i] = features_data

        if features_reduced is not None and features_reduced != 'null':
            features_reduced_data = json.loads(features_reduced)
            with self._lock:  # Ensure thread-safe access to shared data structures
                if model_name + '_reduced' not in self._data[year]:
                    self._data[year][model_name + '_reduced'] = pd.DataFrame(columns=np.arange(0, len(features_reduced_data)))

                self._data[year][model_name + '_reduced'].loc[i] = features_reduced_data

        if cluster is not None and cluster != 'null':
            with self._lock:  # Ensure thread-safe access to shared data structures
                if model_name + "_cluster" not in self._data[year]:
                    self._data[year][model_name + "_cluster"] = []

                self._data[year][model_name + "_cluster"].append(cluster)

    def load_all_features_from_db(self):
        """Loads all features from the Eloisa database."""

        cursor = self._database.cursor()
        cursor.execute('''
            SELECT year, image_clip, image, bands_used, model_name, features, features_reduced, cluster
            FROM images
        ''')

        batch_size = 10000  # You can adjust the batch size as needed
        rows = cursor.fetchmany(batch_size)
        index_offset = 0

        while rows:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(lambda idx_row: self.process_row(idx_row[1], idx_row[0] + index_offset), enumerate(rows)), total=len(rows)))

            index_offset += batch_size
            rows = cursor.fetchmany(batch_size)

    # def scale_features(self, years, model):
    #     """Scales the features using StandardScaler."""
    #     if isinstance(years, int):
    #         scaler = StandardScaler()
    #         self._data[years][model.__name__ + "_reduced"] = scaler.fit_transform(self._data[years][model.__name__])
    #         return

    #     elif isinstance(years, list):

    #         data_list = [self._data[year][model.__name__] for year in years]
            
    #         # Concatenate data from all years
    #         concatenated_data = pd.concat(data_list, axis=0)
            
    #         # Scale the concatenated data
    #         scaler = StandardScaler()
    #         scaled_data = scaler.fit_transform(concatenated_data)
            
    #         # Split the scaled data back into their respective years
    #         start_idx = 0
    #         for year in years:
    #             end_idx = start_idx + len(self._data[year][model.__name__])
    #             self._data[year][model.__name__ + "_reduced"] = scaled_data[start_idx:end_idx]
    #             start_idx = end_idx
            
    #         return

    #     else:
    #         raise ValueError("Year must be an integer or a list of integers.")

    def scale_features(self, years, model):
        """Scales the features using StandardScaler."""
        if isinstance(years, int):
            scaler = StandardScaler()
            self._data[years][model.__name__ + "_reduced"] = scaler.fit_transform(self._data[years][model.__name__])

            # Convert to DataFrame
            self._data[years][model.__name__ + "_reduced"] = pd.DataFrame(self._data[years][model.__name__ + "_reduced"], index=self._data[years][model.__name__].index)

            return

        elif isinstance(years, list):
            # Step 1: Initialize the scaler
            scaler = StandardScaler()

            # Step 2: Incrementally fit the scaler
            for year in years:
                scaler.partial_fit(self._data[year][model.__name__])

            # Step 3: Transform the data in chunks
            for year in years:
                self._data[year][model.__name__ + "_reduced"] = scaler.transform(self._data[year][model.__name__])

                # Convert to DataFrame
                self._data[year][model.__name__ + "_reduced"] = pd.DataFrame(self._data[year][model.__name__ + "_reduced"], index=self._data[year][model.__name__].index)

            
            return

        else:
            raise ValueError("Year must be an integer or a list of integers.")

            
        

    def pca_features(self, years, model, n_components=None, variance_min=None, plot_variance=False):
        """Performs PCA on the features."""

        if n_components is None and variance_min is None:
            raise ValueError("Either n_components or variance_min must be provided.")
        if n_components is not None and variance_min is not None:
            raise ValueError("Only one of n_components or variance_min can be provided.")

        if isinstance(years, int):

            if n_components is None:

                pca = PCA(random_state=self._seed)
                pca.fit(self._data[years][model.__name__ + "_reduced"])

                # If you want to retain variance_min% of the variance
                cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

                if plot_variance:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
                    plt.xlabel('Number of Components')
                    plt.ylabel('Cumulative Explained Variance')
                    plt.title('Explained Variance vs. Number of Components')
                    plt.grid(True)
                    plt.show()

                n_components = np.argmax(cumulative_explained_variance >= variance_min) + 1
                print(f'Number of components to retain {variance_min * 100}% variance: {n_components}')

            pca = PCA(n_components=n_components, random_state=self._seed)
            pca.fit(self._data[years][model.__name__])
            self._data[years][model.__name__ + "_reduced"] = pd.DataFrame(pca.transform(self._data[years][model.__name__ + "_reduced"]), index=self._data[years][model.__name__].index)

        elif isinstance(years, list):

            # Collect and concatenate numpy arrays from all specified years
            data_list = [self._data[year][model.__name__ + "_reduced"] for year in years]
            concatenated_data = np.vstack(data_list)
            
            if n_components is None:
                pca = PCA(random_state=self._seed)
                pca.fit(concatenated_data)
                
                # If you want to retain variance_min% of the variance
                cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
                
                if plot_variance:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
                    plt.xlabel('Number of Components')
                    plt.ylabel('Cumulative Explained Variance')
                    plt.title('Explained Variance vs. Number of Components')
                    plt.grid(True)
                    plt.show()
                
                n_components = np.argmax(cumulative_explained_variance >= variance_min) + 1
                print(f'Number of components to retain {variance_min * 100}% variance: {n_components}')
            
            # Perform PCA with the determined number of components
            pca = PCA(n_components=n_components, random_state=self._seed)
            pca_transformed_data = pca.fit_transform(concatenated_data)
            
            # Split the PCA-transformed data back into their respective years and convert to DataFrames
            start_idx = 0
            for year, data in zip(years, data_list):
                end_idx = start_idx + len(data)
                self._data[year][model.__name__ + "_reduced"] = pd.DataFrame(pca_transformed_data[start_idx:end_idx], index=self._data[year][model.__name__].index)
                start_idx = end_idx
    
    # def pca_features(self, years, model, n_components=None, variance_min=None, plot_variance=False, chunk_size=1000):
    #     """Performs PCA on the features."""

    #     if n_components is None and variance_min is None:
    #         raise ValueError("Either n_components or variance_min must be provided.")
    #     if n_components is not None and variance_min is not None:
    #         raise ValueError("Only one of n_components or variance_min can be provided.")

    #     if isinstance(years, int):
    #         years = [years]  # Convert to list for uniform handling

    #     if isinstance(years, list):
    #         pca = IncrementalPCA(n_components=n_components if n_components is not None else None)

    #         # Accumulate all data
    #         all_data = []
    #         for year in years:
    #             data = self._data[year][model.__name__ + "_reduced"]
    #             all_data.append(data)
    #         all_data = np.vstack(all_data)

    #         # Fit IncrementalPCA on the accumulated data in chunks
    #         for i in range(0, len(all_data), chunk_size):
    #             pca.partial_fit(all_data[i:i + chunk_size])

    #         # Compute cumulative explained variance
    #         cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    #         if variance_min is not None:
    #             if plot_variance:
    #                 plt.figure(figsize=(10, 6))
    #                 plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    #                 plt.xlabel('Number of Components')
    #                 plt.ylabel('Cumulative Explained Variance')
    #                 plt.title('Explained Variance vs. Number of Components')
    #                 plt.grid(True)
    #                 plt.show()

    #             n_components = np.argmax(cumulative_explained_variance >= variance_min) + 1
    #             print(f'Number of components to retain {variance_min * 100}% variance: {n_components}')

    #             # Reinitialize IncrementalPCA with determined n_components
    #             pca = IncrementalPCA(n_components=n_components)
    #             for i in range(0, len(all_data), chunk_size):
    #                 pca.partial_fit(all_data[i:i + chunk_size])

    #         # Transform data for each year incrementally
    #         for year in years:
    #             data = self._data[year][model.__name__ + "_reduced"]
    #             transformed_data = []
    #             for i in range(0, len(data), chunk_size):
    #                 chunk = data[i:i + chunk_size]
    #                 transformed_data.append(pca.transform(chunk))

    #             transformed_data = np.vstack(transformed_data)
    #             self._data[year][model.__name__ + "_reduced"] = pd.DataFrame(transformed_data, index=data.index)

    #     else:
    #         raise ValueError("Year must be an integer or a list of integers.")


    def calc_silhouette_score(self, years, model, kmin=10, kmax=100, increment=10, show_plot=True):
        """Calculates the silhouette score for different numbers of clusters."""

        # Collect data from all specified years
        data_list = [self._data[year][model.__name__ + "_reduced"] for year in years]
        concatenated_data = pd.concat(data_list, axis=0)

        sil = []

        distorsions = []

        for k in tqdm(range(kmin, kmax + 1, increment), desc="Calculating silhouette scores"):
            kmeans = KMeans(n_clusters=k, random_state=self._seed).fit(concatenated_data)
            labels = kmeans.labels_
            sil.append(silhouette_score(concatenated_data, labels, metric='euclidean'))

            distorsions.append(kmeans.inertia_)

        if show_plot:
            # Plot silhouette score
            plt.figure(figsize=(10, 6))
            plt.plot(range(kmin, kmax + 1, increment), sil, marker='o')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score vs. Number of Clusters')
            plt.grid(True)
            plt.show()

        return sil, distorsions
    
    def plot_elbow_curve(self, years, model, kmax=100):
        """Plots the elbow method for different numbers of clusters."""
        # Collect data from all specified years
        data_list = [self._data[year][model.__name__ + "_reduced"] for year in years]
        concatenated_data = pd.concat(data_list, axis=0)
        
        distorsions = []
        for k in range(2, kmax):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(concatenated_data)
            distorsions.append(kmeans.inertia_)

        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(2, kmax), distorsions)
        plt.grid(True)
        plt.title('Elbow curve')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortion')
        plt.show()

    def calc_kmeans_clusters(self, years, model, n_clusters):
        """Calculates the KMeans clusters for the features."""

        # kmeans = KMeans(n_clusters=n_clusters, random_state=self._seed)
        # self._data[year][model.__name__ + "_cluster"] = kmeans.fit_predict(self._data[year][model.__name__ + "_reduced"])

            # Collect data from all specified years
        data_list = [self._data[year][model.__name__ + "_reduced"] for year in years]
        concatenated_data = pd.concat(data_list, axis=0)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self._seed)
        kmeans_labels = kmeans.fit_predict(concatenated_data)
        
        # Split the KMeans labels back into their respective years
        start_idx = 0
        for year in years:
            end_idx = start_idx + len(self._data[year][model.__name__ + "_reduced"])
            self._data[year][model.__name__ + "_cluster"] = kmeans_labels[start_idx:end_idx]
            start_idx = end_idx

        return kmeans
    
    def get_cluster_sequences(self, years, model):
        """Returns the sequences of clusters as they change over time."""
        cluster_sequence = []
        for year in years:
            cluster_sequence.append(list(self._data[year][model.__name__ + "_cluster"]))

        cluster_sequence = list(map(list, zip(*cluster_sequence)))

        return cluster_sequence

    def subspace_clustering(self, years, model, n_clusters, algorithm="lasso_lars", gamma=50, n_jobs=-1, sample_fraction=0.1):
        """Performs subspace clustering on a subset of the features using ElasticNet and applies the results to the rest."""

        if sample_fraction <= 0 or sample_fraction > 1:
            raise ValueError("sample_fraction must be in the range (0, 1].")

        # Collect data from all specified years
        data_list = [self._data[year][model.__name__ + "_reduced"] for year in years]
        concatenated_data = pd.concat(data_list, axis=0)

        # Sample a subset of the data
        sampled_data = concatenated_data.sample(frac=sample_fraction, random_state=self._seed)

        # Perform subspace clustering on the sampled data
        sc = ElasticNetSubspaceClustering(n_clusters=n_clusters, gamma=gamma, n_jobs=n_jobs, algorithm=algorithm)
        sc.fit(sampled_data)
        sampled_labels = sc.labels_

        if sample_fraction == 1:
            cluster_labels = sc.labels_
        else:
            # Train a classifier on the sampled data to predict cluster labels
            classifier = RandomForestClassifier(random_state=self._seed)
            classifier.fit(sampled_data, sampled_labels)

            # Predict cluster labels for the entire dataset
            cluster_labels = classifier.predict(concatenated_data)

        # Split the cluster labels back into their respective years
        start_idx = 0
        for year in years:
            end_idx = start_idx + len(self._data[year][model.__name__ + "_reduced"])
            self._data[year][model.__name__ + "_cluster"] = cluster_labels[start_idx:end_idx]
            start_idx = end_idx

        return sc
    
    def calc_clusters(self, years, model, algorithm='kmeans', **kwargs):
        """Calculates clusters for the features using the specified algorithm."""

        # Collect data from all specified years
        data_list = [self._data[year][model.__name__ + "_reduced"] for year in years]
        concatenated_data = pd.concat(data_list, axis=0)

        if algorithm == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 8)
            clusterer = KMeans(n_clusters=n_clusters, random_state=self._seed)
        elif algorithm == 'optics':
            min_samples = kwargs.get('min_samples', 5)
            max_eps = kwargs.get('max_eps', float('inf'))
            n_jobs = kwargs.get('n_jobs', -1)
            clusterer = OPTICS(min_samples=min_samples, max_eps=max_eps, n_jobs=n_jobs)
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            n_jobs = kwargs.get('n_jobs', -1)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        cluster_labels = clusterer.fit_predict(concatenated_data)

        # Split the cluster labels back into their respective years
        start_idx = 0
        for year in years:
            end_idx = start_idx + len(self._data[year][model.__name__ + "_reduced"])
            self._data[year][model.__name__ + "_cluster"] = cluster_labels[start_idx:end_idx]
            start_idx = end_idx

        return clusterer

    def plot_cluster_counts(self, years, model):
        """Plots the counts of each cluster for all years."""
        cluster_counts = self.get_cluster_counts(years, model)

        # Plot cluster distribution
        plt.figure(figsize=(12, 6))
        plt.bar(cluster_counts.keys(), cluster_counts.values())
        plt.xlabel("Cluster Number")
        plt.ylabel("Number of Images")
        plt.title("Cluster Distribution")
        plt.show()

    def get_cluster_counts(self, years, model):
        """Returns the counts of each cluster for all years."""
        cluster_counts = {}
        for year in years:

            for k, v in dict(Counter(self._data[year][model.__name__ + "_cluster"])).items():
                if k in cluster_counts:
                    cluster_counts[k] = cluster_counts[k] + v
                else:
                    cluster_counts[k] = v

        return cluster_counts

    def get_cluster_location_info(self, year, model, subgeometries, dissolve_by_cluster=False):
        """Returns information about all image clips stored in the Eloisa object."""
        cluster_location_info = pd.DataFrame()

        cluster_location_info["path"] = self.image_names[year]
        cluster_location_info["image_num"] = cluster_location_info["path"].str.extract(r"(\d+)\.tif")
        cluster_location_info["cluster"] = self._data[year][model.__name__ + "_cluster"]

        for i, subgeometry in enumerate(subgeometries):
            cluster_location_info.loc[cluster_location_info["image_num"] == str(i), "geometry"] = subgeometry

        cluster_location_info = gpd.GeoDataFrame(cluster_location_info, geometry="geometry")
        cluster_location_info.crs = "EPSG:4326"

        if dissolve_by_cluster:

            # Create a new Geodataframe where all rows with the same cluster are join together
            cluster_location_info = cluster_location_info.dissolve(by="cluster")

            # Drop image_num column
            cluster_location_info.drop(columns="image_num", inplace=True)

            # Make create a column with the cluster number
            cluster_location_info["cluster"] = cluster_location_info.index

        return cluster_location_info

        



