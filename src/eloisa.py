# type: ignore

import os, sys

from collections import defaultdict

from copy import deepcopy
from pprint import pprint

import rasterio
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

import keras
from keras.preprocessing import image # pylint: disable=import-error 

from tqdm import tqdm

import matplotlib.pyplot as plt


import sqlite3
import pickle

import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


sys.path.insert(1, os.path.join('src'))
import feature_extraction as fe # pylint: disable=wrong-import-position

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

        

    def close_db(self):
        self._database.close()

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

    def show_clip_by_year(self, years: list, index: int):
        """Displays the image clip for the specified years and index."""

        fig, axes = plt.subplots(1, len(years), figsize=(20, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(self._data[years[i]]["image_list"][index])
            ax.axis("off")
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

    def update_database(self):
        """Updates the database with up-to-date data from the Eloisa object."""
        for year in self._data:
            for model_name in self._data[year]:
                if model_name == "image_list" or model_name[-8:] == "_cluster" or model_name[-8:] == "_reduced":
                    continue

                features = self._data[year][model_name]

                

                for i, row in features.iterrows():

                    # Convert features to json
                    features_as_json = json.dumps(row.values.flatten().tolist())
                    features_reduced_as_json = json.dumps(self._data[year][model_name + "_reduced"].iloc[i].values.flatten().tolist() 
                                                          if model_name + "_reduced" in self._data[year] else None)
                    cluster_int = int(self._data[year][model_name + "_cluster"][i]) if model_name + "_cluster" in self._data[year] else None

                    self._database.execute('''UPDATE images
                        SET features = ?, features_reduced = ?, model_name = ?, cluster = ?
                        WHERE year = ? AND image_clip = ?''', 
                        (features_as_json, features_reduced_as_json, model_name, cluster_int, year, self.image_names[year][i]))

        self._database.commit()

    def load_all_features_from_db(self):
        """Loads all features from the Eloisa database."""
        cursor = self._database.cursor()
        cursor.execute('''
            SELECT year, image_clip, image, bands_used, model_name, features, features_reduced, cluster
            FROM images
        ''')

        for i, row in enumerate(tqdm(cursor.fetchall())):
            year, image_clip, image, bands_used, model_name, features, features_reduced, cluster = row

            if year not in self._data:
                self._data[year] = {}

            if 'image_list' not in self._data[year]:
                self._data[year]['image_list'] = []
                self.image_names[year] = []
            
            bands_used_list = json.loads(bands_used)

            with rasterio.open(image) as src:

                img = src.read(bands_used_list)  # Read the first three channels (RGB)
                img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)
                
                # Normalize to 0-255 and convert to uint8
                img = (img - img.min()) / (img.max() - img.min()) * 255
                img = img.astype(np.uint8)

                # Convert to Pillow image
                pil_img = keras.preprocessing.image.array_to_img(img, scale=False)
                pil_img = pil_img.resize((self.image_shape[0], self.image_shape[1]))  # Resize the image to the desired size

                self.image_names[year].append(image)

                self._data[year]["image_list"].insert(i, pil_img)

            if model_name not in self._data[year]:
                self._data[year][model_name] = pd.DataFrame(columns=np.arange(0, len(json.loads(features))))

            self._data[year][model_name].loc[i] = json.loads(features)


            if features_reduced is not None and features_reduced != 'null':

                if model_name + '_reduced' not in self._data[year]:
                    self._data[year][model_name + '_reduced'] = pd.DataFrame(columns=np.arange(0, len(json.loads(features_reduced))))

                self._data[year][model_name + '_reduced'].loc[i] = json.loads(features_reduced)

            if cluster is not None and cluster != 'null':

                if model_name + "_cluster" not in self._data[year]:
                    self._data[year][model_name + "_cluster"] = []
                
                self._data[year][model_name + "_cluster"].insert(i, cluster)

    def scale_features(self, year, model):
        """Scales the features using StandardScaler."""
        scaler = StandardScaler()
        self._data[year][model.__name__ + "_reduced"] = scaler.fit_transform(self._data[year][model.__name__])

    def pca_features(self, year, model, n_components=None, variance_min=None, plot_variance=False):
        """Performs PCA on the features."""

        if n_components is None and variance_min is None:
            raise ValueError("Either n_components or variance_min must be provided.")
        if n_components is not None and variance_min is not None:
            raise ValueError("Only one of n_components or variance_min can be provided.")

        if n_components is None:

            pca = PCA(random_state=self._seed)
            pca.fit(self._data[year][model.__name__ + "_reduced"])

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
        pca.fit(self._data[year][model.__name__])
        self._data[year][model.__name__ + "_reduced"] = pd.DataFrame(pca.transform(self._data[year][model.__name__ + "_reduced"]), index=self._data[year][model.__name__].index)

    def calc_silhouette_score(self, year, model, kmax=100, show_plot=True):
        """Calculates the silhouette score for different numbers of clusters."""

        sil = []

        for k in range(2, kmax+1):
            kmeans = KMeans(n_clusters=k, random_state=self._seed).fit(self._data[year][model.__name__ + "_reduced"])
            labels = kmeans.labels_
            sil.append(silhouette_score(self._data[year][model.__name__ + "_reduced"], labels, metric='euclidean'))

        if show_plot:
            # Plot silhouette score
            plt.figure(figsize=(10, 6))
            plt.plot(range(2, kmax+1), sil, marker='o')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score vs. Number of Clusters')
            plt.grid(True)
            plt.show()

        return sil
    
    def plot_elbow_curve(self, year, model, kmax=100):
        """Plots the elbow method for different numbers of clusters."""
        # Create elbow plot
        X = self._data[year][model.__name__ + "_reduced"]
        distorsions = []
        for k in range(2, kmax):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            distorsions.append(kmeans.inertia_)

        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(2, kmax), distorsions)
        plt.grid(True)
        plt.title('Elbow curve')

    def calc_kmeans_clusters(self, year, model, n_clusters):
        """Calculates the KMeans clusters for the features."""

        kmeans = KMeans(n_clusters=n_clusters, random_state=self._seed)
        self._data[year][model.__name__ + "_cluster"] = kmeans.fit_predict(self._data[year][model.__name__ + "_reduced"])