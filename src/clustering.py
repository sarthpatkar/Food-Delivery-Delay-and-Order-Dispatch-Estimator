import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE


class DeliveryClusterAnalyzer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        self.pca = PCA(n_components=2, random_state=RANDOM_STATE)
        self.elbow_data = []
        self.cluster_labels = None
        self.cluster_profiles = []
        self.pca_coordinates = None
        self.explained_variance = []

    def compute_elbow_curve(self, X_scaled, k_range=range(2, 9)):
        sample_size = min(6000, len(X_scaled))
        X_sample = X_scaled.iloc[:sample_size] if hasattr(X_scaled, "iloc") else X_scaled[:sample_size]

        elbow_results = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
            km.fit(X_sample)
            elbow_results.append({"k": int(k), "inertia": round(float(km.inertia_), 2)})
        self.elbow_data = elbow_results
        return elbow_results

    def fit_clusters_and_pca(self, X_scaled, raw_cleaned_df):
        sample_size = min(6000, len(X_scaled))
        X_sample = X_scaled.iloc[:sample_size] if hasattr(X_scaled, "iloc") else X_scaled[:sample_size]
        raw_sample = raw_cleaned_df.iloc[:sample_size].copy()

        self.kmeans.fit(X_sample)
        self.cluster_labels = self.kmeans.labels_

        pca_transformed = self.pca.fit_transform(X_sample)
        self.explained_variance = [
            round(float(v) * 100, 2) for v in self.pca.explained_variance_ratio_
        ]

        raw_sample["cluster"] = self.cluster_labels
        raw_sample["pca_x"] = np.round(pca_transformed[:, 0], 3)
        raw_sample["pca_y"] = np.round(pca_transformed[:, 1], 3)

        profiles = []
        cluster_names = {
            0: "Express Transit Corridors",
            1: "Standard Urban Delivery",
            2: "Congested High-Delay Zones",
        }

        for c_id in range(self.n_clusters):
            subset = raw_sample[raw_sample["cluster"] == c_id]
            profiles.append({
                "cluster_id": int(c_id),
                "name": cluster_names.get(c_id, f"Zone Cluster {c_id}"),
                "order_count": int(len(subset)),
                "avg_delivery_time": round(float(subset["time_taken_min"].mean()), 1),
                "avg_distance_km": round(float(subset["distance_km"].mean()), 1),
                "avg_rating": round(float(subset["delivery_person_ratings"].mean()), 2),
            })

        self.cluster_profiles = profiles

        plot_sample = raw_sample.sample(min(800, len(raw_sample)), random_state=RANDOM_STATE)
        self.pca_coordinates = [
            {
                "x": float(row["pca_x"]),
                "y": float(row["pca_y"]),
                "cluster": int(row["cluster"]),
                "time": float(row["time_taken_min"]),
                "distance": float(row["distance_km"]),
            }
            for _, row in plot_sample.iterrows()
        ]

        return {
            "elbow_data": self.elbow_data,
            "profiles": self.cluster_profiles,
            "explained_variance": self.explained_variance,
            "pca_coordinates": self.pca_coordinates,
        }
