import numpy as np
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
        X_sample = X_scaled.iloc[:6000]
        elbow_results = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
            km.fit(X_sample)
            elbow_results.append({"k": int(k), "inertia": round(float(km.inertia_), 2)})
        self.elbow_data = elbow_results
        return elbow_results

    def fit_clusters_and_pca(self, X_scaled, raw_cleaned_df):
        X_sample = X_scaled.iloc[:6000]
        raw_sample = raw_cleaned_df.loc[X_sample.index].copy()

        self.kmeans.fit(X_sample)
        self.cluster_labels = self.kmeans.labels_

        pca_transformed = self.pca.fit_transform(X_sample)
        self.explained_variance = [
            round(float(v) * 100, 2) for v in self.pca.explained_variance_ratio_
        ]

        raw_sample["cluster"] = self.cluster_labels
        raw_sample["pca_x"] = np.round(pca_transformed[:, 0], 3)
        raw_sample["pca_y"] = np.round(pca_transformed[:, 1], 3)

        # K-Means label order is arbitrary, so name clusters by how slow they are
        names = ["Short, quick runs", "Typical orders", "Long or congested runs"]
        by_speed = raw_sample.groupby("cluster")["time_taken_min"].mean().sort_values().index
        profiles = []
        for rank, c_id in enumerate(by_speed):
            subset = raw_sample[raw_sample["cluster"] == c_id]
            profiles.append({
                "cluster_id": int(c_id),
                "name": names[rank] if rank < len(names) else f"Cluster {c_id}",
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
