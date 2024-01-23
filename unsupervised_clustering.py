import pandas as pd
from sklearn.cluster import KMeans


class UnsupervisedClustering:
    def __init__(self, data):
        self._data = data.copy()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data.copy()

    def unsupervised_cluster(self, n_clusters):
        cluster = KMeans(n_clusters=n_clusters)
        labels = cluster.fit_predict(self.data)
        self.data['Cluster'] = labels
        return labels

    def apply_rules(self, columns, weights):
        cluster_averages = [self.calculate_weighted_average(
            cluster_label, columns, weights) for cluster_label in self.data['Cluster'].unique()]
        best_cluster_index = cluster_averages.index(max(cluster_averages))
        return cluster_averages, best_cluster_index

    @staticmethod
    def calculate_weighted_average(cluster_label, data, columns, weights):
        cluster_data = data[data['Cluster'] == cluster_label]
        return sum(cluster_data[column].mean() * weight for column, weight in zip(columns, weights))

    def use_cluster(self, cluster_label):
        self.data = self.data[self.data['Cluster'] == cluster_label].copy()

    def __str__(self):
        return f"UnsupervisedClustering with {self.data.shape[0]} data points and {self.data['Cluster'].nunique()} clusters"
