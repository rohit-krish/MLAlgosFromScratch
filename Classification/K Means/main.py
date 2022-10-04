import numpy as np


def eucledean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False) -> None:
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # intialize centroids
        random_sample_idxs = np.random.choice(
            self.n_samples, self.K, replace=False)

        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

        # return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [eucledean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _is_converged(self, centroids_old, centroids_new):
        distances = [eucledean_distance(
            centroids_old[i], centroids_new[i]) for i in range(self.K)]
        return sum(distances) == 0
