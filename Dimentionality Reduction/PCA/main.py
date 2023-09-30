"""
PCA intrestes in finding the component axes that maximize the variance of the data.
"""
import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)

        eigen_values, eigen_vectors = np.linalg.eig(cov)
        eigen_vectors = eigen_vectors.T

        # sort eigenvectors
        idxs = np.argsort(eigen_values)[::-1]
        # capturing most important eigen values
        eigen_values = eigen_values[idxs]
        # eigen vectors corresponding eigen values
        eigen_vectors = eigen_vectors[idxs]

        # store first n eigenvectors
        self.components = eigen_vectors[: self.n_components]
        print(self.components.shape)
        print(self.mean.shape, "mean")

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
