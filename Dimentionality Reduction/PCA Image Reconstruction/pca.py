import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X-self.mean

        # covariance
        # in X row is sample && column in each feature { here we taking transpose is because in the documentation of np.cov it is the other way around ie row is feature and column is sample}
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
        self.components = eigen_vectors[:self.n_components]
        print(self.components.shape)
        print(self.mean.shape, 'mean')

    def transform(self, X):
        # project data
        X = X - self.mean  # in the eqn of the covariance
        res = np.dot(X, self.components.T)
        self.new_mean = np.mean(res, axis=0)
        return res

    def inverse_transform(self, reduced_X):
        reduced_X = reduced_X - self.new_mean
        return np.dot(reduced_X, self.components)
