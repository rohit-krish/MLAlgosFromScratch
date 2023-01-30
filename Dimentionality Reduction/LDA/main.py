import numpy as np


class LDA:
    '''
    LDA assumes that each class follow a Gaussian distribution.
    '''

    def __init__(self, n_components) -> None:
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # S_W, S_B
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros_like(S_W)

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * mean_diff.dot(mean_diff.T)

        A = np.linalg.inv(S_W).dot(S_B)
        eig_vals, eig_vecs = np.linalg.eig(A)
        eig_vecs = eig_vecs.T

        idxs = np.argsort(abs(eig_vals))[::-1]
        eig_vals = eig_vals[idxs]
        eig_vecs = eig_vecs[idxs]

        self.linear_discriminants = eig_vecs[:self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)
