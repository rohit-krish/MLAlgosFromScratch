'''There is also QDA'''

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sb

data = load_iris()
X = data.data
y = data.target


def lda(X, y, n_components):
    n_features = X.shape[1]
    class_labels = np.unique(y)

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

    linear_discriminants = eig_vecs[:n_components]

    return np.dot(X, linear_discriminants.T)


def plot2d():
    X_projected = lda(X, y, 2)

    print('Shape of X:', X.shape)
    print('Shape of transormed X:', X_projected.shape)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2, c=y, cmap='viridis')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.show()


def plot3d():
    X_projected = lda(X, y, 3)

    print('Shape of X:', X.shape)
    print('Shape of transormed X:', X_projected.shape)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    x3 = X_projected[:, 2]

    ax = plt.axes(projection='3d')
    ax.set(
        xlabel='x-axis', ylabel='y-axis', zlabel='z-axis'
    )

    # plt.plot(x1, x2, x3)
    ax.scatter3D(x1, x2, x3, c=y, cmap='viridis')
    plt.show()


plot2d()
plot3d()
