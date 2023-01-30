import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target


def pca(X, n_components):
    X = X - np.mean(X, axis=0)
    cov = np.cov(X.T)

    eig_vals, eig_vecs = np.linalg.eig(cov)
    eig_vecs = eig_vecs.T

    idxs = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idxs]
    eig_vecs = eig_vecs[idxs]

    components = eig_vecs[:n_components]

    return np.dot(X, components.T)


def plot2d():
    X_projected = pca(X, 2)

    print('Shape of X:', X.shape)
    print('Shape of transormed X:', X_projected.shape)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2, c=y, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


def plot3d():
    X_projected = pca(X, 3)

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
