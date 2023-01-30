import numpy as np
from math import isclose


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iter=500) -> None:
        self.lr = learning_rate
        self.n_iter = n_iter
        # each coefficients * features(indipendent variable)
        self.weights = None
        self.bias = None        # intercept

    def mean_squared_error(self, y, y_predicted):  # cost func
        return np.mean((y - y_predicted) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        cost_previous = 0

        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent algo
        for _ in range(self.n_iter):
            y_predicted = np.dot(X, self.weights) + self.bias

            cost = self.mean_squared_error(y, y_predicted)

            # partial derivatives
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # X.T is the transpose of X
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if isclose(cost, cost_previous, rel_tol=1e-18):
                self.coefficients_ = self.weights
                self.intercept_ = self.bias
                break

            cost_previous = cost

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

