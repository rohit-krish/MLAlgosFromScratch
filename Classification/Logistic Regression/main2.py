'''
this implementation is for multi-class classification
'''
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, n_itrs=1000):
        self.lr = lr
        self.n_itrs = n_itrs
        self.weights = None
        self.bias = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros((len(self.classes), n_features))
        self.bias = np.zeros(len(self.classes))

        for class_idx, class_label in enumerate(self.classes):
            # Convert the problem into a binary classification task
            binary_y = np.where(y == class_label, 1, 0)

            # Gradient descent
            for _ in range(self.n_itrs):
                linear_model = np.dot(X, self.weights[class_idx]) + self.bias[class_idx]
                y_predicted = self._sigmoid(linear_model)

                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - binary_y))
                db = (1 / n_samples) * np.sum(y_predicted - binary_y)

                self.weights[class_idx] -= self.lr * dw
                self.bias[class_idx] -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights.T) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # print(y_predicted.shape)
        predicted_classes = np.argmax(y_predicted, axis=1)
        return predicted_classes

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
