import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self.__predict(x) for x in X]
        return np.array(predicted_labels)
    

    def __predict(self, x):
        # compute distances
        distances = [
            self.__euclidean_distance(x, x_train) for x_train in self.X_train
        ]

        # get k nearest neighbors/samples , labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote [most common class label]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def __euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
