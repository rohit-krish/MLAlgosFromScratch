import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from main2 import LogisticRegression

bc = datasets.load_digits()

X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


reg = LogisticRegression()
reg.fit(X_train, y_train)

preds = reg.predict(X_test)

print(accuracy(y_test, preds))
