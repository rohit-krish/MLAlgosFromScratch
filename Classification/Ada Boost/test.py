import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from main import AdaBoost


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    n_correct = len(y_true[y_true == y_pred])
    n_incorrect = len(y_true[y_true != y_pred])
    return acc, n_correct, n_incorrect


data = datasets.load_breast_cancer()
X = data.data
y = data.target

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True
)

clf = AdaBoost(n_clf=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

res = accuracy(y_test, y_pred)
print(f"Accuracy: {res[0]} , n_correct: {res[1]} , n_incorrect: {res[2]}")
