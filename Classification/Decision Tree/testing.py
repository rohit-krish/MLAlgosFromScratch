import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from main import DecisionTree
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
import matplotlib.pyplot as plt

def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = load_breast_cancer()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

clf = DecisionTree(max_depth=10)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

acc = accuracy(y_test,y_pred)

print('Accuracy: ',acc)

cmap = confusion_matrix(y_test,y_pred)
heatmap(cmap,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
