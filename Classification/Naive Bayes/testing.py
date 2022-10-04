import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from main import NaiveBayes

def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X,y = make_classification(n_samples=1000,n_features=10,n_classes=2)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = NaiveBayes()
model.fit(X_train,y_train)
preds = model.predict(X_test)

print('Accuracy:',accuracy(y_test,preds))