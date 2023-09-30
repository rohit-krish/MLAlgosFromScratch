#%%
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from main import KMeans

X, y = make_blobs(
    centers=4, n_samples=500, n_features=2, shuffle=True,
    random_state=42
)
plt.scatter(X[:,0],X[:,1])
#%%
clusters = len(np.unique(y))

model = KMeans(K=clusters, max_iters=150, plot_steps=False)
y_pred = model.predict(X)

#%%
y_pred = y_pred.astype('int8')
plt.scatter(X[:,0],X[:,1],c=y_pred,cmap='viridis')
