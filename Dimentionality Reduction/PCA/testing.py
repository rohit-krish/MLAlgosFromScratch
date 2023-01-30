from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from main import PCA

data = load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)
print('Shape of X:', X.shape)
print('Shape of transormed X', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolors='none', alpha=1,
    # cmap=plt.cm.get_cmap('viridis', 3)
    cmap='viridis'
)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
