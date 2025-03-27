import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn import datasets
from sklearn.decomposition import PCA 
import numpy as np
import cv2

iris = datasets.load_iris()
#X = iris.data[:,:2]
y = iris.target

#print('feature matrix size', X.shape)
print('label vector size', y.shape)

#x_min, x_max = X[:,0].min()-0.5,X[:,0].max()+0.5
#y_min, y_max = X[:,1].min()-0.5,X[:,1].max()+0.5
'''plt.figure(2,figsize=(8,8))
plt.clf()
plt.scatter(X[:,0],X[:,1],c=y,cmap = plt.cm.Set1,edgecolors='k')
plt.xlabel('Sepal lenght')
plt.ylabel('Sepal width')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())
plt.show()'''

X_ = iris.data[:, :3]
print('feature matrix size', X_.shape)

fig = plt.figure(1, figsize=(4, 3))
plt.clf()

ax = fig.add_subplot(121, projection="3d", elev=48, azim=134)
ax.set_position([0.45, 0, 0.45, 1])
plt.cla()

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X_[y == label, 0].mean(),
        X_[y == label, 1].mean() + 1.5,
        X_[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )

y = np.choose(y, [1, 2, 0]).astype(float)

ax.scatter(X_[:, 0], X_[:, 1], X_[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")
ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
ax.set_zlabel("Petal length")

X = iris.data
pca = PCA(n_components=3)
pca.fit(X)
X_red = pca.transform(X)

print("Dimensionality reduction applied to X: ", X_red)
print("Percentage of variance explained by each of the selected components: ", pca.explained_variance_ratio_)

fig = plt.figure(2, figsize=(4, 3))
plt.clf()

ax2 = fig.add_subplot(111, projection="3d")
ax2.set_position([0.45, 0, 0.45, 1])

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax2.text3D(
        X_red[y == label, 0].mean(),
        X_red[y == label, 1].mean() + 1.5,
        X_red[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )

ax2.scatter(X_red[:, 0], X_red[:, 1], X_red[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")

plt.show()