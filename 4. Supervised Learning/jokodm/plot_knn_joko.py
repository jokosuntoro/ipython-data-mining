import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

from .plot_helpers import discrete_scatter


def plot_knn_joko(n_neighbors=1):
    X = np.array([[1.1, 1.1], [2.1, 6.1], [2.3, 2.5], 
                  [3.4, 1.2], [4.8, 4.6], [4.2, 5.1], 
                  [8.5, 4.7], [9.3, 3.9], [2.4, 7.1],
                  [3.5, 7.3], [6.2, 9.4], [7.5, 8.1], 
                  [7.6, 7.2], [8.1, 8.3], [9.1, 7.3], 
                  [9.8, 9.8], [9.9, 6.2]])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    X_test = np.array([[3.5, 6.5], [9.5, 5.5]])
    dist = euclidean_distances(X, X_test)
    closest = np.argsort(dist, axis=0)

    for x, neighbors in zip(X_test, closest.T):
        for neighbor in neighbors[:n_neighbors]:
            plt.arrow(x[0], x[1], X[neighbor, 0] - x[0],
                      X[neighbor, 1] - x[1], head_width=0, fc='k', ec='k')

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    test_points = discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers="*")
    training_points = discrete_scatter(X[:, 0], X[:, 1], y)