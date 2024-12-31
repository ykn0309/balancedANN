'''
Author: Kainan Yang ykn0309@whu.edu.cn
Date: 2024-12-30 15:50:40
LastEditors: Kainan Yang ykn0309@whu.edu.cn
LastEditTime: 2024-12-31 17:16:02
FilePath: /balanceANN/ANN.py
'''
import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from Kmeans import Kmeans
from hyperparameters import *

class ANN:
    def __init__(self, X, x):
        self.X = X # data
        self.x = x # query point
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.k = self.n // KDTREE_NODE_SIZE + 1 # cluster number
        self.labels = np.zeros(self.n, dtype=int)
        self.centroids = np.zeros((self.k, self.d)) # centriud matrix
        self.kdtree:KDTree = None

    def iNN(self):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.X)
        _, indices = nbrs.kneighbors([self.x])
        return indices[0][0]
    
    def process_labels(self, labels, cluster_size):
        for i in range(self.k):
            if cluster_size[i] == 0:
                labels -= 1
        return labels

    def lloyd_kmeans(self):
        kmeans = Kmeans(self.X, self.k)
        labels, centroids = kmeans.run_lloyd_kmeans()
        cluster_size = np.bincount(labels, minlength=self.k)
        self.labels = self.process_labels(labels, cluster_size)
        non_empty_idx = np.where(cluster_size > 0)[0]
        self.centroids = centroids[non_empty_idx]
        
    def balanced_kmeans(self):
        kmeans = Kmeans(self.X, self.k)
        labels, centroids = kmeans.run_balanced_kmeans()
        cluster_size = np.bincount(labels, minlength=self.k)
        self.labels = self.process_labels(labels, cluster_size)
        non_empty_idx = np.where(cluster_size > 0)[0]
        self.centroids = centroids[non_empty_idx]
        
    def lloyd_ANN(self):
        self.lloyd_kmeans()
        self.kdtree = KDTree(self.centroids)
        _, indices = self.kdtree.query([self.x], k=1)
        cluster_idx = indices[0][0]
        XX = self.X[self.labels == cluster_idx]
        distances = euclidean_distances([self.x], XX)[0]
        nn = np.argmin(distances)
        idx = np.where(np.all(self.X == XX[nn], axis=1))[0][0]
        print('lloyd', idx)
        return idx, XX[nn], distances[nn]

    def balanced_ANN(self):
        self.balanced_kmeans()
        self.kdtree = KDTree(self.centroids)
        _, indices = self.kdtree.query([self.x], k=1)
        cluster_idx = indices[0][0]
        XX = self.X[self.labels == cluster_idx]
        distances = euclidean_distances([self.x], XX)[0]
        nn = np.argmin(distances)
        idx = np.where(np.all(self.X == XX[nn], axis=1))[0][0]
        print('balanced', idx)
        return idx, XX[nn], distances[nn]