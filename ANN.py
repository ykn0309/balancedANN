'''
Author: Kainan Yang ykn0309@whu.edu.cn
Date: 2024-12-30 15:50:40
LastEditors: Kainan Yang ykn0309@whu.edu.cn
LastEditTime: 2025-01-02 15:01:56
FilePath: /balanceANN/ANN.py
'''
import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from Kmeans import Kmeans
from hyperparameters import *

class ANN:
    def __init__(self, X):
        self.X = X # data
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.k = self.n // CLUSTER_NODE_SIZE + 1 # cluster number
        self.lloyd_labels = np.zeros(self.n, dtype=int)
        self.balanced_labels = np.zeros(self.n, dtype=int)
        self.lloyd_centroids = np.zeros((self.k, self.d)) # Lloyd centriod matrix
        self.balanced_centroids = np.zeros((self.k, self.d)) # balanced centriod matrix
        self.lloyd_kdtree:KDTree = None
        self.balanced_kdtree:KDTree = None

    def iNN(self, x):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.X)
        _, indices = nbrs.kneighbors([x])
        return indices[0][0]
    
    def process_labels(self, labels, cluster_size):
        # 创建一个映射表
        new_labels_map = {}
        new_label = 0
        for i in range(len(cluster_size)):
            if cluster_size[i] > 0:
                new_labels_map[i] = new_label
                new_label += 1
            else:
                new_labels_map[i] = -1  # 标记空 cluster

        # 使用映射表更新 labels
        processed_labels = np.vectorize(new_labels_map.get)(labels)
        return processed_labels

    def lloyd_kmeans(self):
        kmeans = Kmeans(self.X, self.k)
        labels, centroids = kmeans.run_lloyd_kmeans()
        cluster_size = np.bincount(labels, minlength=self.k)
        self.lloyd_labels = self.process_labels(labels, cluster_size)
        non_empty_idx = np.where(cluster_size > 0)[0]
        self.lloyd_centroids = centroids[non_empty_idx]
        self.lloyd_kdtree = KDTree(self.lloyd_centroids)
        
    def balanced_kmeans(self):
        kmeans = Kmeans(self.X, self.k)
        labels, centroids = kmeans.run_balanced_kmeans()
        cluster_size = np.bincount(labels, minlength=self.k)
        self.balanced_labels = self.process_labels(labels, cluster_size)
        non_empty_idx = np.where(cluster_size > 0)[0]
        self.balanced_centroids = centroids[non_empty_idx]
        self.balanced_kdtree = KDTree(self.balanced_centroids)
        
    def lloyd_ANN(self, x):
        if self.lloyd_kdtree == None:
            self.lloyd_kmeans()
        _, indices = self.lloyd_kdtree.query([x], k=1)
        cluster_idx = indices[0][0]
        XX = self.X[self.lloyd_labels == cluster_idx]
        distances = euclidean_distances([x], XX)[0]
        nn = np.argmin(distances)
        idx = np.where(np.all(self.X == XX[nn], axis=1))[0][0]
        print('lloyd', idx)
        return idx, XX[nn], distances[nn]

    def balanced_ANN(self, x):
        if self.balanced_kdtree == None:
            self.balanced_kmeans()
        _, indices = self.balanced_kdtree.query([x], k=1)
        cluster_idx = indices[0][0]
        XX = self.X[self.balanced_labels == cluster_idx]
        distances = euclidean_distances([x], XX)[0]
        nn = np.argmin(distances)
        idx = np.where(np.all(self.X == XX[nn], axis=1))[0][0]
        print('balanced', idx)
        return idx, XX[nn], distances[nn]