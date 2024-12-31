'''
Author: Kainan Yang ykn0309@whu.edu.cn
Date: 2024-12-30 11:59:11
LastEditors: Kainan Yang ykn0309@whu.edu.cn
LastEditTime: 2024-12-31 16:15:01
FilePath: /balancedANN/Kmeans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Kainan Yang ykn0309@whu.edu.cn
Date: 2024-12-30 11:59:11
LastEditors: Kainan Yang ykn0309@whu.edu.cn
LastEditTime: 2024-12-31 12:01:07
FilePath: /balanceANN/Kmeans.py
'''
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from hyperparameters import *

class Kmeans:
    def __init__(self, X, k):
        self.max_iterations = MAX_ITER
        self.w = PENTALTY_WEIGHT # weight of balanced pentalty
        self.X = X # data
        self.k = k # number of clusters
        self.n = X.shape[0] # number of samples
        self.d = X.shape[1] # demension
        self.centroids = self.init_centroids() # centroids
        self.balaced_cluster_size = self.n // k # balanced cluster size
        self.labels = np.zeros(self.n, dtype=int)

    def init_centroids(self):
        centroids = self.X[np.random.choice(self.n, self.k, replace=False)]
        return centroids

    def lloyd_assign(self):
        distances = euclidean_distances(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)
        
    def balanced_assign(self):
        distances = euclidean_distances(self.X, self.centroids)
        cluster_sizes = np.bincount(self.labels, minlength=self.k)
        pentalties = self.w * (cluster_sizes - self.balaced_cluster_size) ** 2
        for i in range(self.k):
            distances[:, i] += pentalties[i]
        self.labels = np.argmin(distances, axis=1)

    def refine(self):
        for i in range(self.k):
            self.centroids[i] = np.array(np.mean(self.X[self.labels == i], axis=0) if np.any(self.labels == i) else self.centroids[i])

    def run_lloyd_kmeans(self):
        # for i in range(self.max_iterations):
        #     old_labels = self.labels.copy()
        #     self.lloyd_assign()
        #     self.refine()
        #     if np.all(old_labels == self.labels):
        #         break
        kmeans = KMeans(n_clusters=self.k, random_state=42)
        kmeans.fit(self.X)
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_
        return self.labels, self.centroids
    
    def run_balanced_kmeans(self):
        self.run_lloyd_kmeans()
        for i in range(self.max_iterations):
            old_labels = self.labels.copy()
            self.balanced_assign()
            self.refine()
            if np.all(old_labels == self.labels):
                break
        return self.labels, self.centroids