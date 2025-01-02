'''
Author: Kainan Yang ykn0309@whu.edu.cn
Date: 2024-12-30 17:15:19
LastEditors: Kainan Yang ykn0309@whu.edu.cn
LastEditTime: 2025-01-02 10:32:05
FilePath: /balanceANN/main.py
'''
import numpy as np
from ANN import ANN

def run(X, Y):
    query_num = Y.shape[0]
    lloyd_success_count = 0
    balanced_success_count = 0
    ann = ANN(X)
    for i in range(query_num):
        x = Y[i]
        true_idx = sift_groundtruth[i][0]
        lloyd_idx, _, _ = ann.lloyd_ANN(x)
        balanced_idx, _, _ = ann.balanced_ANN(x)
        if lloyd_idx == true_idx:
            lloyd_success_count += 1
        if balanced_idx == true_idx:
            balanced_success_count += 1
    print(lloyd_success_count, balanced_success_count)
    lloyd_precision = lloyd_success_count / query_num
    balanced_precision = balanced_success_count / query_num
    return lloyd_precision, balanced_precision

if __name__ == '__main__':
    sift_base = np.fromfile('./siftsmall/siftsmall_base.fvecs', dtype=np.float32)
    sift_base = sift_base.reshape(-1, 129)[:, 1:]
    sift_query = np.fromfile('./siftsmall/siftsmall_query.fvecs', dtype=np.float32)
    sift_query = sift_query.reshape(-1, 129)[:, 1:]
    sift_groundtruth = np.fromfile('./siftsmall/siftsmall_groundtruth.ivecs', dtype=np.int32)
    sift_groundtruth = sift_groundtruth.reshape(-1, 101)[:, 1:]
    lloyd_precision, balanced_precision = run(sift_base, sift_query)
    print(lloyd_precision, balanced_precision)