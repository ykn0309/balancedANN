'''
Author: Kainan Yang ykn0309@whu.edu.cn
Date: 2024-12-30 17:15:19
LastEditors: Kainan Yang ykn0309@whu.edu.cn
LastEditTime: 2024-12-30 20:52:52
FilePath: /balanceANN/main.py
'''
import numpy as np
from ANN import ANN

def run(X, Y):
    query_num = Y.shape[0]
    lloyd_success_count = 0
    balanced_success_count = 0
    for i in range(query_num):
        x = Y[i]
        ann = ANN(X, x)
        true_idx = ann.iNN()
        lloyd_idx, _, _ = ann.lloyd_ANN()
        balanced_idx, _, _ = ann.balanced_ANN()
        if lloyd_idx == true_idx:
            lloyd_success_count += 1
        if balanced_idx == true_idx:
            balanced_success_count += 1
    lloyd_precision = lloyd_success_count / query_num
    balanced_precision = balanced_success_count / query_num
    return lloyd_precision, balanced_precision

if __name__ == '__main__':
    X = np.random.rand(100, 2)  # 随机生成数据
    query_point = np.array([0.5, 0.5])  # 查询点

    ann = ANN(X, query_point)
    idx, nearest_point, distance = ann.lloyd_ANN()
    print(f"Lloyd: {nearest_point}, 距离: {distance}")
    
    idx, nearest_point, distance = ann.balanced_ANN()
    print(f"Balanced: {nearest_point}, 距离: {distance}")