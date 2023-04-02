#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 青年有志
# @time    : 2023-03-30 21:26
# @function: kmeans
# @version : V1

import math
import random
import numpy as np
from sklearn import datasets,cluster
import matplotlib.pyplot as plt

# kmeans
class MyKmeans:
    def __init__(self, k, n=20):
        self.k = k
        self.n = n

    def fit(self, x, centers=None):
        # 第一步，随机选择 K 个点, 或者指定
        if centers is None:
            idx = np.random.randint(low=0, high=len(x), size=self.k)
            centers = x[idx]
        # print(centers)

        inters = 0
        while inters < self.n:
            # print(inters)
            # print(centers)
            points_set = {key: [] for key in range(self.k)}

            # 第二步，遍历所有点 P，将 P 放入最近的聚类中心的集合中
            for p in x:
                nearest_index = np.argmin(
                    np.sum((centers - p) ** 2, axis=1) ** 0.5)
                points_set[nearest_index].append(p)

            # 第三步，遍历每一个点集，计算新的聚类中心
            for i_k in range(self.k):
                centers[i_k] = sum(points_set[i_k]) / len(points_set[i_k])

            inters += 1

        return points_set, centers


iris = datasets.load_iris()
data = iris['data'][:,:2]

m = MyKmeans(3)
points_set, centers = m.fit(data)

print(centers)

# visualize result
cat1 = np.asarray(points_set[0])
cat2 = np.asarray(points_set[1])
cat3 = np.asarray(points_set[2])

for ix, p in enumerate(centers):
    plt.scatter(p[0], p[1], color='C{}'.format(ix), marker='^',
                edgecolor='black', s=256)

plt.scatter(cat1[:, 0], cat1[:, 1], color='green')
plt.scatter(cat2[:, 0], cat2[:, 1], color='red')
plt.scatter(cat3[:, 0], cat3[:, 1], color='blue')
plt.title('Hierarchical clustering with k=3')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()