#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 青年有志
# @time    : 2023-03-30 22:13
# @function: the script is used to do something.
# @version : V1

from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

iris = datasets.load_iris()
data = iris['data'][:,:2]

loss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=100).fit(data)
    loss.append(kmeans.inertia_ / len(data) / 3)

plt.title('K with loss')
plt.plot(range(1, 10), loss)
plt.show()
