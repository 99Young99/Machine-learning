#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 青年有志
# @time    : 2023-03-30 22:05
# @function: the script is used to do something.
# @version : V1

# using sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
data = iris['data'][:,:2]
kmeans = KMeans(n_clusters=3, max_iter=100).fit(data)
gt_labels__ = kmeans.labels_
centers__ = kmeans.cluster_centers_

print(gt_labels__)

print(centers__)

# visualize result

cat1 = data[gt_labels__ == 0]
cat2 = data[gt_labels__ == 1]
cat3 = data[gt_labels__ == 2]

for ix, p in enumerate(centers__):
    plt.scatter(p[0], p[1], color='C{}'.format(ix), marker='^',
                edgecolor='black', s=256)

plt.scatter(cat1[:, 0], cat1[:, 1], color='green')
plt.scatter(cat2[:, 0], cat2[:, 1], color='red')
plt.scatter(cat3[:, 0], cat3[:, 1], color='blue')
plt.title('kmeans using sklearn with k=3')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()