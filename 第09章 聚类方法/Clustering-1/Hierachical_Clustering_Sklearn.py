#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 青年有志
# @time    : 2023-03-30 21:27
# @function: sklearn 中实现的层次聚类
# @version : V1


import numpy as np
from sklearn import datasets,cluster
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris['data'][:,:2]


sk = cluster.AgglomerativeClustering(3)
sk.fit(data)
labels_ = sk.labels_
print(labels_)

# visualize result of sklearn
cat1_ = data[np.where(labels_==0)]
cat2_ = data[np.where(labels_==1)]
cat3_ = data[np.where(labels_==2)]

plt.scatter(cat1_[:,0], cat1_[:,1], color='green')
plt.scatter(cat2_[:,0], cat2_[:,1], color='red')
plt.scatter(cat3_[:,0], cat3_[:,1], color='blue')
plt.title('Hierarchical clustering with k=3')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()
