#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 青年有志
# @time    : 2023-04-03 15:52
# @function: the script is used to do something.
# @version : V1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data = loadmat('data/ex7data1.mat')

# data
X = data['X']

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()

def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

U, S, V = pca(X)

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

# 通过 pca 降维后，投影到低维空间
Z = project_data(X, U, 1)


# 降维后的结果
print(Z)

# 可以通过反向转换步骤来恢复原始数据。
def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z, U, 1)

# 可视化恢复后的数据
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()