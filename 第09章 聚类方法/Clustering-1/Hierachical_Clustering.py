#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 青年有志
# @time    : 2023-03-30 16:46
# @function: 自定义实现层次聚类算法，缺点为没有将数据归一化
# @version : V1

import math
import random
import numpy as np
from sklearn import datasets,cluster
import matplotlib.pyplot as plt

# 定义聚类数的节点
class ClusterNode:
    def __init__(self, vec, left=None, right=None, distance=-1, id=None, count=1):
        """
        :param vec: 保存两个数据聚类后形成新的中心
        :param left: 左节点
        :param right:  右节点
        :param distance: 两个节点的距离
        :param id: 用来标记哪些节点是计算过的
        :param count: 记录叶子节点的个数
        """
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count

# 定义计算欧拉距离的函数
def euler_distance(point1: np.ndarray, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)

# 层次聚类（聚合法）
# https://zhuanlan.zhihu.com/p/32438294
class Hierarchical:
    def __init__(self, k):
        self.k = k
        self.labels = None

    def fit(self, x):

        # 将每个样本划分为每一个类，并为每个类进行编号
        nodes = [ClusterNode(vec=v, id=i) for i, v in enumerate(x)]

        # 存储两两的类距离
        distances = {}
        point_num, feature_num = x.shape

        # 初始化记录每个样本的类别
        self.labels = [-1] * point_num

        currentclustid = -1

        # 开始层次聚类算法
        while (len(nodes)) > self.k:
            min_dist = math.inf
            nodes_len = len(nodes)
            closest_part = None

            ## 计算两两类之间的类距离，保存到 distances 中
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):

                    # 提取两个类的编号,构成元组
                    d_key = (nodes[i].id, nodes[j].id)

                    # 计算两个类之间的欧式距离， key 保存两个类的编号， value 为距离
                    if d_key not in distances:
                        distances[d_key] = euler_distance(nodes[i].vec, nodes[j].vec)

                    # 更新当前类 i 与其他类的距离，保存最小的距离与索引
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)

            # 获取距离最小的两个类的索引
            part1, part2 = closest_part

            # 获取对应的类
            node1, node2 = nodes[part1], nodes[part2]

            # 取两个类的平均值作为当前合并后的类的中心点
            new_vec = [(node1.vec[i] * node1.count + node2.vec[i] * node2.count) / (node1.count + node2.count) for i in range(feature_num)]

            # 创建合并后的新类
            new_node = ClusterNode(vec=new_vec,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   id=currentclustid,
                                   count=node1.count + node2.count)
            currentclustid -= 1

            # 计算后将节点删除
            del nodes[part2], nodes[part1]

            # 将新类加入到节点集合中
            nodes.append(new_node)

        self.nodes = nodes

        # 获取新的节点集合 self.nodes ，对所有节点进行重新编号
        self.calc_label()

    def calc_label(self):
        """
        调取聚类的结果
        """
        for i, node in enumerate(self.nodes):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    # 递归对相同的类进行编号，相同的类 label 相同
    def leaf_traversal(self, node: ClusterNode, label):
        """
        递归遍历叶子节点
        """
        if node.left == None and node.right == None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)

iris = datasets.load_iris()
data = iris['data'][:,:2]

# 原始样本点
x = data[:,0]
y = data[:,1]
plt.scatter(x, y, color='green')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()

my = Hierarchical(3)
my.fit(data)

# 每个样本类
labels = np.array(my.labels)
print(labels)


# visualize result
cat1 = data[np.where(labels==0)]
cat2 = data[np.where(labels==1)]
cat3 = data[np.where(labels==2)]

plt.scatter(cat1[:,0], cat1[:,1], color='green')
plt.scatter(cat2[:,0], cat2[:,1], color='red')
plt.scatter(cat3[:,0], cat3[:,1], color='blue')
plt.title('Hierarchical clustering with k=3')
plt.xlim(4, 8)
plt.ylim(1, 5)
plt.show()