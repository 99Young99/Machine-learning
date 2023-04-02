#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : 青年有志
# @time    : 2023-03-30 16:46
# @function: the script is used to do something.
# @version : V1

'''
数据集：iris
数据集数量：150
-----------------------------
运行结果：
ARI：0.56
运行时长：177.0s
'''

import numpy as np
import math
import time
from scipy.special import comb


# 定义加载数据的函数
def load_data(file):
    '''
    INPUT:
    file - (str) 数据文件的路径
    
    OUTPUT:
    Xarray - (array) 特征数据数组
    Ylist - (list) 类别标签列表
    
    '''
    Xlist = []  # 定义一个列表用来保存每条数据
    Ylist = []  # 定义一个列表用来保存每条数据的类别标签
    fr = open(file)
    for line in fr.readlines():  # 逐行读取数据，鸢尾花数据集每一行表示一个鸢尾花的特征和类别标签，用逗号分隔
        cur = line.split(',')
        label = cur[-1]
        X = [float(x) for x in cur[:-1]]  # 用列表来表示一条特征数据
        Xlist.append(X)
        Ylist.append(label)
    Xarray = np.array(Xlist)  # 将特征数据转换为数组类型，方便之后的操作
    print('Data shape:', Xarray.shape)
    print('Length of labels:', len(Ylist))
    return Xarray, Ylist


# 定义标准化函数，对每一列特征进行min-max标准化，将数据缩放到0-1之间
# 标准化处理对于计算距离的机器学习方法是非常重要的，因为特征的尺度不同会导致计算出来的距离倾向于尺度大的特征，为保证距离对每一列特征都是公平的，必须将所有特征缩放到同一尺度范围内
def Normalize(Xarray):
    '''
    INPUT:
    Xarray - (array) 特征数据数组
    
    OUTPUT:
    Xarray - (array) 标准化处理后的特征数据数组
    
    '''
    for f in range(Xarray.shape[1]):
        maxf = np.max(Xarray[:, f])
        minf = np.min(Xarray[:, f])
        for n in range(Xarray.shape[0]):
            Xarray[n][f] = (Xarray[n][f]-minf) / (maxf-minf) 
    return Xarray


# 定义计算两条数据间的距离的函数，这里计算的是欧式距离
def cal_distance(xi, xj):
    '''
    INPUT:
    Xi - (array) 第i条特征数据
    Xj - (array) 第j条特征数据
    
    OUTPUT:
    dist - (float) 两条数据的欧式距离
    
    '''
    dist = 0
    for col in range(len(xi)):
        dist += (xi[col]-xj[col]) ** 2
    dist = math.sqrt(dist)
    return dist


# 定义计算所有特征数据两两之间距离的函数
def Distances(Xarray):
    '''
    INPUT:
    Xarray - (array) 特征数据数组
    
    OUTPUT:
    dists - (array) 两两数据的欧式距离数组
    
    '''
    # 定义一个数组用来保存两两数据的距离
    dists = np.zeros((Xarray.shape[0], Xarray.shape[0]))
    for n1 in range(Xarray.shape[0]):
        for n2 in range(n1):
            dists[n1][n2] = cal_distance(Xarray[n1], Xarray[n2])
            dists[n2][n1] = dists[n1][n2]
    return dists


# 定义计算两类的类间距离的函数，这里计算的是最短距离
def cal_groupdist(g1, g2, group_dict, dists):
    '''
    INPUT:
    g1 - (int) 类别1的标签
    g2 - (int) 类别2的标签
    group_dict - (dict) 类别字典
    dists - (array) 两两数据的欧式距离数组
    
    OUTPUT:
    (int) 类间最短距离
    
    '''
    d = []
    # 循环计算两类之间两两数据的距离
    for xi in group_dict[g1]:
        for xj in group_dict[g2]:
            if xi != xj:
                d.append(dists[xi][xj])
    return min(d)


# 定义层次聚类函数
def Clustering(Xarray, k, dists):
    '''
    INPUT:
    Xarray - (array) 特征数据数组
    k - (int) 设定的类别数
    dists - (array) 两两数据的欧式距离数组
    
    OUTPUT:
    group_dict - (dict) 类别字典
    
    '''
    # 定义一个空字典，用于保存聚类所产生的所有类别
    group_dict = {}

    # 首先将每条数据都分到不同的类，数据的类别标签为 0-(N-1)，其中 N 为数据条数,(开始时每个样本数据就是一个类)
    for n in range(Xarray.shape[0]):
        group_dict[n] = [n]

    # newgroup 表示新的类别标签，此时下一个类别标签为 N
    newgroup = Xarray.shape[0]

    # 当类别数大于我们所设定的类别数 k 时，不断循环进行聚类
    while len(group_dict.keys()) > k:

        print('Number of groups:', len(group_dict.keys()))

        # 定义一个空字典，用于保存两两类之间的间距，其中字典的值为元组(g1, g2)，表示两个类别标签，字典的键为这两个类别的间距
        # group_dists 中 的 key 为两两类的最小距离，values 为类标签组成的元组
        group_dists = {}

        # 循环计算 group_dict 中两两类别之间的间距，保存到 group_dists 中
        for g1 in group_dict.keys():
            for g2 in group_dict.keys():
                if g1 != g2:
                    if (g1, g2) not in group_dists.values():
                        d = cal_groupdist(g1, g2, group_dict, dists)
                        group_dists[d] = (g1, g2)

        # 取类别之间的最小间距
        group_mindist = min(list(group_dists.keys()))

        # 取类别之间的最小间距
        mingroups = group_dists[group_mindist]

        # 定义一个列表，用于保存所产生的新类中包含的数据，这里用之前对每条数据给的类别标签 0-(N-1) 来表示
        new = []
        for g in mingroups:

            # 将间距最小的两类中包含的数据保存在 new 列表中
            new.extend(group_dict[g])

            # 然后在 group_dict 中移去这两类
            del group_dict[g]

        print("newgroup: ", newgroup, "new: ", new)

        # 此时聚类所产生的新类中包含的数据即为以上两类的中包含的数据的聚合，给新类贴上类别标签为 newgroup，保存到 group_dict 中
        group_dict[newgroup] = new

        # 产生下一个类别标签
        newgroup += 1

    return group_dict


# 定义计算调整兰德系数(ARI)的函数，调整兰德系数是一种聚类方法的常用评估方法
def Adjusted_Rand_Index(group_dict, Ylist, k):
    '''
    INPUT:
    group_dict - (dict) 类别字典
    Ylist - (list) 类别标签列表
    k - (int) 设定的类别数
    
    OUTPUT:
    (int) 调整兰德系数
    
    '''
    # 定义一个数组，用来保存聚类所产生的类别标签与给定的外部标签各类别之间共同包含的数据数量
    group_array = np.zeros((k, k))

    # 定义一个空字典，用来保存外部标签中各类所包含的数据，结构与group_dict相同
    y_dict = {}
    for i in range(len(Ylist)):
        if Ylist[i] not in y_dict:
            y_dict[Ylist[i]] = [i]
        else:
            y_dict[Ylist[i]].append(i)
    # 循环计算 group_array 的值
    for i in range(k):
        for j in range(k):
            for n in range(len(Ylist)):
                if n in group_dict[list(group_dict.keys())[i]] and n in y_dict[list(y_dict.keys())[j]]:

                    # 如果数据 n 同时在 group_dict 的类别 i 和 y_dict 的类别j中，group_array[i][j] 的数值加一
                    group_array[i][j] += 1

    # 定义兰德系数(RI)
    RI = 0

    # 定义一个数组，用于保存聚类结果 group_dict 中每一类的个数
    sum_i = np.zeros(k)

    # 定义一个数组，用于保存外部标签 y_dict 中每一类的个数
    sum_j = np.zeros(k)
    for i in range(k):
        for j in range(k):
            sum_i[i] += group_array[i][j]
            sum_j[j] += group_array[i][j]
            if group_array[i][j] >= 2:

                # comb 用于计算 group_array[i][j] 中两两组合的组合数
                RI += comb(group_array[i][j], 2)
    # ci 保存聚类结果中同一类中的两两组合数之和
    ci = 0

    # cj 保存外部标签中同一类中的两两组合数之和
    cj = 0
    for i in range(k):
        if sum_i[i] >= 2:
            ci += comb(sum_i[i], 2)
    for j in range(k):
        if sum_j[j] >= 2:
            cj += comb(sum_j[j], 2)

    # 计算 RI 的期望
    E_RI = ci * cj / comb(len(Ylist), 2)

    # 计算 RI 的最大值
    max_RI = (ci + cj) / 2

    # 返回调整兰德系数的值
    return (RI-E_RI) / (max_RI-E_RI)

if __name__ == "__main__":
    # 加载数据
    Xarray, Ylist = load_data('..\iris.data')

    # 保存开始时间
    start = time.time()

    # 对特征数据进行标准化处理
    Xarray = Normalize(Xarray)

    # 设定聚类数为3
    k = 3

    # 计算特征数据的距离数组
    dists = Distances(Xarray)
    print(dists)

    # 进行层次聚类, key 为不同的类， value 为类标签
    group_dict = Clustering(Xarray, k, dists)

    # 保存结束时间
    end = time.time()
    print(group_dict)

    # 计算 ARI 用来评估聚类结果
    ARI = Adjusted_Rand_Index(group_dict, Ylist, k)
    print('Adjusted Rand Index:', ARI)
    print('Time:', end-start)
