#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/8/1 21:18
@Author  : 鲍凯辰
@File    : Q3.py
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('accessory1.xls')
data = data.values.tolist()
location = [i[1:3] for i in data]
location = np.array(location)
km = KMeans(n_clusters=150).fit(location)
center = km.cluster_centers_
# for i in range(835):
#     plt.scatter(location[i, 1], location[i, 0], c='blue', marker='o')
# for i in range(150):
#     plt.scatter(center[i, 1], center[i, 0], c='black', marker='*')
# plt.show()
result = center.tolist()
for i in range(150):
    s = 0
    for j in km.labels_:
        if j == i:
            s += 1
    result[i].append(s)
result = np.array(result)
DF = pd.DataFrame(result)
DF.to_csv('Q3_data.csv')
