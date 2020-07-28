#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/28 10:19
@Author  : 鲍凯辰
@File    : Q1_KMeans.py
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import re

data = pd.read_excel('accessory1.xls')
data = data.values.tolist()
location = [i[1:3] for i in data]
location = np.array(location)
km = KMeans(n_clusters=6).fit(location)
distance = []
center = km.cluster_centers_.tolist()
l_location = location.tolist()
cost = [k[3] for k in data]
y = []
for i in range(0, 835):
    classify = km.labels_[i]
    if classify == 4:
        X = tuple(center[classify])
        Y = tuple(location[i])
        result = geodesic(X, Y)
        result = str(result)
        pattern = re.compile(r'\d+')
        r = pattern.findall(result)
        distance.append(int(r[0]))
        y.append(cost[i])
x = np.array([distance]).T.astype(np.float32)
# y = np.array([[k[3] for k in data]]).T
y = np.array([y]).T
plt.scatter(x, y, marker='o', color='red')
plt.show()
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# lm = linear_model.LinearRegression().fit(x_train, y_train)
# print(lm.intercept_)
# print(lm.coef_)
# res = lm.predict(x_test)
# compare = np.insert(res, [1], y_test, axis=1)
# print(compare)
# color = ['red', 'green', 'black', 'yellow']
# for j in range(0, 4):
#     plt.scatter(location[km.labels_ == j, 1], location[km.labels_ == j, 0], c=color[j])
# plt.show()
