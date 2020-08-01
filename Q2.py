#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/31 16:01
@Author  : 鲍凯辰
@File    : Q2.py
"""
from scipy.optimize import fsolve
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import re


data = pd.read_excel('accessory1.xls')
data = data.values.tolist()
location = [i[1:3] for i in data]
location = np.array(location)
km = KMeans(n_clusters=4).fit(location)
center = km.cluster_centers_.tolist()
center1 = tuple(center[0])
center2 = tuple(center[1])
center_point1 = tuple()
center_point2 = tuple()
min_d = 100
min_k1 = 0
for k in range(0, 835):
    X = tuple(location[k])
    d = geodesic(center1, X)
    d = str(d)
    part = re.compile(r'\d+')
    d = part.findall(d)
    d = float(d[0] + '.' + d[1])
    if d < min_d:
        min_k1 = k
        center_point1 = X
        min_d = d
print(min_k1, min_d, center_point1, center1)
min_d = 100
min_k2 = 0
for k in range(0, 835):
    X = tuple(location[k])
    d = geodesic(center2, X)
    d = str(d)
    part = re.compile(r'\d+')
    d = part.findall(d)
    d = float(d[0] + '.' + d[1])
    if d < min_d:
        min_k2 = k
        center_point2 = X
        min_d = d
print(min_k2, min_d, center_point2, center2)
# color = ['red', 'green', 'black', 'yellow']
# for j in range(0, 4):
#     plt.scatter(location[km.labels_ == j, 1], location[km.labels_ == j, 0], c=color[j])
# plt.show()
customer_data = pd.read_excel('accessory2.xlsx')
customer_data = customer_data.values.tolist()
result1 = tuple()
result2 = tuple()
min_d1 = 100
min_j1 = 0
for j in range(0, 1877):
    x = customer_data[j][1].split()
    float_x = tuple([float(i) for i in x])
    distance = geodesic(center_point1, float_x)
    distance = str(distance)
    part = re.compile(r'\d+')
    distance = part.findall(distance)
    distance = float(distance[0] + '.' + distance[1])
    if distance < min_d1:
        result1 = float_x
        min_d1 = distance
        min_j1 = j
print(min_j1, min_d1, result1, center_point1)
min_d2 = 100
min_j2 = 0
for j in range(0, 1877):
    x = customer_data[j][1].split()
    float_x = tuple([float(i) for i in x])
    distance = geodesic(center_point2, float_x)
    distance = str(distance)
    part = re.compile(r'\d+')
    distance = part.findall(distance)
    distance = float(distance[0] + '.' + distance[1])
    if distance < min_d2:
        result2 = float_x
        min_d2 = distance
        min_j2 = j
print(min_j2, min_d2, result2, center_point2)
money = list()
for i in data:
    money.append(i[3])
print(money)
money1 = money[min_k1]
money2 = money[min_k2]


def func(i):
    x, y = i[0], i[1]
    return [
        sqrt(x/min_d1**2 + y*money1**2)-0.99,
        sqrt(x/min_d2**2 + y*money2**2)-0.99
    ]


aa = fsolve(func, np.array([0, 0]))
print(aa)
print(aa[1], type(aa[1]))
test = sqrt(aa[0]/min_d1**2 + aa[1] * money1**2)
print(test)

