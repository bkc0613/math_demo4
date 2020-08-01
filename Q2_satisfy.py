#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/7/31 23:46
@Author  : 鲍凯辰
@File    : Q2_satisfy.py
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sqrt
import pandas as pd
from geopy.distance import geodesic
import re
import heapq
np.set_printoptions(threshold=10000)
past_data = pd.read_excel('accessory1.xls')
past_data = past_data.values.tolist()
customer_data = pd.read_excel('accessory2.xlsx')
customer_data = customer_data.values.tolist()


def satisfy(distance, price):
    s = 0.5 * distance + 0.5 * price
    return s


def calculate_distance(X, Y):
    distance = geodesic(X, Y)
    distance = str(distance)
    part = re.compile(r'\d+')
    distance = part.findall(distance)
    distance = float(distance[0] + '.' + distance[1])
    return distance


def distance_w():
    dis = [list() for i in range(0, 835)]
    location = [i[1:3] for i in past_data]
    f = open('distance1.txt', 'w')
    for i in range(0, 835):
        print(i)
        x = tuple(location[i])
        for j in range(0, 1864):
            y = customer_data[j][1].split()
            float_y = tuple([float(k) for k in y])
            dist = calculate_distance(x, float_y)
            dis[i].append(dist)
            f.write(str(dist))
            f.write(' ')
        f.write('\n')
    f.close()


def get_n_dis():
    dis_data = pd.read_table('distance1.txt', header=None, sep=' ')
    dis_data = dis_data.iloc[:, 0:1864]
    dis_data = dis_data.values.tolist()
    dis_data = np.array(dis_data)
    s = MinMaxScaler(feature_range=(0, 1))
    s_dis = s.fit_transform(dis_data)
    for i in range(835):
        for x in range(1864):
            s_dis[i, x] = 1 - s_dis[i, x]
    return s_dis


def get_satisfy(s_dis, money):
    money = np.array([money]).T
    sat = [list() for i in range(835)]
    for i in range(835):
        m = money[i, 0]
        for x in range(1864):
            d = s_dis[i, x]
            s = satisfy(d, m)
            sat[i].append(s)
    return sat


def get_limit():
    li = list()
    for x in customer_data:
        g = x[2] * 835 / 12468
        if g > 1:
            li.append(int(g)*2)
        elif g > 0.5:
            li.append(1)
        else:
            li.append(0)
    return li


def get_threshold():
    satisfy_data = pd.read_table('satisfy.txt', header=None, sep=' ')
    satisfy_data = satisfy_data.iloc[:, 0:1864]
    satisfy_data = satisfy_data.values.tolist()
    for i in range(835):
        if past_data[i][4] == 1:
            extreme = max(satisfy_data[i])
            print('<' + str(extreme))
        else:
            extreme = max(satisfy_data[i])
            print('>' + str(extreme))


money = []
for x in range(835):
    money.append(0)
limit = get_limit()
sumx = 0
for m in limit:
    sumx = m + sumx
print(sumx)
print(limit)
n_dis = get_n_dis()
c = list()
for n in range(34):
    print(n)
    chosen = []
    print(money)
    sat_matrix = get_satisfy(n_dis, money)
    sat_matrix = np.array(sat_matrix)
    print(sat_matrix)
    for i in range(1864):
        arr = sat_matrix[:, i]
        arr = arr.tolist()
        re1 = map(arr.index, heapq.nlargest(limit[i], arr))
        re1 = list(re1)
        for a in re1:
            if arr[a] > 0.51:
                chosen.append(a)
                for j in range(1864):
                    sat_matrix[a, j] = -1
    for k in range(835):
        if k not in chosen:
            money[k] = money[k] + 0.025
    print(chosen)
    print(len(chosen))
    c = chosen
print(c)
result = [[0 for i in range(835)], list()]
for i in c:
    result[0][i] = 1
for i in range(0, 835):
    mr = money[i] * 20 + 65
    result[1].append(mr)
result = np.array(result).T
DF = pd.DataFrame(result, columns=['任务完成情况', '花费'])
print(DF)
DF.to_csv('Q2_result.csv')
