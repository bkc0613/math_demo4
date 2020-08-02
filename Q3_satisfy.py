#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""code_info
@Time    : 2020 2020/8/1 22:25
@Author  : 鲍凯辰
@File    : Q3_satisfy.py
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import re
import heapq
np.set_printoptions(threshold=10000)
past_data = pd.read_csv('Q3_data.csv')
past_data = past_data.values.tolist()
customer_data = pd.read_excel('accessory2.xlsx')
customer_data = customer_data.values.tolist()
# 获得配额排名前170的会员
lim = list()
for a in customer_data:
    lim.append(a[2])
re1 = heapq.nlargest(170, enumerate(lim), key=lambda x: x[1])
index, vals = zip(*re1)
index = list(index)
index.sort(reverse=False)
max_lim = list()
for i in index:
    max_lim.append(customer_data[i])


def satisfy(distance, price):  # 计算吸引度
    s = 0.5 * distance + 0.5 * price
    return s


def calculate_distance(X, Y):  # 计算任务包与会员间距离
    distance = geodesic(X, Y)
    distance = str(distance)
    part = re.compile(r'\d+')
    distance = part.findall(distance)
    distance = float(distance[0] + '.' + distance[1])
    return distance


def distance_w():  # 保存距离信息
    dis = [list() for i in range(0, 150)]
    location = [i[0:2] for i in past_data]
    f = open('Q3_distance1.txt', 'w')
    for i in range(0, 150):
        x = tuple(location[i])
        for j in range(0, 170):
            y = max_lim[j][1].split()
            float_y = tuple([float(k) for k in y])
            dist = calculate_distance(x, float_y)
            dis[i].append(dist)
            f.write(str(dist))
            f.write(' ')
        f.write('\n')
    f.close()


def get_n_dis():  # 标准化距离矩阵
    dis_data = pd.read_table('Q3_distance1.txt', header=None, sep=' ')
    dis_data = dis_data.iloc[:, 0:170]
    dis_data = dis_data.values.tolist()
    dis_data = np.array(dis_data)
    s = MinMaxScaler(feature_range=(0, 1))
    s_dis = s.fit_transform(dis_data)
    for i in range(150):
        for x in range(170):
            s_dis[i, x] = 1 - s_dis[i, x]
    return s_dis


def get_satisfy(s_dis, money):  # 获得每个任务包对每个会员的吸引度矩阵
    money = np.array([money]).T
    sat = [list() for i in range(150)]
    for i in range(150):
        m = money[i, 0]
        for x in range(170):
            d = s_dis[i, x]
            s = satisfy(d, m)
            sat[i].append(s)
    return sat


money = []
# 初始化任务包平均定价为0
for x in range(150):
    money.append(0)
n_dis = get_n_dis()
c = list()
# 循环得到已选任务包数收敛时的定价
for n in range(33):
    chosen = []
    sat_matrix = get_satisfy(n_dis, money)
    sat_matrix = np.array(sat_matrix)
    for i in range(170):
        arr = sat_matrix[:, i]
        arr = arr.tolist()
        re1 = map(arr.index, heapq.nlargest(1, arr))
        re1 = list(re1)
        for a in re1:
            if arr[a] > 0.51:  # 吸引度阈值
                chosen.append(a)
                for j in range(170):
                    sat_matrix[a, j] = -1
    for k in range(150):  # 更新定价列表
        if k not in chosen:
            money[k] = money[k] + 0.025
    c = chosen
# 保存优化结果
result = [[0 for i in range(150)], list(), list()]
for i in c:
    result[0][i] = 1
for i in range(0, 150):
    mr = money[i] * 20 + 65
    result[1].append(mr)
for i in past_data:
    result[2].append(i[2])
result = np.array(result).T
DF = pd.DataFrame(result, columns=['任务完成情况', '花费', '包含任务数'])
print(DF)
DF.to_csv('Q3_result.csv')
