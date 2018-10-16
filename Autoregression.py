# -*- coding: UTF-8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
# auto-regression 求解函数
def autoregression(timeSeq, M):
    N = timeSeq.shape[0]
    X = []
    y = []
    for i in range(N-M):
        rowVector = timeSeq[i:i+M]
        X.append(rowVector[::-1])
        y.append(timeSeq[i+M])
    X = np.array(X)
    y = np.array(y)
    # 把y变成列向量
    y = np.expand_dims(y,axis=1)
    part1 = np.linalg.inv(np.matmul(np.transpose(X), X))
    part2 = np.matmul(np.transpose(X), y)
    a_opt = np.matmul(part1, part2)
    return a_opt

# 输入的rawData必须是一个2维矩阵
# 返回值是一个于rawData行数相同，列数为order的矩阵
def generate_autoregressive_features(rawData, order):
    AR_features = []
    for i in rawData:
        AR_features.append(np.squeeze(autoregression(i, order)))
    return np.array(AR_features)
