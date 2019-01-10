# -*- coding: UTF-8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import operator

def get_absolute_max(data):
    r'''
    获得输入数据的绝对值的最大值点和相应的位置
    data为一维向量
    ----------------
    返回值为最大值对应的index和value
    '''
    data_abs = np.abs(data)
    index, value = max(enumerate(data_abs), key=operator.itemgetter(1))
    value = data[index]
    return index, value


def get_value_max(data):
    r'''
    获得输入数据的值的最大值点和相应的位置
    data为一维向量
    ----------------
    返回值为最大值对应的index和value
    '''
    index, value = max(enumerate(data), key=operator.itemgetter(1))
    value = data[index]
    return index, value