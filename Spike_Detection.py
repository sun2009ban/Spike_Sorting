#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pdb
'''
通过template matching提取数据中的spike的
author: swt
date: 2017/09/15
email: sun_wentao@outlook.com
'''

def spike_detect_threshold(data, threshold, overlap=10):
    '''
    data 为输入数据
    threshold 为阈值，绝对值通过阈值的均可以认为有可能为spike
    overlap 决定了两个spike之间最小的宽度，必须在overlap间全是静息时间才可以
    -------------------------------------------------------
    返回值 spikeDict是一个字典，其key是 1,2,3,... 表示的是spike的ID
    而每个key对应一个list，里面是两个值，分别是spike开始和结束的下标。
    '''
    spikeDict = {} # 字典，每个key有两项，开始，结束

    count = 0
    dataLen = data.size

    spikeProperties = []

    i = 0

    while i < dataLen:
        if np.abs(data[i]) > threshold and np.abs(data[i-1]) < threshold:
            #判断开头越过阈值前的静息时间是否够长
            j = 0
            silentTime = []
            spikeProperties = []

            while j < overlap and i - 1 - j >= 0:
                silentTime.append( np.abs(data[i - 1 - j]) < threshold )
                j = j + 1

            if np.all(silentTime):
                count = count + 1
                spikeProperties.append(i - j // 2) # 把spike的开始写入

                while i + 1 < dataLen:
                    if np.abs(data[i]) > threshold and np.abs(data[i+1]) < threshold:
                        #判断结尾的静息时间是否够长
                        l = 0
                        silentTime = []
                        while (l < overlap) and i + 1 + l <= dataLen - 1:
                            silentTime.append( np.abs(data[i + 1 + l]) < threshold )
                            l = l + 1

                        if np.all(silentTime):
                            spikeProperties.append(i + l // 2)
                            spikeDict[count] = spikeProperties
                            break
                        else:
                            i = i + 1

                    i = i + 1

        i = i + 1

    return spikeDict


# 寻找spike的函数

def spike_detect(x, free_threshold, amplitude_threshold, max_value_position=20, fixed_length=64):
    '''
    x 是输入的数据 np.array， 一维向量
    free_threshold 是在每个spike前面需要空的间隔
    amplitude_threshold 是幅值超过多少被认定为threhold
    fixed_length 是截取信号的宽度，如果为None，则认为是变宽度信号，由算法自身判断信号长度
    ## 注意：使用中一定要把x的最大值放在20的位置

    '''
    max_value_index = max_value_position
    x_seq = x.flatten('C')
    times = x_seq.shape[0]
    spike = {}
    spike_index = 0

    i = 0
    while i < (times - free_threshold):
        free_threshold_flag =[]

        for j in range(free_threshold):
            free_threshold_flag.append( np.abs(x_seq[i+j]) < amplitude_threshold )

        free_threshold_flag.append(np.abs(x_seq[i+free_threshold]) > amplitude_threshold)

        if np.all(free_threshold_flag):
            #满足 free_threshold 这么长之内均低于 amplitude_threshold, 而超过 free_threshold 则蹦到 amplitude_threshold之上

            k = 1
            while (i+free_threshold+k < times) and (x_seq[i+free_threshold+k] > amplitude_threshold):
                k = k + 1
            # spike_max_id 超过 aplitude_threshold 的这一段中的最大值的位置
            spike_max_id = i + free_threshold + np.argmax( np.abs(x_seq[i+free_threshold: i+free_threshold+k+1]) )
            # record the index of the spike maxima

            # 记录spike的起始
            if spike_max_id >=max_value_position:
                spike[str(spike_index)] = [spike_max_id - max_value_index]

                # 为固定长度
                # 记录spike的结尾
                spike_end_id = spike_max_id + fixed_length - max_value_index
                # 记录spike的终止
                if spike_end_id < times:
                    spike[str(spike_index)].append( spike_end_id )

                spike_index = spike_index + 1
                i = spike_end_id


        i =  i + 1

    # 这个下面的方式是一个技巧，dict做循环同时循环中有删除操作时，需要使用copy()来复制一下
    # python 2.7 就是有这个毛病，不会自动复制。
    for keys in spike.copy():
        if len(spike[keys]) != 2:
            del spike[keys]

    return spike

def noise_level_estimate(x, fs):
    r'''
    估计x的噪音水平
    x 为数据
    fs 为采样率
    '''
    dataLen = len(x)
    winLen = int(100 / 1000 * fs) # 用100ms作为每一小段的长度
    winNum = int(dataLen // winLen)

    if winLen < 20:
        warnings.warn('Please increase sampling frequency!')
    assert winNum > 3 #保证至少分出三个来

    x_new = np.reshape(x[:winNum * winLen], (winNum, winLen))

    var_x = np.var(x_new, axis = 1)

    sorted_var_x = np.sort(var_x, axis=None)

    diff_var_x = np.diff(sorted_var_x)

    noise_num = int(winNum * 0.3) + 1

    i = 0
    noise_level = 0
    while i < noise_num and diff_var_x[i] < 1.5 * sorted_var_x[0]:
        noise_level = noise_level + sorted_var_x[i]
        i = i + 1

    return np.sqrt(noise_level / (i+1))


def get_template(spike, max_value_index, before_max_len, after_max_len):
    '''
    spike: 一个提取了spike的信号，
    max_value_index: spike中最大值的位置，注意这里输入的数值是把spike自己看作一个向量，是相对于spike这个向量的位置
    before_max_len: 最大值前面取出多长
    after_max_len: 最大值后面取出多长 
    '''
    spike_template = np.zeros(before_max_len + 1 + after_max_len)

    spike_len = len(spike)

    before_max_real_len = min(max_value_index, before_max_len)
    after_max_real_len = min(spike_len - max_value_index - 1, after_max_len)

    spike_template[before_max_len - before_max_real_len : before_max_len + after_max_real_len] = spike[max_value_index - before_max_real_len : max_value_index + after_max_real_len]
    return spike_template
