# -*- coding: UTF-8 -*-
from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def power_spectrum(data, time_step, plot_out=False, color='b', linestyle='-', alpha=1, label='None'):
    r'''
    data: 输入数据
    time_step: 1/sampling_frequency
    plot_out: True/False 决定是否输出图像
    '''
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, time_step)
    if plot_out:
        idx = np.argsort(freqs)
        plt.plot(freqs[idx], ps[idx],color=color, linestyle=linestyle, alpha=alpha, label=label)
    return ps
