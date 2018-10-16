# -*- coding: UTF-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter, freqz, iirnotch

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch(cutoff, fs, Q=None):
    w0 = cutoff / (fs / 2)
    if Q is None:
        Q = 3/w0 # 质量系数
    assert w0 > 0 and w0 <= 1
    b, a = iirnotch(w0, Q)
    w, h = freqz(b, a)
    # Generate frequency axis
    freq = w*fs/(2*np.pi)
    # Plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(freq, 20*np.log10(abs(h)), color='blue')
    ax.set_title("Frequency Response")
    ax.set_ylabel("Amplitude (dB)", color='blue')
    ax.set_ylim([-25, 10])
    ax.grid()
    return

def notch_filter(data, cutoff, fs):
    '''
    陷波滤波器，cutoff为被限制频率，值为0 < cutoff < (fs/2)
    '''
    w0 = cutoff / (fs / 2)
    Q = min(3/w0, 3) # 质量系数
    assert w0 > 0 and w0 <= 1
    b, a = iirnotch(w0, Q)
    y = lfilter(b, a, data)
    return y
