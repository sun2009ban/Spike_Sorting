#coding=utf-8
from warnings import warn
import operator

from scipy import interpolate
import numpy as np

'''
这个是从网络上下载的参考程序
'''

def split_cells(spikes, idx, which='all'):
    """Return the spike features splitted into separate cells
    """

    if which == 'all':
        classes = np.unique(idx)
    else:
        classes = which

    data = spikes['data']
    time = spikes['time']
    spikes_dict = dict([(cl, {'data': data[:, idx == cl, :], 'time': time})
                        for cl in classes])
    return spikes_dict


def remove_spikes(spt_dict, remove_dict, tolerance):
    """Remove spikes with given spike times from the spike time
    structure """
    spt_data = spt_dict['data']
    spt_remove = remove_dict['data']

    mn, mx = tolerance

    for t in spt_remove:
        spt_data = spt_data[(spt_data > (t + mx)) | (spt_data < (t + mn))]

    spt_ret = spt_dict.copy()
    spt_ret['data'] = spt_data
    return spt_ret


def detect_spikes(spike_data, thresh='auto', edge="rising",
                  contact=0):
    r"""Detects spikes in extracellular data using amplitude thresholding.

    Parameters
    ----------
    spike_data : dict
        extracellular waveforms
    thresh : float or 'auto'
        threshold for detection. if thresh is 'auto' it will be
        estimated from the data.
    edge : {'rising', 'falling'}
        which edge to trigger on
    contact : int, optional
        index of tetrode contact to use for detection, defaults to
        first contact 这个其实是通道编号

    Returns
    -------
    spt_dict : dict
        dictionary with 'data' key which contains detected threshold
        crossing in miliseconds 记录多少毫秒超过阈值了，以miliseconds为单位，
        难怪后面有个1000.0

    """

    sp_data = spike_data['data'][contact, :]

    FS = spike_data['FS'] # 采样频率

    edges = ('rising', 'max', 'falling', 'min')
    if isinstance(thresh, basestring):
        if thresh == 'auto':
            thresh_frac = 8.0
        else:
            thresh_frac = float(thresh)

        thresh = thresh_frac * np.sqrt(sp_data[:10 * FS].var(dtype=np.float64))
        if edge in edges[2:]:
            thresh = -thresh

    if edge not in edges:
        raise TypeError("'edge' parameter must be 'rising' or 'falling'")

    op1, op2 = operator.lt, operator.gt

    if edge in edges[2:]:
        op1, op2 = op2, op1

    '''
    np.where是个高端操作，其可以找到满足后面限制条件值为True的位置，而且针对高维矩阵
    有特殊的操作，比较犀利，返回tuple，采用 i, = 的形式可以把tuple的值取出来。
    '''
    i, = np.where(op1(sp_data[:-1], thresh) & op2(sp_data[1:], thresh))

    spt = i * 1000.0 / FS #以毫秒为单位

    return {'data': spt, 'thresh': thresh, 'contact': contact}


def filter_spt(spike_data, spt_dict, sp_win):
    r'''
    这个函数很简单，idx返回的是输入的spt_dict中的越过threshold毫秒中在区域内的，相当于
    利用sp_win掐头去尾而已。
    '''
    spt = spt_dict['data'] # 记录数据越过threhold的时间点
    sp_data = spike_data['data']
    FS = spike_data['FS']

    try:
        n_pts = sp_data.shape[1]
    except IndexError:
        n_pts = len(sp_data)
    max_time = n_pts * 1000.0 / FS # max_time 数据的总时间

    t_min = np.max((-sp_win[0], 0))
    t_max = np.min((max_time, max_time - sp_win[1]))
    idx, = np.nonzero((spt >= t_min) & (spt <= t_max))
    return idx


def extract_spikes(spike_data, spt_dict, sp_win,
                   resample=1, contacts='all'):
    """Extract spikes from recording.
    这个应该是最重要的一个函数了
    Parameters
    ----------
    spike_data : dict
       extracellular data (see :ref:`raw_recording`)
    spt : dict
       spike times structure (see :ref:`spike_times`)
    sp_win : list of int
       temporal extent of the wave shape

    Returns
    -------
    wavedict : dict
       spike waveforms structure (see :ref:`spike_wave`)

    具体解析一下wavedict['data']部分，其为一个三维矩阵，分别对应于
    spike形状 x 第几个spike（其对应的spike时间是由spt决定的哦） x 第几个通道

    """
    sp_data = spike_data['data'] # 原始数据
    n_contacts = spike_data['n_contacts'] # 通道数目

    if contacts == "all":
        contacts = np.arange(n_contacts)
    elif isinstance(contacts, int):
        contacts = np.array([contacts])
    else:
        contacts = np.asarray(contacts)

    FS = spike_data['FS']
    spt = spt_dict['data'] # spike time，记录的是spike的时间
    idx = np.arange(len(spt))
    inner_idx = filter_spt(spike_data, spt_dict, sp_win)
    outer_idx = idx[~np.in1d(idx, inner_idx)]

    indices = (spt / 1000.0 * FS).astype(np.int32) #indices是把spt转化为下标
    win = (np.asarray(sp_win) / 1000.0 * FS).astype(np.int32) #win是把sp_win从以ms为单位转化为下标宽度
    time = np.arange(win[1] - win[0]) * 1000.0 / FS + sp_win[0] # time是相当于时间真实时间，
    n_contacts, n_pts = sp_data.shape

    # auxiliary function to find a valid spike window within data range
    minmax = lambda x: np.max([np.min([n_pts, x]), 0]) # minmax是一个函数，其有一个输入变量x
    spWave = np.zeros((len(time), len(spt), len(contacts)),
                      dtype=np.float32)

    for i in inner_idx:
        sp = indices[i]
        spWave[:, i, :] = np.atleast_2d(sp_data[contacts,
                                                sp + win[0]:sp + win[1]]).T # 这个操作还是比较高端，np.atleast_2d()使得输入至少以二维矩阵表示
		# 对于 inner_idx 的，就直接在spt两边取了，这个很直接，而且宽度能保证

    for i in outer_idx:
        sp = indices[i]
        l, r = map(minmax, sp + win)
        if l != r:
            spWave[(l - sp) - win[0]:(r - sp) - win[0], i, :] = \
                sp_data[contacts, l:r].T

    wavedict = {"data": spWave, "time": time, "FS": FS}

    if len(idx) != len(inner_idx):
        is_valid = np.zeros(len(spt), dtype=np.bool)
        is_valid[inner_idx] = True
        wavedict['is_valid'] = is_valid

    if resample != 1:
        warn("resample argument is deprecated."
             "Please update your code to use function"
             "resample_spikes", DeprecationWarning)
        wavedict = resample_spikes(wavedict, FS * resample)
    return wavedict


def resample_spikes(spikes_dict, FS_new):
    """Upsample spike waveforms using spline interpolation"""

    sp_waves = spikes_dict['data']
    time = spikes_dict['time']
    FS = spikes_dict['FS']

    resamp_time = np.arange(time[0], time[-1], 1000.0 / FS_new)
    n_pts, n_spikes, n_contacts = sp_waves.shape

    spike_resamp = np.empty((len(resamp_time), n_spikes, n_contacts))

    for i in range(n_spikes):
        for contact in range(n_contacts):
            tck = interpolate.splrep(time, sp_waves[:, i, contact], s=0)
            spike_resamp[:, i, contact] = interpolate.splev(resamp_time,
                                                            tck, der=0)

    return {"data": spike_resamp, "time": resamp_time, "FS": FS}


def align_spikes(spike_data, spt_dict, sp_win, type="max", resample=1,
                 contact=0, remove=True):
    """Aligns spike waves and returns corrected spike times

    Parameters
    ----------
    spike_data : dict
    spt_dict : dict 这个可以通过 detect_spikes() 得到,里面主要是cross threshold的时间信息
    sp_win : list of int 这个貌似不太对，list of float, 表示在下面type的左、右各拓宽多少。
    type : {'max', 'min'}, optional
    resample : int, optional
    contact : int, optional 处理哪个通道，注意一次只能处理一个通道的数据
    remove : bool, optiona

    Returns
    -------
    ret_dict : dict
       spike times of aligned spikes

    """

    tol = 0.1

    if (sp_win[0] > -tol) or (sp_win[1] < tol):
        warn('You are using very short sp_win. '
             'This may lead to alignment problems.')

    spt = spt_dict['data'].copy()

    idx_align = np.arange(len(spt)) # np.arange生成的就是np.array [0,1,2,.. len(spt)-1]

    #go in a loop until all spikes are correctly aligned
    iter_id = 0
    while len(idx_align) > 0:
        spt_align = {'data': spt[idx_align]}
        spt_inbound = filter_spt(spike_data, spt_align, sp_win)
        idx_align = idx_align[spt_inbound]
        sp_waves_dict = extract_spikes(spike_data, spt_align, sp_win,
                                       resample=resample, contacts=contact)

        sp_waves = sp_waves_dict['data'][:, spt_inbound, 0] # 一次只能处理一个通道的数据
        time = sp_waves_dict['time']

        if type == "max":
            i = sp_waves.argmax(0)
        elif type == "min":
            i = sp_waves.argmin(0)

        #move spike markers 把目标marker移动到中心，最初时spt是 原始数据cross threshold的时间
        shift = time[i]
        spt[idx_align] += shift

        #if spike maximum/minimum was at the edge we have to extract it at the
        # new marker and repeat the alignment

        idx_align = idx_align[(shift < (sp_win[0] + tol)) |
                              (shift > (sp_win[1] - tol))]
        iter_id += 1

    ret_dict = {'data': spt}

    if remove:
        #remove double spikes
        FS = spike_data['FS']
        ret_dict = remove_doubles(ret_dict, 1000.0 / FS)

    return ret_dict


def remove_doubles(spt_dict, tol, distant=1):
    '''
    这个函数的作用是保证相邻两个spike间的距离的，如果两者距离过近就会被合并为一个
    而距离由下面的 intisi > distant决定, distant为采样点数int类型。
    '''
    new_dict = spt_dict.copy()
    spt = spt_dict['data']

    isi = np.diff(spt)
    intisi = (isi/tol).astype('int')

    if len(spt) > 0:
        spt = spt[np.concatenate(([True], intisi > distant))]

    new_dict['data'] = spt
    return new_dict


def merge_spikes(spike_waves1, spike_waves2):
    """Merges two sets of spike waves

    Parameters
    ----------

    spike_waves1 : dict
    spike_waves2 : dict
        spike wavefroms to merge; both spike wave sets must be defined
        within the same time window and with the same sampling
        frequency

    Returns
    -------

    spike_waves : dict
        merged spike waveshapes

    clust_idx : array
        labels denoting to which set the given spike originally belonged to
    """

    sp_data1 = spike_waves1['data']
    sp_data2 = spike_waves2['data']

    sp_data = np.hstack((sp_data1, sp_data2))
    spike_waves = spike_waves1.copy()
    spike_waves['data'] = sp_data

    clust_idx = np.concatenate((np.ones(sp_data1.shape[1]),
                               np.zeros(sp_data2.shape[1])))

    return spike_waves, clust_idx


def merge_spiketimes(spt1, spt2, sort=True):
    """Merges two sets of spike times

    Parameters
    ----------
    spt1 : dict
    spt2 : dict
    sort : bool, optional
        False if you don't want to be the spike times sorted.

    Returns
    -------
    spt : dict
        dictionary with merged spike time arrrays under data key
    clust_idx : array
        labels denoting to which set the given spike originally belonged
        to

    """

    spt_data1 = spt1['data']
    spt_data2 = spt2['data']

    spt_data = np.concatenate((spt_data1, spt_data2))
    i = spt_data.argsort()
    spt_data = spt_data[i]

    clust_idx = np.concatenate((np.ones(spt_data1.shape[0]),
                               np.zeros(spt_data2.shape[0])))
    clust_idx = clust_idx[i]
    spt_dict = {"data": spt_data}

    return spt_dict, clust_idx
