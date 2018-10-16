import numpy as np
import Spike_Detection as SD
import Filter
import matplotlib.pyplot as plt
import Utilities as util
import pdb
import os

class spike_sorting(object):
	def __init__(self, raw_data, fs=1000, cutoff=100, threshold=0.1, overlap=10, save_interval_path=None, 
					before_max=5, after_max=5, plot=False):
		# 只接受一维度数据
		self.raw_data = np.squeeze(raw_data)
		assert len(self.raw_data.shape) == 1, "only vector is allowed" 
		#减去平均
		self.raw_data = raw_data - np.mean(raw_data)
		self.maximum_data = max(self.raw_data)
		self.minimum_data = min(self.raw_data)

		# 高通滤波参数
		self.fs = fs
		self.cutoff = cutoff

		# 通过阈值提取spike
		self.threshold = threshold
		self.overlap = overlap

		# 计算spike之间的interval
		self.save_interval_path = save_interval_path

		# 提取spike的waverform
		self.before_max = before_max
		self.after_max = after_max

		self.plot = plot


	def go(self):
		# 数据高通滤波
		filtered_data = self._highpass(self.raw_data, self.fs, self.cutoff, self.plot)
		# 提取spike，把性质存储在dict中
		spikes_dict = self._spike_threshold(filtered_data, self.threshold, self.overlap, self.plot)
		# 计算和存储intervals
		self._get_intervals(spikes_dict, self.save_interval_path, self.plot)
		# 获取每个spike的waveform，加入spike_dict中
		self.spikes_dict = self._get_spike_waveform(filtered_data, spikes_dict, self.before_max, self.after_max, self.plot)
		# 获取spike的template
		self.spike_template = self._get_spike_template(spikes_dict, self.before_max, self.after_max, self.plot)
		# 用提取的template代替spike的位置
		self.replace_data, self.remain_data = self._replace_spikes_with_template(filtered_data, spikes_dict, self.spike_template, self.before_max, self.after_max, self.plot)

	def _highpass(self, data, fs=1000, cutoff=100, plot=False):
		'''
		高通滤波 可做可不做，看数据状态
		'''
		filtered_data = Filter.butter_highpass_filter(data, cutoff, fs)

		if plot: 
			plt.figure()
			plt.subplot(2,1,1)
			plt.title('Raw')
			plt.plot(data)
			plt.ylim(self.minimum_data, self.maximum_data)

		if plot:
			plt.subplot(2,1,2)
			plt.title('Filter')
			plt.plot(filtered_data)
			plt.ylim(self.minimum_data, self.maximum_data)
			plt.show()

		return filtered_data

	def _spike_threshold(self, filtered_data, threshold=0.2, overlap=10, plot=False):
		'''
		输入:
		threshold = 0.2 # 提取幅值高度
		overlap = 10 #两个spike间的间隔，overlap为两个提取的spike之间的最小距离
		
		输出:
		return 提取的spikes，注意spike是按照1,2,3...排序的从1开始的
		'''
		spikes_dict = SD.spike_detect_threshold(filtered_data, threshold, overlap)
		nb_spikes = len(spikes_dict)
		if plot:
			plt.figure()
			plt.plot(filtered_data, 'b')
			for s in spikes_dict.keys():
			    x = np.arange(spikes_dict[s][0], spikes_dict[s][1])
			    y = filtered_data[spikes_dict[s][0] : spikes_dict[s][1]]
			    plt.plot(x, y, 'r')
			    plt.ylim(self.minimum_data, self.maximum_data)
			plt.show()

		'''
		提取每段spike绝对值最大的地方，并将最大值的地方也存储到spikes_dict中
		'''
		for s in spikes_dict.keys():
		    y = filtered_data[spikes_dict[s][0] : spikes_dict[s][1]]
		    max_value_index, _ = util.get_value_max(y)
		    spikes_dict[s].append(spikes_dict[s][0] + max_value_index)

		if plot:
			plt.plot(filtered_data, 'b')
			plt.ylim(self.minimum_data, self.maximum_data)
			for s in spikes_dict.keys():
				plt.axvline(x=spikes_dict[s][2], color='k', linestyle='--')

			plt.show()

		return spikes_dict


	def _get_intervals(self, spikes_dict, save_path=None, plot=False):
		'''
		求出spike间的间隔的列表
		'''
		intervals = []
		for i in range(1, len(spikes_dict.keys())):
		    inter = spikes_dict[i+1][2] - spikes_dict[i][2]
		    intervals.append(inter)

		if plot:
			plt.figure()
			plt.title('Intervals')
			plt.hist(np.array(intervals), 100)
			plt.show()

		# 保存spike的间隔
		if save_path is not None:
			np.save(save_path, intervals)
			print('saved the interpulse intervals')

		return
	

	def _get_spike_waveform(self, filtered_data, spikes_dict, before_max=5, after_max=5, plot=False):
		'''
		提取所有spike的波形
		想法:
		从最大值两端取出一定的宽度，不足的地方用零来填补
		注意函数的输入参数！
		输入:
		before_max: 最大值前面取出多少作为waveform
		after_max: 最大值后面取出多少作为wavefrom

		备注：
			对于第n个spike
			spike_dict[n][0] spike的起始时间
			spike_dict[n][1] spike的终止时间
			spike_dict[n][2] spike的最大值时间
			spike_dict[n][3] spike的template，是个numpy array
		'''
		nb_spikes = len(spikes_dict)
		print('No.spikes:', nb_spikes)

		for s in spikes_dict.keys():
		    y = filtered_data[spikes_dict[s][0] : spikes_dict[s][1]]
		    y_template = SD.get_template(y, spikes_dict[s][2] - spikes_dict[s][0], before_max, after_max)
		    spikes_dict[s].append(y_template)

		if plot:
			random_pick = 8 # 从里面抽取多少个template来输出
			random_choise_index = np.random.choice(np.arange(nb_spikes), random_pick)
			
			plt.figure()
			for i, index in enumerate(random_choise_index):
				plt.subplot(random_pick, 1, i + 1)
				plt.plot(spikes_dict[index + 1][3])
			plt.show()

		return spikes_dict

	def _get_spike_template(self, spikes_dict, before_max=5, after_max=5, plot=False):
		'''
		提取出的spike_template合成为一个spike_template，直接相加平均
		'''
		nb_spikes = len(spikes_dict)
		spike_template = np.zeros(before_max + 1 + after_max)
		for s in spikes_dict.keys():
			spike_template += spikes_dict[s][3]
		spike_template = spike_template / nb_spikes
		if plot:
			plt.figure()
			plt.title('Template')
			plt.plot(spike_template)
			plt.show()
		return spike_template

	def _replace_spikes_with_template(self, filtered_data, spikes_dict, spike_template, before_max=5, after_max=5, plot=False):
		'''
		用提取到的template代替原来的信号部分
		'''
		replaced_data = np.zeros(len(filtered_data))

		for s in spikes_dict.keys():
			spike_time = spikes_dict[s][2]
			spike_time_before = min(spike_time,  before_max)
			spike_time_after = min(len(replaced_data) - spike_time - 1, after_max)
			spike_pick = spike_template[before_max - spike_time_before : before_max + spike_time_after + 1]
			replaced_data[ spike_time - spike_time_before : spike_time + spike_time_after + 1] = spike_pick

		if plot:
			plt.figure()
			plt.title('Replaced')
			plt.plot(replaced_data)
			plt.ylim(self.minimum_data, self.maximum_data)
			plt.show() 

		'''
		提取出去剩余的部分
		'''
		remain_data = filtered_data - replaced_data

		if plot:
			plt.figure()
			plt.title('Remain')
			plt.plot(remain_data)
			plt.ylim(self.minimum_data, self.maximum_data)
			plt.show()

		return replaced_data, remain_data

if __name__ == '__main__':
	file_path = './data/EMG/EMG1.npy'
	raw_data = np.load(file_path)
	raw_data = raw_data[0:32]
	for i, raw_data_row in enumerate(raw_data):
		sorter = spike_sorting(raw_data_row, plot=True)
		sorter.go()
		for s in sorter.spikes_dict:
			spikeTime = sorter.spikes_dict[s][2]
			plt.vlines(spikeTime, 0, 1, linewidth=8, color='blue')
			plt.xlim(0, raw_data.shape[1])
	plt.show()
