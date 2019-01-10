import numpy as np
import Spike_Detection as SD
import Filter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import Utilities as util
import pdb
import os
from matplotlib import cm

class spike_sorting(object):
	#输出图片
	fig_1 = Figure(figsize=(5,2), dpi=100) # _highpass
	fig_2 = Figure(figsize=(5,2), dpi=100) #_spike_threshold
	fig_3 = Figure(figsize=(5,2), dpi=100) #_get_intervals
	fig_4 = Figure(figsize=(5,2), dpi=100) #_get_spike_waveform
	fig_5 = Figure(figsize=(5,2), dpi=100) #_get_spike_template
	fig_6 = Figure(figsize=(5,2), dpi=100) #_replace_spikes_with_template
	fig_7 = Figure(figsize=(5,2), dpi=100) #_replace_spikes_with_template	
	fig_8 = Figure(figsize=(5,2), dpi=100) #_cluster_spike_waveform

	def __init__(self, raw_data, fs=1000, cutoff=100, threshold=0.1, overlap=10, save_interval_path=None, 
					before_max=5, after_max=5, nb_clusters=2, plot=False):
		# 只接受一维度数据
		self.raw_data = np.squeeze(raw_data)
		assert len(self.raw_data.shape) == 1, "only vector is allowed" 
		#减去平均
		self.raw_data = raw_data - np.mean(raw_data)

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

		self.nb_clusters = nb_clusters # 给spike分成多少个类
		self.color = cm.coolwarm(np.linspace(0.1,0.9,nb_clusters))		

	def go(self):
		# 数据高通滤波
		filtered_data = self._highpass(self.raw_data, self.fs, self.cutoff, self.plot)
		
		# 提取spike，把性质存储在dict中
		spikes_dict = self._spike_threshold(filtered_data, self.threshold, self.overlap, self.plot)
		
		# 获取每个spike的waveform，加入spike_dict中
		self.spikes_dict = self._get_spike_waveform(filtered_data, spikes_dict, self.before_max, self.after_max, self.plot)
		
		# 依据spike的waveform进行分类, 加入spike_dict中
		self.spikes_dict = self._get_spike_cluster(self.spikes_dict, self.nb_clusters, self.plot)

		# 计算和存储intervals
		self._get_intervals(spikes_dict, self.save_interval_path, self.plot)
		
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
			subplot_1 = spike_sorting.fig_1.add_subplot(2,1,1, title='Raw')
			subplot_1.plot(data)

			subplot_2 = spike_sorting.fig_1.add_subplot(2,1,2, title='Filtered')
			subplot_2.plot(filtered_data)

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
			subplot = spike_sorting.fig_2.add_subplot(111)
			subplot.plot(filtered_data, 'b')
			for s in spikes_dict.keys():
			    x = np.arange(spikes_dict[s][0], spikes_dict[s][1])
			    y = filtered_data[spikes_dict[s][0] : spikes_dict[s][1]]
			    subplot.plot(x, y, 'r')

		'''
		提取每段spike绝对值最大的地方，并将最大值的地方也存储到spikes_dict中
		'''
		for s in spikes_dict.keys():
		    y = filtered_data[spikes_dict[s][0] : spikes_dict[s][1]]
		    max_value_index, _ = util.get_value_max(y)
		    spikes_dict[s].append(spikes_dict[s][0] + max_value_index)

		if plot:
			subplot.plot(filtered_data, 'b')
			for s in spikes_dict.keys():
				subplot.axvline(x=spikes_dict[s][2], color='k', linestyle='--')

		return spikes_dict


	def _get_intervals(self, spikes_dict, save_path=None, plot=False):
		'''
		求出spike间的间隔的列表
		'''
		all_intervals = {}
		for cluster in range(self.nb_clusters):
			spike_times = []
			for s in spikes_dict.keys():
				if spikes_dict[s][4] == cluster:
					spike_times.append(spikes_dict[s][2])
			intervals = []
			if len(spike_times) > 1:
				for i in range(len(spike_times) - 1):
					intervals.append(spike_times[i+1] - spike_times[i])
			else:
				intervals.append(0)
			all_intervals[cluster] = intervals

		if plot:
			for label in all_intervals.keys():
				if label == 0:
					subplot = spike_sorting.fig_3.add_subplot(self.nb_clusters, 1, label + 1, title='Intervals')
					subplot.hist(np.array(all_intervals[label]), 100, color=self.color[label])
				else:
					subplot = spike_sorting.fig_3.add_subplot(self.nb_clusters, 1, label + 1)
					subplot.hist(np.array(all_intervals[label]), 100, color=self.color[label])

		# 保存spike的间隔
		if save_path is not None:
			np.save(save_path, all_intervals)
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
			
			for i, index in enumerate(random_choise_index):
				if i == 0:
					subplot = spike_sorting.fig_4.add_subplot(random_pick, 1, i + 1, title='Waveforms')
					subplot.plot(spikes_dict[index + 1][3])
				else:
					subplot = spike_sorting.fig_4.add_subplot(random_pick, 1, i + 1)
					subplot.plot(spikes_dict[index + 1][3])					

		return spikes_dict

	def _get_spike_template(self, spikes_dict, before_max=5, after_max=5, plot=False):
		'''
		提取出的spike_template合成为一个spike_template，直接相加平均
		'''
		spike_template_dict = {}

		for cluster in range(self.nb_clusters):
			nb_spikes = 0
			spike_template = np.zeros(before_max + 1 + after_max)
			for s in spikes_dict.keys():
				if spikes_dict[s][4] == cluster:
					spike_template += spikes_dict[s][3]
					nb_spikes += 1
			spike_template = spike_template / nb_spikes
			spike_template_dict[cluster] = spike_template

		if plot:
			for label in spike_template_dict.keys():
				if label == 0:
					subplot = spike_sorting.fig_5.add_subplot(self.nb_clusters, 1, label + 1, title='Template')
					subplot.plot(spike_template_dict[label], color=self.color[label])
				else:
					subplot = spike_sorting.fig_5.add_subplot(self.nb_clusters, 1, label + 1)
					subplot.plot(spike_template_dict[label], color=self.color[label])					


		return spike_template_dict

	def _get_spike_cluster(self, spikes_dict, nb_clusters, plot=False):
		'''
		spikes_dict[n][4] 是每个spike的label
		'''
		from Spike_Sorting import spike_sorting
		from sklearn.cluster import KMeans

		from sklearn.decomposition import PCA
		from mpl_toolkits.mplot3d import Axes3D

		def get_features(data):
			'''
			wavelet, frequency, autoregression
			'''
			features = np.copy(data)

			return features

		def get_kmeans(data, get_features, nb_clusters=2):
			'''
			data: 原始数据 nb_samples x 数据长度
			get_features: 从数据中提取出features的函数
			nb_clusters: 分类的数目
			'''
			features = get_features(data)
			if nb_clusters > 1:
				kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(features)
				labels = kmeans.labels_
			else:
				labels = np.zeros(data.shape[0])

			return features, labels

		waveforms = []
		for s in spikes_dict.keys():
			waveforms.append(np.reshape(spikes_dict[s][3], (1, -1) ))
		waveforms = np.concatenate(waveforms)

		features, labels = get_kmeans(waveforms, get_features, nb_clusters)
		
		for s in spikes_dict.keys():
			spikes_dict[s].append(labels[s-1])

		if plot:
			pca = PCA(n_components=3)
			features_pca = pca.fit_transform(features)
			subplot = spike_sorting.fig_8.add_subplot(111, title='Clustering', projection='3d')
			subplot.scatter(features_pca[:, 2], features_pca[:, 0], features_pca[:, 1], c=self.color[labels], edgecolor='k')		

		return spikes_dict


	def _replace_spikes_with_template(self, filtered_data, spikes_dict, spike_template_dict, before_max=5, after_max=5, plot=False):
		'''
		用提取到的template代替原来的信号部分
		'''
		replaced_data = np.zeros(len(filtered_data))

		for s in spikes_dict.keys():
			spike_time = spikes_dict[s][2]
			spike_time_before = min(spike_time,  before_max)
			spike_time_after = min(len(replaced_data) - spike_time - 1, after_max)
			spike_label = spikes_dict[s][4]
			spike_pick = spike_template_dict[spike_label][before_max - spike_time_before : before_max + spike_time_after + 1]
			replaced_data[ spike_time - spike_time_before : spike_time + spike_time_after + 1] = spike_pick

		if plot:
			subplot = spike_sorting.fig_6.add_subplot(111, title='Replaced')
			subplot.plot(replaced_data)

		'''
		提取出去剩余的部分
		'''
		remain_data = filtered_data - replaced_data

		if plot:
			subplot = spike_sorting.fig_7.add_subplot(111, title='Remain')
			subplot.plot(remain_data)

		return replaced_data, remain_data

	@staticmethod
	def _clear_figure():
		spike_sorting.fig_1.clf()
		spike_sorting.fig_2.clf()
		spike_sorting.fig_3.clf()
		spike_sorting.fig_4.clf()
		spike_sorting.fig_5.clf()
		spike_sorting.fig_6.clf()
		spike_sorting.fig_7.clf()
		spike_sorting.fig_8.clf()


if __name__ == '__main__':
	file_path = './data/EMG/EMG1.npy'
	raw_data = np.load(file_path)

	sorter = spike_sorting(raw_data, plot=True)
	sorter.go()
	for s in sorter.spikes_dict:
		spikeTime = sorter.spikes_dict[s][2]
		plt.vlines(spikeTime, 0, 1, linewidth=8, color='blue')
		plt.xlim(0, raw_data.shape[0])
	plt.show()
