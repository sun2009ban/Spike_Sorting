import numpy as np
import Spike_Detection as SD
import Filter
import matplotlib.pyplot as plt
import Utilities as util
import pdb
import os

def spike_sorting(raw_data, inter_save_filename, plot=False):
	'''
	读取数据
	'''
	raw_data = np.squeeze(raw_data)

	#减去平均
	raw_data = raw_data - np.mean(raw_data)
	maximum_data = max(raw_data)
	minimum_data = min(raw_data)

	fs = 1000
	if plot: 
		plt.figure()
		plt.subplot(2,1,1)
		plt.title('Raw')
		plt.plot(raw_data)
		plt.ylim(minimum_data, maximum_data)
	'''
	高通滤波 可做可不做，看数据状态
	'''
	DO_FILTER = True
	if DO_FILTER:
	    cutoff = 100
	    filtered_data = Filter.butter_highpass_filter(raw_data, cutoff, fs)
	else:
	    filtered_data = raw_data

	if plot:
		plt.subplot(2,1,2)
		plt.title('Filter')
		plt.plot(filtered_data)
		plt.ylim(minimum_data, maximum_data)
		plt.show()

	'''
	提取spikes，注意spike是按照1,2,3...排序的从1开始的
	'''
	threshold = 0.2 # 幅值高度
	overlap = 10 #两个spike间的间隔
	spikes_dict = SD.spike_detect_threshold(filtered_data, threshold, overlap)
	nb_spikes = len(spikes_dict)
	if plot:
		plt.figure()
		plt.plot(filtered_data, 'b')
		for s in spikes_dict.keys():
		    x = np.arange(spikes_dict[s][0], spikes_dict[s][1])
		    y = filtered_data[spikes_dict[s][0] : spikes_dict[s][1]]
		    plt.plot(x, y, 'r')
		    plt.ylim(minimum_data, maximum_data)
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
		plt.ylim(minimum_data, maximum_data)
		for s in spikes_dict.keys():
			plt.axvline(x=spikes_dict[s][2], color='k', linestyle='--')

		plt.show()


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
	#np.save(os.path.join('./interval', inter_save_filename), intervals)
	#print('saving: ', inter_save_filename)
	

	return spikes_dict

	'''
	提取所有spike的统一模板
	想法:
	从最大值两端取出一定的宽度，不足的地方用零来填补
	注意函数的输入参数！
	备注：
		对于第n个spike
		spike_dict[n][0] spike的起始时间
		spike_dict[n][1] spike的终止时间
		spike_dict[n][2] spike的最大值时间
		spike_dict[n][3] spike的template，是个numpy array
	'''
	before_max = 5
	after_max = 5
	for s in spikes_dict.keys():
	    y = filtered_data[spikes_dict[s][0] : spikes_dict[s][1]]
	    y_template = SD.get_template(y, spikes_dict[s][2] - spikes_dict[s][0], before_max, after_max)
	    spikes_dict[s].append(y_template)

	random_pick = 8 # 从里面抽取多少个template来输出
	random_choise_index = np.random.choice(np.arange(nb_spikes), random_pick)
	print('No.spikes:', nb_spikes)

	plt.figure()
	for i, index in enumerate(random_choise_index):
		plt.subplot(random_pick, 1, i + 1)
		plt.plot(spikes_dict[index + 1][3])
		plt.ylim(minimum_data, maximum_data)
	plt.show()


	'''
	提取出的spike_template合成为一个spike_template，直接相加平均
	'''
	spike_template = np.zeros(before_max + 1 + after_max)
	for s in spikes_dict.keys():
		spike_template += spikes_dict[s][3]
	spike_template = spike_template / nb_spikes
	plt.figure()
	plt.title('Template')
	plt.plot(spike_template)
	plt.ylim(minimum_data, maximum_data)
	plt.show()


	'''
	用提取到的spike代替原来的信号部分
	'''
	replaced_data = np.zeros(len(filtered_data))

	for s in spikes_dict.keys():
		spike_time = spikes_dict[s][2]
		spike_time_before = min(spike_time,  before_max)
		spike_time_after = min(len(replaced_data) - spike_time - 1, after_max)
		spike_pick = spike_template[before_max - spike_time_before : before_max + spike_time_after + 1]
		replaced_data[ spike_time - spike_time_before : spike_time + spike_time_after + 1] = spike_pick

	plt.figure()
	plt.title('Replaced')
	plt.plot(replaced_data)
	plt.ylim(minimum_data, maximum_data)
	plt.show() 

	save_path = file_path[:-4] + '_sorted.npy'
	np.save(save_path, replaced_data)


	'''
	提取出去剩余的部分
	'''
	remain_data = filtered_data - replaced_data
	plt.figure()
	plt.title('Remain')
	plt.plot(remain_data)
	plt.ylim(minimum_data, maximum_data)
	plt.show()


if __name__ == '__main__':
	file_path = './data/MUAP/result_.npy'
	raw_data = np.load(file_path)
	raw_data = raw_data[0:32]
	plt.figure()
	nb_plots = len(raw_data)
	for i, raw_data_row in enumerate(raw_data):
		ax = plt.subplot(nb_plots, 1, i+1)
		plt.xticks([])
		plt.yticks([])
		ax.spines['bottom'].set_linewidth(0.5)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['right'].set_visible(False)
		spike_dict = spike_sorting(raw_data[i], 'inter_' + str(i) + '.npy', False)
		for s in spike_dict:
			spikeTime = spike_dict[s][2]
			plt.vlines(spikeTime, 0, 1, linewidth=8, color='blue')
			plt.xlim(0, raw_data.shape[1])
	plt.show()