import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

FILE_EXTENSIONS = [".npy"]

def is_file(filename):
    '''
    判断filename是否是以FILE_EXTENSIONS中的为结尾
    '''
    return any(filename.endswith(extension) for extension in FILE_EXTENSIONS)

def walk_through_dir(directory):
    '''
    遍历目录dir下面的以FILE_EXTENSIONS为结尾的文件
    返回值为文件的路径
    '''
    file_path = []

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):
            if is_file(fname):
                path = os.path.join(directory, fname) #把目录和
                file_path.append(path)

    return file_path



if __name__ == '__main__':
	intervals = []

	data_dir = './real'
	file_paths = walk_through_dir(data_dir)
	for filename in file_paths:
		inter = np.load(filename)
		intervals.append(inter)

	intervals = np.concatenate(intervals)
	intervals = intervals
	plt.hist(intervals, bins=20, normed=True, color='c')
	plt.xlim(0, 300)
	plt.ylim(0, 0.02)
	plt.yticks(np.arange(0, 0.021, 0.004))
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlabel('Inter-pulse Interval (ms)', fontsize=14)
	plt.show()