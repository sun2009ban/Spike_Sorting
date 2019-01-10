import numpy as np
from Spike_Sorting import spike_sorting
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import Filter
import pdb

def get_features(data):
    features = np.copy(data)
    return features

def kmeans(data, get_features, nb_clusters=2, pca_components=3):
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

    pca = PCA(pca_components)
    features_pca = pca.fit_transform(features)

    return features, features_pca, labels


fs = 1000
cutoff = 100
data_dims = 512
threshold = 0.1
overlap = 3

raw_data = np.load('./data/S00201_resample_norm.npy')
filtered_data = Filter.butter_highpass_filter(raw_data, cutoff, fs)

filtered_data = filtered_data[:data_dims]
plt.plot(filtered_data)
plt.show()

spike_sorting_class = spike_sorting(filtered_data, threshold=threshold, overlap=overlap, plot=True)
spike_sorting_class.go()

nb_clusters = 7
color = cm.coolwarm(np.linspace(0.1,0.9,nb_clusters))

spikes_dict = spike_sorting_class.spikes_dict
waveforms = []

for s in spikes_dict.keys():
    waveforms.append(np.reshape(spikes_dict[s][3], (1, -1) ))

waveforms = np.concatenate(waveforms)

features, features_pca, labels = kmeans(waveforms, get_features, nb_clusters, 3)


# 显示分类
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(features_pca[:, 2], features_pca[:, 0], features_pca[:, 1], c=color[labels], edgecolor='k')
plt.show()

# 显示结果
plt.figure()
plt.plot(filtered_data)
for s in spikes_dict.keys():
    x_axis = np.arange(spikes_dict[s][2] - 6, spikes_dict[s][2] + 4)
    plt.plot(x_axis, filtered_data[x_axis], color=color[labels[s-1]])

plt.show()
