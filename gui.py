import numpy as np
import tkinter as tk
from tkinter import filedialog
import pdb
from Spike_Sorting import spike_sorting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #用于matplotlib和tkinter交互的界面
import os

file_path = None # global 是用来记录文件位置的
threshold = None
overlap = None
spike_sorting_class = None
save_dir = None

root = tk.Tk() # 创建一个tkinter的窗口
root.title('Spike Sorting!')
root.geometry("1000x500")
#root.withdraw() # 关闭这个窗口

# 打开原始文件相关的
open_button_label = tk.Label(root, text='choose your file path', width=20, height=2)
open_button_label.pack()

tk_file_path = tk.StringVar() # show 显示文件路径的variable

def _return_path():
    global file_path
    file_path = filedialog.askopenfilename()
    tk_file_path.set(file_path)

open_button = tk.Button(root, textvariable=tk_file_path, width=100, height=2, command=_return_path)
open_button.pack()

# 输入相关属性设置
entry_threhold = tk.Entry(root, show=None)
entry_overlap = tk.Entry(root, show=None)
entry_cluster = tk.Entry(root, show=None)

entry_threhold_label = tk.Label(root, text='threhold', width=8, height=1)
entry_overlap_label = tk.Label(root, text='overlap', width=8, height=1)
entry_cluster_label = tk.Label(root, text='cluster', width=8, height=1)

entry_threhold_label.place(x=0, y=100)
entry_threhold.place(x=100, y=100)
entry_overlap_label.place(x=200, y=100)
entry_overlap.place(x=300, y=100)
entry_cluster_label.place(x=400, y=100)
entry_cluster.place(x=500, y=100)

canvas_1 = FigureCanvasTkAgg(spike_sorting.fig_1, master=root)
canvas_2 = FigureCanvasTkAgg(spike_sorting.fig_2, master=root)
canvas_3 = FigureCanvasTkAgg(spike_sorting.fig_3, master=root)
canvas_4 = FigureCanvasTkAgg(spike_sorting.fig_4, master=root)
canvas_5 = FigureCanvasTkAgg(spike_sorting.fig_5, master=root)
canvas_6 = FigureCanvasTkAgg(spike_sorting.fig_6, master=root)
canvas_7 = FigureCanvasTkAgg(spike_sorting.fig_7, master=root)
canvas_8 = FigureCanvasTkAgg(spike_sorting.fig_8, master=root)

canvas_1.get_tk_widget().place(x=0, y=200)
canvas_2.get_tk_widget().place(x=500, y=200)
canvas_3.get_tk_widget().place(x=1000, y=200)
canvas_4.get_tk_widget().place(x=0, y=400)
canvas_5.get_tk_widget().place(x=500, y=400)
canvas_6.get_tk_widget().place(x=1000, y=400)
canvas_7.get_tk_widget().place(x=0, y=600)
canvas_8.get_tk_widget().place(x=500, y=600)

# 计算和显示结果
def _spike_sorting():
    global file_path
    global threshold
    global overlap
    global spike_sorting_class

    #获取参数
    threshold = float(entry_threhold.get())
    overlap = int(entry_overlap.get())
    cluster = int(entry_cluster.get())
    print('Threshold: ', threshold)
    print('overlap: ', overlap)
    print('number of clusters: ', cluster)

    assert file_path is not None
    assert threshold is not None 
    assert overlap is not None        

    raw_data = np.load(file_path)
    spike_sorting_class = spike_sorting(raw_data, threshold=threshold, overlap=overlap, nb_clusters=cluster, plot=True)
    spike_sorting_class.go()

    # 输出绘制在画报上面
    canvas_1.draw()
    canvas_2.draw()
    canvas_3.draw()
    canvas_4.draw()
    canvas_5.draw()
    canvas_6.draw()
    canvas_7.draw()
    canvas_8.draw()

spike_sorting_button = tk.Button(root, text='spike sorting', width=15, height=2, command=_spike_sorting)
spike_sorting_button.place(x=700, y=100)

# 放置清除全部绘图的按键
def _clear_canvas():
    spike_sorting._clear_figure()
    # 输出绘制在画报上面
    canvas_1.draw()
    canvas_2.draw()
    canvas_3.draw()
    canvas_4.draw()
    canvas_5.draw()
    canvas_6.draw()
    canvas_7.draw()
    canvas_8.draw()    

clear_button = tk.Button(root, text='clear canvas', width=15, height=2, command=_clear_canvas)
clear_button.place(x=850, y=100)

# 设置保存结果按键
save_dir_label = tk.Label(root, text='choose your save path', width=20, height=2)
save_dir_label.place(x=1000, y=600)

tk_save_dir = tk.StringVar() # show 显示文件路径的variable

def _save_dir():
    global save_dir
    save_dir = filedialog.askdirectory()
    tk_save_dir.set(save_dir)

save_path_button = tk.Button(root, textvariable=tk_file_path, width=20, height=2, command=_save_dir)
save_path_button.place(x=1200, y=600)

def _save():
    np.save(os.path.join(save_dir, 'spike_template.npy'), spike_sorting_class.spike_template)
    np.save(os.path.join(save_dir, 'replace_data.npy'), spike_sorting_class.replace_data)
    np.save(os.path.join(save_dir, 'remain_data.npy'), spike_sorting_class.remain_data)

save_button = tk.Button(root, text='save', width=10, height=2, command=_save)
save_button.place(x=1200, y=700)

root.mainloop()

