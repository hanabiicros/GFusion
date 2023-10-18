from collections import defaultdict
import os
from tkinter import E
import numpy as np
import torch
import networkx as nx
import scipy.sparse as sp
import pynvml
import sys
from sklearn.metrics import euclidean_distances
import torch.nn.functional as F
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline
from random import sample
# 直方图
from scipy.stats import norm #使用直方图和最大似然高斯分布拟合绘制分布

# mi = -50000
# ma = 50000
# step = 1
# deltas = pickle.load(open("/home/zyb/projects/HGO_v2/data/DukeMTMC-reID_market-train/sorted_false_cam1_cam2_deltas.pickle", 'rb'))
# xx = list(range(mi,ma,step))
# # for i in range(len(xx)):
# #     xx[i] = xx[i]/10
# y1 = []
# y2 = []
# y3 = []
# y4 = []
# y5 = []
# x1 = deltas[0][1]
# x1 = [x for x in x1 if x >= mi and x <= ma]
# x2 = deltas[0][2]
# x2 = [x for x in x2 if x >= mi-25 and x <= ma+25]
# x3 = deltas[0][3]
# x3 = [x for x in x3 if x >= mi-25 and x <= ma+25]
# x4 = deltas[0][4]
# x4 = [x for x in x4 if x >= mi-25 and x <= ma+25]
# x5 = deltas[0][5]
# x5 = [x for x in x5 if x >= mi-25 and x <= ma+25]

# for i in xx:
#     x1_1 = [x for x in x1 if x >= i-25 and x <= i+25]
#     x2_1 = [x for x in x2 if x >= i-25 and x <= i+25]
#     x3_1 = [x for x in x3 if x >= i-25 and x <= i+25]
#     x4_1 = [x for x in x4 if x >= i-25 and x <= i+25]
#     x5_1 = [x for x in x5 if x >= i-25 and x <= i+25]
#     y1.append(len(x1_1))
#     y2.append(len(x2_1))
#     y3.append(len(x3_1))
#     y4.append(len(x4_1))
#     y5.append(len(x5_1))

# x1 = sample(x1,63)
# sns.kdeplot(data=x1,label='cam 1-2',color='#cc0000', linewidth = 3,linestyle=(0,(5,10)))

# plt.plot(xx,y1,marker = 'o',color="#bce672",linestyle="-",markersize=0.1,linewidth=1,label="cam 0-1")
# plt.plot(xx,y2,marker = 'o',color="#ffa631",linestyle="-",markersize=0.1,linewidth=1,label="cam 0-2")
# plt.plot(xx,y3,marker = 'o',color="#177cb0",linestyle="-",markersize=0.1,linewidth=1,label="cam 0-3")
# plt.plot(xx,y4,marker = 'o',color="#c32136",linestyle="-",markersize=0.1,linewidth=1,label="cam 0-4")
# plt.plot(xx,y5,marker = 'o',color="#5d513c",linestyle="-",markersize=0.1,linewidth=1,label="cam 0-5")
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# fig,ax = plt.subplots()
# n,bins_num,pat = ax.hist([x1,x2,x3,x4,x5],color=["#bce672","#ffa631","#177cb0","#c32136","#5d513c"],label=['cam 0-1','cam 0-2', 'cam 0-3', 'cam 0-4', 'cam 0-5'],bins=range(mi,ma,step), alpha=0)
# x = bins_num[:len(range(mi,ma,step))-1]

# ax.plot(x,n[0],marker = 'o',color="#bce672",linestyle="-",markersize=0.3,linewidth=1,label="cam 0-1")
# ax.plot(x,n[1],marker = 'o',color="#ffa631",linestyle="-",markersize=0.3,linewidth=1,label="cam 0-2")
# ax.plot(x,n[2],marker = 'o',color="#177cb0",linestyle="-",markersize=0.3,linewidth=1,label="cam 0-3")
# ax.plot(x,n[3],marker = 'o',color="#c32136",linestyle="-",markersize=0.3,linewidth=1,label="cam 0-4")
# ax.plot(x,n[4],marker = 'o',color="#5d513c",linestyle="-",markersize=0.3,linewidth=1,label="cam 0-5")

# plt.hist([x1,x2],bins=len(x1),color=["#bce672","#ffa631"],label=["cam 0-1","cam 0-2"])
# sns.displot(data=x1,fill=False,label='cam 0-1',color='#bce672',kde=True)
# sns.displot(data=[{"tp":0, "dt":x1},{"tp":1, "dt":x2}],x="dt",fill=False,element="poly",bins=len(x1),hue="tp")
# sns.displot(data=x2,x="t",fill=False,element="poly",color='#ffa631',bins=len(x2),label='cam 0-2',hue='cam to cam')
# sns.displot(data=x3,x="t",fill=False,element="poly",color='#177cb0',bins=len(x3),label='cam 0-3',hue='cam to cam')
#ffa631

# plt.tick_params(labelsize=15)
# # plt.xticks(fontsize=10)
# # plt.yticks(fontsize=10)
# plt.ylabel(r'probability',fontsize=30)
# plt.xlabel(r'$\Delta$t',fontsize=30)
# plt.legend(loc='upper right',fontsize=20)  # 显示图例
# plt.savefig("/home/zyb/projects/HGO_v2/figures/fig_1_a.png")

# # 2. 绘图
# plt.scatter(x1,  # 横坐标
#             y1,  # 纵坐标
#             c='#bce672',  # 点的颜色
#             label='cam 0-1',s=5)  # 标签 即为点代表的意思

# plt.scatter(x2, y2, c='#ffa631',label='cam 0-2',s=5) 
# plt.scatter(x3, y3, c='#177cb0',label='cam 0-3',s=5) 
# plt.scatter(x4, y4, c='#c32136',label='cam 0-4',s=5) 
# plt.scatter(x5, y5, c='#5d513c',label='cam 0-5',s=5) 
# # 3.展示图形
# plt.xlabel(r'$\Delta$t')
# plt.legend()  # 显示图例
# plt.savefig("time_interval_true.png")


plt.figure(dpi=800,figsize=(10,8))
# train
# ks = [1,2,3,4,5,6,7,8,9,10]
# M_map_ks = [24.8, 74.9, 75.4, 75.1, 74.9, 74.9, 74.5, 73.3, 72.8, 72.1]
# MSMT17_map_ks = [3.5, 27.9, 26.6, 24.9, 22.8, 21.2, 19.6, 17.6, 16.5, 14.9]
# kl = [10,20,30,40,50,60,70,80,90,100]
# M_map_kl = [68.9, 73.5, 75.3, 75.0, 75.6, 75.4, 75.0, 75.2, 75.2, 75.2]
# MSMT17_map_kl = [18.2, 23.4, 24.1, 25.6, 26.2, 26.6, 26.6, 26.9, 27.4, 27.3]

# test
ks = [1,2,3,4,5,6,7,8,9,10]
M_map_ks = [54.8, 83.9, 85.9, 86.6, 86.5, 86.4, 85.9, 85.1, 84.4, 83.6]
MSMT17_map_ks = [15.5, 53.2, 52.1, 50.6, 49.2, 47.7, 46.5, 45.3, 44.1, 43.0]

kl = [10,20,30,40,50,60,70,80,90,100]
M_map_kl = [83.5, 85.6, 86.0, 86.2, 86.1, 85.9, 85.7, 85.4, 85.3, 85.0]
MSMT17_map_kl = [40.0, 46.8, 49.5, 50.8, 50.9, 52.1, 52.4, 52.5, 52.6, 52.7]



#设置折点属性
plt.tick_params(labelsize=20)
plt.plot(kl, M_map_kl, c='#177cb0', ls='-',  marker='o',markersize=10)
plt.plot(kl, MSMT17_map_kl, c='#ffa400', ls='-', marker='s',markersize=10)

plt.xlabel("$k_{l}$",fontsize=30)
plt.ylabel("mAP(%)",fontsize=30)
plt.legend(('Market-1501', 'MSMT17'), loc='best',fontsize=20)  
plt.title('mAP',fontsize=30) 

plt.savefig("/home/zyb/projects/HGO_v2/figures/test_kl_map.png")

# pynvml.nvmlInit()

# a = [0,1,2,0,5]
# a = np.array(a)
# b = np.nonzero(a)
# print(b[0])

# vg_features = torch.load("/home/zyb/projects/h-go/logs/old/d2m/train_graphsage/vision/train_g_features.pth")
# vg_features = F.normalize(vg_features, p=2, dim=1)
# sim = []
# features = torch.FloatTensor(vg_features)
# for i in range(len(vg_features)):
#     query = features[i].view(-1,1) 
#     score = torch.mm(features,query) 
#     score = score.squeeze(1).cpu()
#     score = score.numpy()
#     sim.append(score)

# sim = np.array(sim)


# sim = np.exp(-dist_m)
# sim = F.cosine_similarity(vg_features[:10].unsqueeze(1), vg_features[:10].unsqueeze(0), dim=-1)
# print(sim)

# cams1 = [1,2]
# cams1 = np.array(cams1)
# flags = []
# for i in range(len(cams)):
#     flag = (cams1[i] == cams[i])
#     flags.append(flag)

# a = sp.eye(3,format='lil')
# a[0,0] = 2

# a = [[1,1],[2,3.3333333333]]
# a = sp.lil_matrix(a, dtype=np.float32)
# tuples = zip(a.row,a.col,a.data)
# b = sorted(tuples, key=lambda x: (x[0], -x[2]))

# sparse_m = sp.bsr_matrix(np.array([[1,0,0,0,1], [1,0,0,0,1]]))
# sparse_m = torch.from_numpy(sparse_m.toarray())
# print(sparse_m)
# b = sp.coo_matrix(a, dtype=np.float32)
# indices = torch.LongTensor([[0,0], [1,1], [2,2]])#稀疏矩阵中非零元素的坐标
# indices = indices.t() #一定要转置，因为后面sparse.FloatTensor的第一个参数就是该变量，要求是一个含有两个元素的列表，每个元素也是一个列表。第一个子列表是非零元素所在的行，第二个子列表是非零元素所在的列。

# values = torch.FloatTensor([3,4,5])
# mat = torch.sparse.FloatTensor(indices,values,[4,4])
# print(mat)
# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense()
# device = torch.device("cuda", 0)
# y = torch.randn((200, 200, 200, 200), device=device)
# y = y.cpu()
# y = torch.randn((200, 200, 200, 200), device=device)
# y = y.cpu()

# y = torch.randn((200, 200, 200, 200), device=device)
# y = torch.randn((200, 200, 200, 200), device=device)
# y = torch.randn((200, 200, 200, 200), device=device)
# y = torch.randn((200, 200, 200, 200), device=device)
# y = torch.randn((200, 200, 200, 200), device=device)
# y = torch.randn((200, 200, 200, 200), device=device)
# y = torch.randn((200, 200, 200, 200), device=device)

# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# print(meminfo.used)

# torch.cuda.empty_cache()

# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
# print(meminfo.used)
# a = np.ones([3,3], dtype=np.float64)
# print(a[0][:1])
# b = np.ones([3,3], dtype=np.float64)

# p = torch.zeros((1,1))
# a = torch.from_numpy(a)

# c = np.ones([1,4])
# c = torch.from_numpy(c)
# b = torch.cat((a,p), dim = 1)
# d = torch.cat((b,c), dim = 0)
# print(d)
# a = np.array([[1,2.1],[3,3]])
# b = np.sum(a,axis=1)
# for i,ai in enumerate(a):
#     a[i] /= b[i]
# print(a)
# b = np.array([4,5,6])
# print(np.where((b >=5) & (b < 6)))
# # for ind,j in enumerate(a):
# #     print(ind)
# # a=[[1.2,2.1],[1.4,2.6]]
# # a = torch.tensor(a)

# # print(a)
# g = nx.Graph()
# g.add_edge('img1','img3', weight = 1)
# g.add_edge('img1','img3', weight = 2)
# print(g.edges.data("weight"))
# g.add_edge('img2','img3', weight = 1)
# g.add_edge('img2','img4', weight = 1)
# g.add_node('casc')
# print(g.nodes())
# q_f = ['img1','img2']
# g_f = ['img3','img4']

# a = q_f.copy()
# a.remove('img1')
# print(q_f)
# g = nx.convert_node_labels_to_integers(g)
# mapping = {old_label:new_label for new_label, old_label in enumerate(g.nodes())}
# H = nx.relabel_nodes(g, mapping)
# print(H.nodes())
# print(np.where(np.in1d(g.nodes(),q_f))[0])

# for i, x in enumerate(a):
#     for j,y in enumerate(x):
#         print(y)
# def safe_link(src, dst):
#     if os.path.islink(dst):
#         os.unlink(dst)
#     os.symlink(src, dst)

# working_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# source_dir = os.path.join(working_dir, 'eval/market2duke')
# print(source_dir)
# safe_link('/home/zyb/projects/h-go/eval/market2duke-train/pid.txt', 'data/market_DukeMTMC-reID-train/renew_pid.log')