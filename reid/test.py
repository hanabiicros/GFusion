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

pynvml.nvmlInit()

a = [0,1,2,0,5]
a = np.array(a)
b = np.nonzero(a)
print(b[0])

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