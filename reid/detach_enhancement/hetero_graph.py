import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale 
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sci
import os.path as osp
import time
from reid.img_st_fusion import stmain
import pyhocon
from reid.graphsage.dataCenter import *
from reid.graphsage.models import *
from reid.graphsage.utils import *
from reid.evaluation.re_rank import re_ranking
from reid.graphsage.dataCenter import *
import networkx as nx
from reid.ge import DeepWalk,Node2Vec
from reid.models import DGI, LogReg
from reid.utils import process
    


class HG(object):
    eps=0.1
    def __init__(self, lamda, features, real_labels, cams):
        self.features = features
        self.real_labels = real_labels
        self.lamda = lamda
        self.cams = cams
        self.pd_labels = None
        self.neg_ratio = None
        self.cam_2_imgs = []
        self.check_graph = np.zeros([self.real_labels.size,
                                     self.real_labels.size], dtype=self.features.dtype)
        for j, p in enumerate(self.real_labels):
            index = np.where(self.real_labels == p)
            self.check_graph[j, index] = 1.

        # hyper parameter
        self.general_graph = False
        self.homo_ap = False

    def heter_cam_normalization(self, k1_graph):
        for i in range(len(k1_graph)):
            index = np.where(k1_graph[i] != 0.)
            weights = k1_graph[i][index]
            cd_c = self.cams[index]
            tag_c_set = set(cd_c)
            for c in tag_c_set:
                c_index = np.where(cd_c == c)
                w = weights[c_index]

                w = len(w) / len(cd_c) * w / np.sum(w)  
                k1_graph[i][index[0][c_index]] = w
        print(np.sum(k1_graph, axis=1))
        print('heter_cam_normalization')
        return k1_graph

    def row_normalization(self, sim, exp=False):
        if exp:
            return np.exp(sim) / np.sum(np.exp(sim), axis=1)[:, np.newaxis]
        else:
            # todo try np.max
            return sim / np.sum(sim, 1)[:, np.newaxis]
    
    def old_delta_propagation(self, ks=2, kd=4, k2=11, opt=None, indexs = None):
        print("propagation")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)
        # sim = []
        # features = torch.FloatTensor(self.features)
        # for i in range(len(self.features)):
        #     query = features[i].view(-1,1) 
        #     score = torch.mm(features,query) 
        #     score = score.squeeze(1).cpu()
        #     score = score.numpy()
        #     # predict index
        #     # index = np.argsort(score)  #from small to large
        #     # index = index[::-1]
        #     # score = score[index]
        #     # indexs.append(index)
        #     sim.append(score)

        
        # sim = np.array(sim)
        print(sim[0:3, 0:10])

        # for i, person_ap_scores in enumerate(sim):
        #     cur_max_vision = max(person_ap_scores)
        #     cur_min_vision = min(person_ap_scores)
        #     sim[i] = (sim[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
        
        
        st_scores = stmain(opt, indexs, scores = sim, flag = True)

        # for i in range(len(st_scores)):
        #     max_i = max(st_scores[i])
        #     for j,score in enumerate(st_scores[i]):
        #         if score != 0 and score != -1 and score != -2:
        #             st_scores[i][j] /= max_i

        # sim = 0.8 * sim + 0.2 * st_scores

        # # 用贝叶斯融合分数作为相似性分数
        # for i, person_ap_scores in enumerate(sim):
        #     cur_max_vision = max(person_ap_scores)
        #     cur_min_vision = min(person_ap_scores)
        #     sim[i] = (sim[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
        #     # sim[i] = np.exp(sim[i] * 3)
        #     # cur_max_vision = max(sim[i])
        #     # cur_min_vision = min(sim[i])
        #     # sim[i] = (sim[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
        
        # persons_cross_scores = [[0 for i in range(len(St))] for j in range(len(St))]
        # for i, st_scores in enumerate(St):
        #     # cross_scores = list()
        #     for j, st_score in enumerate(st_scores):
        #             # cross_score = St[i][j] * sim[i][j] 
        #             persons_cross_scores[i][j] = St[i][j] * sim[i][j] 
        #             # if i < 5 and persons_cross_scores[i][j] != 0:
        #             #     print(persons_cross_scores[i][j], '\n')
        # #     cross_scores.append(cross_score)
        # # persons_cross_scores.append(cross_scores)
        
        # max_score = max([max(predict_cross_scores) for predict_cross_scores in persons_cross_scores])
        # print('max_cross_score %f' % max_score)

        # for i, person_cross_scores in enumerate(persons_cross_scores):
        #     for j, person_cross_score in enumerate(person_cross_scores):
        #         if person_cross_score > 0: # diff camera
        #             # diff seq not sort and not normalize
        #             persons_cross_scores[i][j] /= max_score
        #         else: # same camera and (diff camere && rand score == -1    )
        #             persons_cross_scores[i][j] = sim[i][j]

        # sim = np.array(persons_cross_scores)

        # # 直接将时空矩阵和视觉矩阵相乘得到相似性矩阵
        # for i, s_scores in enumerate(St):
        #     for j, s_score in enumerate(s_scores):
        #         if s_score != 0:
        #             sim[i][j] *= s_score

        # for i in range(len(st_scores)):
        #     # index = np.where((st_scores[i] != 0.) & (st_scores[i] != -1))
        #     max_i = max(st_scores[i])
        #     min_i = min(st_scores[i])
        #     for j,score in enumerate(st_scores[i]):
        #         if score != 0 and score != -1:
        #             st_scores[i][j] = (st_scores[i][j] - min_i) / (max_i - min_i)
        
        # top_st_index = np.argsort(-st_scores)[:, :20]
        # st_graph = self.kneighbors2(st_scores, kd)   # 这里取top-k作为其邻居的参数k，可以进行调整，然而并没啥效果
        # st_graph = self.heter_cam_normalization(st_graph)

        
        # # k1_graph = self.heteo_stkneighbors(sim, st_scores, ks, 8)
        # # k1_graph = self.heter_cam_normalization(k1_graph)

        # hyper parameter
        # if self.general_graph:
        #     print("----general graph----")
        #     k1_graph = self.kneighbors2(sim, kd)
        # else:
        #     # k1_graph = self.heteo_vst_kneighbors(sim, st_scores, ks, kd)
        #     k1_graph = self.heteo_kneighbors(sim, ks, kd)
        # if self.homo_ap:
        #     print("----homo-ap----")
        #     k1_graph = self.row_normalization(k1_graph, exp=False)
        # else:
        #     k1_graph = self.heter_cam_normalization(k1_graph)

        # k1_graph[np.diag_indices_from(k1_graph)] = 1.
        # k1_graph = self.heter_cam_normalization(k1_graph)

        # print("k1_graph:{}".format(k1_graph[:4,:10]))
        # print("st_graph:{}".format(st_graph[:4,:10]))
        ## DGI

        # training params
        # batch_size = 1
        # # nb_epochs = 10000
        # nb_epochs = 10
        # patience = 20
        # lr = 0.001
        # l2_coef = 0.0
        # drop_prob = 0.0
        # hid_units = 512
        # sparse = False
        # nonlinearity = 'prelu' # special name to separate parameters

        # # adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
        # # features, _ = process.preprocess_features(features)
        # features = self.features

        # nb_nodes = features.shape[0]
        # ft_size = features.shape[1]
        # # print("nb_nodes:{},ft_size:{}".format(nb_nodes,ft_size))
        # # nb_classes = labels.shape[1]

        # # adj = process.normalize_adj(k1_graph)
        # adj = k1_graph
        # # a = np.eye(len(k1_graph))
        # # adj = adj + a
        # # adj_den = adj.todense()
        # # print("norm_adj:{}".format(adj_den[:3,:10]))
        # # if sparse:
        # #     sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        # # else:
        # #     # adj = (adj + sp.eye(adj.shape[0])).todense()   ##这里对邻接矩阵归一化之后为什么还要加上单位矩阵
        # #     adj = adj.todense()
        
        # # print("norm_adj2:{}".format(adj[:3,:10]))
        # features = torch.FloatTensor(features[np.newaxis])
        # if not sparse:
        #     adj = torch.FloatTensor(adj[np.newaxis])
        # # labels = torch.FloatTensor(labels[np.newaxis])
        # # idx_train = torch.LongTensor(idx_train)
        # # idx_val = torch.LongTensor(idx_val)
        # # idx_test = torch.LongTensor(idx_test)

        # model = DGI(ft_size, hid_units, nonlinearity)
        # optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

        # # if torch.cuda.is_available():
        # #     print('Using CUDA')
        # #     model.cuda()
        # #     features = features.cuda()
        # #     if sparse:
        # #         sp_adj = sp_adj.cuda()
        # #     else:
        # #         adj = adj.cuda()
        #     # labels = labels.cuda()
        #     # idx_train = idx_train.cuda()
        #     # idx_val = idx_val.cuda()
        #     # idx_test = idx_test.cuda()

        # b_xent = nn.BCEWithLogitsLoss()
        # xent = nn.CrossEntropyLoss()
        # cnt_wait = 0
        # best = 1e9
        # best_t = 0

        # for epoch in range(nb_epochs):
        #     model.train()
        #     optimiser.zero_grad()

        #     idx = np.random.permutation(nb_nodes)
        #     shuf_fts = features[:, idx, :]

        #     lbl_1 = torch.ones(batch_size, nb_nodes)
        #     lbl_2 = torch.zeros(batch_size, nb_nodes)
        #     lbl = torch.cat((lbl_1, lbl_2), 1)

        #     # if torch.cuda.is_available():
        #     #     shuf_fts = shuf_fts.cuda()
        #     #     lbl = lbl.cuda()
            
        #     logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) 

        #     loss = b_xent(logits, lbl)

        #     print('Loss:', loss)

        #     if loss < best:
        #         best = loss
        #         best_t = epoch
        #         cnt_wait = 0
        #         torch.save(model.state_dict(), 'best_dgi.pkl')
        #     else:
        #         cnt_wait += 1

        #     if cnt_wait == patience:
        #         print('Early stopping!')
        #         break

        #     loss.backward()
        #     optimiser.step()

        # print('Loading {}th epoch'.format(best_t))
        # model.load_state_dict(torch.load('best_dgi.pkl'))

        # embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

        # vg_features = embeds[0]
        # vg_features = F.normalize(vg_features, p=2, dim=1)

        # node2vec
        # top_dist_index = np.argsort(-sim)[:, :10]
        # # top_gst_index = np.argsort(-st_scores)[:, :5]
        # g = nx.Graph()
        # h = nx.Graph()

        # for i in range(len(top_dist_index)):
        #     k1_i_nonzero = np.nonzero(k1_graph[i])[0]
        #     for index,j in enumerate(k1_i_nonzero):
        #         g.add_edge(train_fnames[i], train_fnames[j], weight = k1_graph[i][j])
        
        # for node in train_fnames:
        #     g.add_edge(node, node, weight=1.)
        
        # for i in range(len(st_graph)):
        #     st_i_nonzero = np.nonzero(st_graph[i])
        #     for index,j in enumerate(st_i_nonzero[0]):
        #         h.add_edge(train_fnames[i], train_fnames[j], weight = st_scores[i][j])
        
        # for node in train_fnames:
        #     h.add_edge(node, node, weight=1.)

        # model1 = Node2Vec(g, walk_length=10, num_walks=10,
        #              p=0.5, q=4, workers=1, use_rejection_sampling=1)
        # model2 = Node2Vec(h, walk_length=10, num_walks=10,
        #              p=0.5, q=4, workers=1, use_rejection_sampling=1)

        # model1.train(window_size=10, iter=3)
        # model2.train(window_size=10, iter=3)
        # embeddings1 = model1.get_embeddings()
        # embeddings2 = model2.get_embeddings()
        
        # g_embeddings1 = []
        # g_embeddings2 = []
        # for i in train_fnames:
        #     g_embeddings1.append(embeddings1[i])
        #     # g_embeddings2.append(embeddings2[i])
        
        # g_embeddings1 = np.array(g_embeddings1)
        # g_embeddings1 = torch.FloatTensor(g_embeddings1)
        # g_embeddings1 = F.normalize(g_embeddings1, p=2, dim=1)

        # g_embeddings2 = np.array(g_embeddings2)
        # g_embeddings2 = torch.FloatTensor(g_embeddings2)
        # g_embeddings2 = F.normalize(g_embeddings2, p=2, dim=1)


        ## graphsage
        sum_scores = []
        gg = defaultdict(set)
        gg_positive = defaultdict(set)
        gg_negetive = defaultdict(set)
        adj_lists = defaultdict(set)

        # for i in range(len(k1_graph)):
        #     k1_i_nonzero = np.nonzero(k1_graph[i])[0]
        #     for index, j in enumerate(k1_i_nonzero):
        #         score = k1_graph[i][j]
        #         gg_positive[i].add(j)
        #         gg_positive[j].add(i)
        #         adj_lists[i].add(j)
        #         adj_lists[j].add(i)
        #         gg[i,j] = score
        #         gg[j,i] = score

        # for i in range(len(k1_graph)):
        #     top_indexs = np.argsort(-sim[i])[:100]
        #     for j in top_indexs:
        #         if st_scores[i][j] == -1 and sim[i][j] < 0.5:
        #             gg_negetive[i].add(j)
        #             gg_negetive[j].add(i)

        # for i in range(len(k1_graph)):
        #     k1_i_nonzero = np.nonzero(k1_graph[i])[0]
        #     st_i_nonzero = np.nonzero(st_graph[i])[0]
        #     u = set(k1_i_nonzero).union(set(st_i_nonzero))
        #     for index, j in enumerate(u):
        #         if st_scores[i][j] != -1 or sim[i][j] > 0.5:
        #             score = opt.ratio * k1_graph[i][j] + (1 - opt.ratio) * st_graph[i][j] 
        #             score = k1_graph[i][j]
        #             gg_positive[i].add(j)
        #             gg_positive[j].add(i)
        #             adj_lists[i].add(j)
        #             adj_lists[j].add(i)
        #             gg[i,j] = score
        #             gg[j,i] = score


        # for i in range(len(sim)):
        #     top_indexs = np.argsort(-sim[i])[:50]
        #     for index, j in enumerate(top_indexs):
        #         if st_scores[i][j] != -1:
        #             if index < 5:
        #                 gg_positive[i].add(j)
        #                 gg_positive[j].add(i)
        #             if index < 10 or (sim[i][j] > 0.8 and st_scores[i][j] > 0):
        #                 if i != j:
        #                     adj_lists[i].add(j)
        #                     adj_lists[j].add(i)
        #                     gg[i,j] = sim[i][j] 
        #                     gg[j,i] = sim[i][j] 
        #             else:
        #                 gg_negetive[i].add(j)
        #                 gg_negetive[j].add(i)
        #         if st_scores[i][j] == -1:
        #             gg_negetive[i].add(j)
        #             gg_negetive[j].add(i)
        #     adj_lists[i].add(i)
        #     gg[i,i] = 1

        for i in range(len(sim)):
            top_indexs = np.argsort(-sim[i])[:50]
            for index, j in enumerate(top_indexs):
                if st_scores[i][j] != -1:
                    if index < 5 or st_scores[i][j] > 0:
                        if i != j:
                            adj_lists[i].add(j)
                            adj_lists[j].add(i)
                            gg[i,j] = sim[i][j] 
                            gg[j,i] = sim[i][j]

            adj_lists[i].add(i)
            gg[i,i] = 1

        # config = pyhocon.ConfigFactory.parse_file(opt.config)
        # # load data
        # ds = opt.target
        # dataCenter = DataCenter(config)

        # # 构建gallery视觉图
        # device = torch.device("cuda", 0)
        # dataCenter.load_dataSet(ds, len_g=len(sim), isVision=True, use_sparse=opt.use_sparse)

        # gallery_features = torch.FloatTensor(self.features)
        # gallery_features = gallery_features.to(device)

        # graphSage1 = GraphSage(config['setting.num_layers'], gallery_features.size(1), config['setting.hidden_emb_size'], gallery_features, gg, adj_lists, device, gcn=True, agg_func=opt.agg_func)
        # graphSage1.to(device)

        # num_labels = 0
        # classification = Classification(config['setting.hidden_emb_size'], num_labels)
        # classification.to(device)

        # unsupervised_loss = UnsupervisedLoss(adj_lists, getattr(dataCenter, ds+'_train'), gg_positive, gg_negetive, device)


        # print('GraphSage with vision Net Unsupervised Learning')

        # for epoch in range(opt.g_epochs):
        #     print('----------------------EPOCH %d-----------------------' % epoch)
        #     graphSage1, classification = apply_model(dataCenter, ds, graphSage1, classification, unsupervised_loss, opt.b_sz, opt.unsup_loss, device, opt.learn_method)
        
        # vg_features = get_gnn_embeddings(graphSage1, len(adj_lists))
        # vg_features = vg_features.cpu()
        # vg_features = F.normalize(vg_features, p=2, dim=1)
        # # vg_dir = os.path.join(opt.logs_dir, 'train_g_features.pth')
        # # torch.save(vg_features, vg_dir)
        # vg_features = vg_features.numpy()
        # torch.cuda.empty_cache()

        

        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # st_graph = torch.Tensor(st_graph)
        # I = torch.eye(st_graph.shape[0])
        # S = torch.inverse(I - self.lamda * st_graph)
        # del I
        # st_sim = torch.mm(S, st_graph)
        # del S, st_graph

        # y = torch.ones_like(st_sim)
        # st_sim = torch.where(st_sim != 0, st_sim, y)
        # del y
        # # # context information
        # # # print(opt.ratio)
        # # # sim = self.heteo_sim(sim, st_sim, opt.ratio)
        # print("k1_sim:{}".format(sim[:3,:10]))
        # print("st_sim:{}".format(st_sim[:3,:10]))
        # sim =  opt.ratio * sim + (1 - opt.ratio) * st_sim
        sim = sim.numpy() 
        sim = self.get_gause_sim(sim, delta)

        # sim = []
        # # st_sim = []
        # for i in range(len(self.features)):
        #     query = vg_features[i].view(-1,1) 
        #     score = torch.mm(vg_features,query) 
        #     score = score.squeeze(1).cpu()
        #     score = score.numpy()
        #     sim.append(score)
        # sim = np.array(sim)
        
        # for i in range(len(self.features)):
        #     query = vg_features2[i].view(-1,1) 
        #     score = torch.mm(vg_features2,query) 
        #     score = score.squeeze(1).cpu()
        #     score = score.numpy()
        #     st_sim.append(score)

        
        # st_sim = np.array(st_sim)
        # sim =  opt.ratio * sim + (1 - opt.ratio) * st_sim
        # print("after graphsage")
        # print(sim[0:3, 0:10])

        # if self.general_graph:
        #     print("----general graph----")
        #     k1_graph = self.kneighbors(sim, kd)
        # else:
        #     k1_graph = self.heteo_kneighbors2(sim, sim2, ks, kd)
        # if self.homo_ap:
        #     print("----homo-ap----")
        #     k1_graph = self.row_normalization(k1_graph, exp=False)
        # else:
        #     k1_graph = self.heter_cam_normalization(k1_graph)
        
        # # propagation
        # k1_graph = torch.Tensor(k1_graph)
        # I = torch.eye(k1_graph.shape[0])
        # S = torch.inverse(I - self.lamda * k1_graph)
        # del I
        # sim = torch.mm(S, k1_graph)
        # del S, k1_graph

        # sim = sim.numpy()
        # print("after propagation")
        # print(sim[0:3, 0:10])
        # vg_features = vg_features.numpy()
        # sim = self.get_gause_sim(vg_features, delta)
        # print("after gause sim")
        # print(sim[0:3, 0:10])
        # if epocht != 0:
        #     k2 = 14
        # print(k2)
        k2_graph = self.kneighbors2(sim, k2)
        graph_target = self.split_as_camera(k2_graph)  #将图像按从camera 0开始进行分类，最前面的属于camera 0
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target
    
    def old_tracklet_propagation(self, ks=2, kd=4, k2=11):
        print("gause kernel")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)

        # hyper parameter
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            cams = self.cams[:, np.newaxis]
            flag = (cams.T == cams)
            sim_same_camera = sim * flag
            real_labels = self.real_labels[:, np.newaxis]
            labels_flag = (real_labels.T == real_labels)
            print(labels_flag)
            label_same_sim = sim_same_camera*labels_flag
            flag = 1-flag
            sim_diff_camera = sim * flag
            k_diff_sim = self.kneighbors(sim_diff_camera, kd)
            k1_graph = k_diff_sim + label_same_sim

        if self.homo_ap:
            print("----homo-ap----")
            k1_graph = self.row_normalization(k1_graph, exp=False)
        else:
            k1_graph = self.heter_cam_normalization(k1_graph)
            
        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # context information
        sim = sim.numpy()
        sim = self.get_gause_sim(sim, delta)
        k2_graph = self.kneighbors(sim, k2)
        graph_target = self.split_as_camera(k2_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target


    def only_graph(self, ks=2, kd=4, k2=11):
        print("gause kernel")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            k1_graph = self.heteo_kneighbors(sim, ks, kd)
        graph_target = self.split_as_camera(k1_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target

    def old_propagation(self, ks=2, kd=4, k2=11):
        print("gause kernel")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)

        # hyper parameter
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)
        else:
            k1_graph = self.heteo_kneighbors(sim, ks, kd)
        if self.homo_ap:
            k1_graph = self.row_normalization(k1_graph, exp=False)
        else:
            k1_graph = self.heter_cam_normalization(k1_graph)

        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # context information
        sim = sim.numpy()
        sim = self.get_gause_sim(sim, delta)
        k2_graph = self.kneighbors(sim, k2)
        graph_target = self.split_as_camera(k2_graph)  
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target


    def old_cos_propagation(self, ks=2, kd=4, k2=11):
        print("cosine sim")

        sim = self.get_cossim(self.features)


        # hyper parameter
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            k1_graph = self.heteo_kneighbors(sim, ks, kd)
        if self.homo_ap:
            k1_graph = self.row_normalization(k1_graph, exp=False)
        else:
            k1_graph = self.heter_cam_normalization(k1_graph)


        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # context information
        sim = sim.numpy()
        sim = self.get_cossim(sim)
        k2_graph = self.kneighbors(sim, k2)
        graph_target = self.split_as_camera(k2_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target

    def split_as_camera(self, graph_result):

        for c in range(np.max(self.cams)+1):
            if c == 0:
                result_reconstruct = np.squeeze(graph_result
                                                [:, np.where(self.cams == c)])
            else:
                result_reconstruct = np.concatenate(
                    (result_reconstruct,
                     np.squeeze(graph_result[:, np.where(self.cams == c)])
                     ),
                    axis=1
                )
        return result_reconstruct

    def heteo_sim(self, sim_matrix, st_matrix, ratio):
        cams = self.cams[:, np.newaxis]
        flag = (cams.T == cams)
        sim_same_camera = sim_matrix * flag
        flag = 1-flag
        sim_diff_camera = sim_matrix * flag
        st_diff_camera = st_matrix * flag
        diff = ratio * sim_diff_camera + (1 - ratio) * st_diff_camera
        return sim_same_camera + diff
    
    def heteo_vst_kneighbors(self, sim_matrix, st_matrix, ks, kd):
        cams = self.cams[:, np.newaxis]
        flag = (cams.T == cams)
        # inside use 0 to pad the empty
        sim_same_camera = sim_matrix * flag
        # st_same_camera = St * flag
        flag = 1-flag
        sim_diff_camera = sim_matrix * flag
        st_matrix1 = st_matrix * flag
        # print("st_matrix is equal:{}".format((st_matrix==st_matrix1).all()))
        # st_diff_camera = St * flag
        k_same_sim = self.kneighbors(sim_same_camera, ks)                   #1、在建立图的边的时候，可以单独以时空迁移概率建立边
        k_diff_sim = self.vst_kneighbors(sim_diff_camera, st_matrix1, kd)       #2、结合时空分数和视觉分数作为最终的相似性矩阵
        return k_diff_sim+k_same_sim

    def heteo_kneighbors(self, sim_matrix, ks, kd):
        cams = self.cams[:, np.newaxis]
        flag = (cams.T == cams)
        # inside use 0 to pad the empty
        sim_same_camera = sim_matrix * flag
        # st_same_camera = St * flag
        flag = 1-flag
        sim_diff_camera = sim_matrix * flag
        # st_diff_camera = St * flag
        k_same_sim = self.kneighbors2(sim_same_camera, ks)                   #1、在建立图的边的时候，可以单独以时空迁移概率建立边
        k_diff_sim = self.kneighbors2(sim_diff_camera, kd)       #2、结合时空分数和视觉分数作为最终的相似性矩阵
        return k_diff_sim+k_same_sim
    
    def heteo_kneighbors2(self, sim_matrix, sim_matrix2, ks, kd):
        cams = self.cams[:, np.newaxis]
        flag = (cams.T == cams)
        # inside use 0 to pad the empty
        sim_same_camera = sim_matrix * flag
        sim_same_camera2 = sim_matrix2 * flag
        # st_same_camera = St * flag
        flag = 1-flag
        sim_diff_camera = sim_matrix * flag
        sim_diff_camera2 = sim_matrix2 * flag
        # st_diff_camera = St * flag
        k_same_sim = self.kneighbors2(sim_same_camera, sim_same_camera2, ks)                   #1、在建立图的边的时候，可以单独以时空迁移概率建立边
        k_diff_sim = self.kneighbors2(sim_diff_camera, sim_diff_camera2, kd)       #2、结合时空分数和视觉分数作为最终的相似性矩阵
        return k_diff_sim+k_same_sim
    
    def heteo_stkneighbors(self, sim_matrix, st_matrix, ks, kd):
        cams = self.cams[:, np.newaxis]
        flag = (cams.T == cams)
        # inside use 0 to pad the empty
        sim_same_camera = sim_matrix * flag
        st_same_camera = st_matrix * flag
        flag = 1-flag
        sim_diff_camera = sim_matrix * flag
        st_diff_camera = st_matrix * flag
        k_same_sim = self.Stkneighbors(sim_same_camera, st_same_camera, ks)                   #1、在建立图的边的时候，可以单独以时空迁移概率建立边
        k_diff_sim = self.Stkneighbors(sim_diff_camera, st_diff_camera, kd)       #2、结合时空分数和视觉分数作为最终的相似性矩阵
        return k_diff_sim+k_same_sim

    def Stkneighbors(self, sim_matrix, st_matrix, knn, unifyLabel=None):
        k_sim = np.zeros_like(sim_matrix)
        argpart = np.argpartition(-sim_matrix, knn)  # big move before knn，将前knn个相似度大的数排在前面并获得相应下标
        # argpartSt = np.argpartition(-st_matrix, 4)
        # for i in range(argpart.shape[0]):
        #     diff = np.setdiff1d(argpartSt[i, 0:4],argpart[i, 0:knn])  #diff = argpartSt[i, 0:knn]
        #     k_sim[i, diff] = st_matrix[i, diff]                         # k_sim[i, diff] = alpha * st_matrix[i, diff] + (1 - alpha) * sim_matrix[i, diff]
        #     if i == 0:
        #         print(k_sim[0, diff])

        row_index = np.arange(sim_matrix.shape[0])[:, None]
        if unifyLabel:
            k_sim[row_index, argpart[:, 0:knn]] = unifyLabel
        else:
            k_sim[row_index, argpart[:, 0:knn]] = 0.65 * sim_matrix[row_index, argpart[:, 0:knn]] + 0.35 * st_matrix[row_index, argpart[:, 0:knn]]
        
        print(sim_matrix[0, argpart[0, 0:knn]], st_matrix[0, argpart[0, 0:knn]])
        return k_sim
    
    def vst_kneighbors(self, sim_matrix, st_matrix, knn, unifyLabel=None):
        k_sim = np.zeros_like(sim_matrix)
        # argpart = np.argpartition(-sim_matrix, knn)  # big move before knn，将前knn个相似度大的数排在前面并获得相应下标
        sort_indexs = np.argsort(-sim_matrix)[: , :100]
        # sort_st_indexs = np.argsort(-st_matrix)[: , :10]
        # row_index = np.arange(sim_matrix.shape[0])[:, None]
        # if unifyLabel:
        #     k_sim[row_index, argpart[:, 0:knn]] = unifyLabel
        # else:
        #     k_sim[row_index, argpart[:, 0:knn]] = sim_matrix[row_index, argpart[:, 0:knn]]
        
        # for i in range(len(sort_indexs)):
        #     for index, j in enumerate(sort_indexs[i]):
        #         if st_matrix[i][j] != -1:
        #             k_sim[i][j] = sim_matrix[i][j]
        
        for i in range(len(sort_indexs)):
            for index, j in enumerate(sort_indexs[i]):
                if st_matrix[i][j] !=-1:
                    # k_sim[i][j] = sim_matrix[i][j]
                    # if i < 3:
                    #     print(sim_matrix[i][j])
                    if index < knn:
                        k_sim[i][j] = sim_matrix[i][j]
                        k_sim[j][i] = sim_matrix[i][j]
                    elif sim_matrix[i][j] > 0.8 and st_matrix[i][j] > 0:
                        k_sim[i][j] = sim_matrix[i][j]
                        k_sim[j][i] = sim_matrix[i][j]
                        if i <= 5:
                            print(k_sim[i][j])
            
        # print("k2_diff_graph:{}".format(k_sim[0:3, 0:10]))    
        return k_sim

    def kneighbors(self, sim_matrix, knn, unifyLabel=None):
        k_sim = np.zeros_like(sim_matrix)
        argpart = np.argpartition(-sim_matrix, knn)  # big move before knn，将前knn个相似度大的数排在前面并获得相应下标

        row_index = np.arange(sim_matrix.shape[0])[:, None]
        if unifyLabel:
            k_sim[row_index, argpart[:, 0:knn]] = unifyLabel
        else:
            k_sim[row_index, argpart[:, 0:knn]] = sim_matrix[row_index, argpart[:, 0:knn]]
            k_sim[argpart[:, 0:knn], row_index] = sim_matrix[row_index, argpart[:, 0:knn]]
            
        print(k_sim[0:3, 0:10])    
        return k_sim
    
    def kneighbors2(self, sim_matrix, knn, unifyLabel=None):
        k_sim = np.zeros_like(sim_matrix)
        argpart = np.argpartition(-sim_matrix, knn)  # big move before knn，将前knn个相似度大的数排在前面并获得相应下标

        row_index = np.arange(sim_matrix.shape[0])[:, None]
        if unifyLabel:
            k_sim[row_index, argpart[:, 0:knn]] = unifyLabel
        else:
            k_sim[row_index, argpart[:, 0:knn]] = sim_matrix[row_index, argpart[:, 0:knn]]
            
        print(k_sim[0:3, 0:10])    
        return k_sim

    def get_gause_sim(self, features, delta=1.):
        distance = euclidean_distances(features, squared=True)
        distance /= 2 *delta**2
        sim = np.exp(-distance)
        return sim

    def get_cossim(self, features, temp=1.):
        return cosine_similarity(features)

    # def k_means(self, n_clusters):
    #     y_pred = KMeans(n_clusters, n_jobs=8).fit_predict(self.features)
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
    #
    # def ahc(self, n_clusters):
    #     dist = euclidean_distances(self.features)
    #     y_pred = AgglomerativeClustering(n_clusters, affinity="precomputed", linkage="average").fit_predict(dist)
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
    #
    # def sp(self, n_clusters):
    #     y_pred = SpectralClustering(n_clusters).fit_predict(self.features)
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
    #
    # def dbscan(self, epoch):
    #     rerank_dist = re_ranking(self.features)
    #     if epoch ==0:
    #         tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
    #         tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    #         tri_mat = np.sort(tri_mat, axis=None)
    #         top_num = np.round(1.6e-3 * tri_mat.size).astype(int)
    #         self.eps = tri_mat[:top_num].mean()
    #         print('eps in cluster: {:.3f}'.format(self.eps))
    #
    #     eps=self.eps
    #     y_pred = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8).fit_predict(rerank_dist)
    #
    #     max_label = np.max(y_pred)
    #     print(max_label)
    #     # return y_pred
    #     # take care that may has -1
    #     for index in np.where(y_pred == -1)[0]:
    #         y_pred[index] = max_label + 1
    #         max_label += 1
    #     print("new label: ", max_label, np.max(y_pred))
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #
    #     return y_pred
    #
    # def ap(self):
    #     dist = self.kneighbors(euclidean_distances(self.features), 50)
    #     y_pred = AffinityPropagation(preference=np.median(dist)).fit_predict(self.features)
    #
    #     max_label = np.max(y_pred)
    #     for index in np.where(y_pred == -1):
    #         y_pred[index] = max_label + 1
    #         max_label += 1
    #     print(max_label)
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
