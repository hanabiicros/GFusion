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
        
    def get_ordinary_sim(self, features):
        sim = []
        features = torch.FloatTensor(features)
        for i in range(len(features)):
            query = features[i].view(-1,1) 
            score = torch.mm(features,query) 
            score = score.squeeze(1).cpu()
            score = score.numpy()
            sim.append(score)
        sim = np.array(sim)
        return sim

    def old_delta_propagation(self, k2=14, opt=None, t_train=None):
        print("propagation")
        delta = 1.
        sim=[]
        if opt.wo_gaussian_sim:
            sim = self.get_ordinary_sim(self.features)
            print("wo guassian_sim")
        else:
            sim = self.get_gause_sim(self.features, delta)
    
        print(sim[0:3, 0:10])

        if opt.use_old_hgo:
        # hyper parameter
            if self.general_graph:
                print("----general graph----")
                k1_graph = self.kneighbors2(sim, 4)
            else:
                # k1_graph = self.heteo_vst_kneighbors(sim, st_scores, ks, kd)
                k1_graph = self.heteo_kneighbors(sim, 2, 4)
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


        ## graphsage
        gg = defaultdict(set)
        adj_lists = defaultdict(set)

        train_indexs =  np.argsort(-sim)[ : , :opt.kl]

        if not opt.wo_st_train:
            indexs = train_indexs[:,:opt.ks]
            st_scores = stmain(opt, indexs, t_train=t_train, train_indexs=train_indexs, flag = True)
            for i in range(len(sim)):
                for index, j in enumerate(train_indexs[i]):
                    if index < opt.ks or st_scores[i][index] > 0:
                        if i != j:
                            adj_lists[i].add(j)
                            adj_lists[j].add(i)
                            gg[i,j] = sim[i][j] 
                            gg[j,i] = sim[i][j]
                adj_lists[i].add(i)
                gg[i,i] = 1
        else:
            for i in range(len(sim)):
                for index, j in enumerate(train_indexs[i]):
                    if index < opt.ks:
                        if i != j:
                            adj_lists[i].add(j)
                            adj_lists[j].add(i)
                            gg[i,j] = sim[i][j] 
                            gg[j,i] = sim[i][j]
                adj_lists[i].add(i)
                gg[i,i] = 1

        config = pyhocon.ConfigFactory.parse_file(opt.config)
        # load data
        ds = opt.target
        dataCenter = DataCenter(config)

        device = torch.device("cuda", 0)
        dataCenter.load_dataSet(ds, len_g=len(sim), isVision=True, use_sparse=opt.use_sparse)

        gallery_features = torch.FloatTensor(self.features)
        gallery_features = gallery_features.to(device)

        graphSage1 = GraphSage(config['setting.num_layers'], gallery_features.size(1), config['setting.hidden_emb_size'], gallery_features, gg, adj_lists, device, gcn=True, agg_func=opt.agg_func)
        graphSage1.to(device)

        # num_labels = 0
        # classification = Classification(config['setting.hidden_emb_size'], num_labels)
        # classification.to(device)

        # unsupervised_loss = UnsupervisedLoss(adj_lists, getattr(dataCenter, ds+'_train'), gg_positive, gg_negetive, device)


        # print('GraphSage with vision Net Unsupervised Learning')

        # for epoch in range(opt.g_epochs):
        #     print('----------------------EPOCH %d-----------------------' % epoch)
        #     graphSage1, classification = apply_model(dataCenter, ds, graphSage1, classification, unsupervised_loss, opt.b_sz, opt.unsup_loss, device, opt.learn_method)
        
        vg_features = get_gnn_embeddings(graphSage1, len(adj_lists))
        vg_features = vg_features.cpu()
        vg_features = F.normalize(vg_features, p=2, dim=1)
        # vg_dir = os.path.join(opt.logs_dir, 'train_g_features.pth')
        # torch.save(vg_features, vg_dir)
        # vg_features = vg_features.numpy()
        # torch.cuda.empty_cache()
        
        if opt.wo_gaussian_sim:
            sim = self.get_ordinary_sim(vg_features)
            print("wo guassian_sim")
        else:
            sim = self.get_gause_sim(vg_features, delta)
        print("after gause sim")
        print(sim[0:3, 0:10])
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
