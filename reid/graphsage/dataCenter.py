import sys
import os

from collections import defaultdict
import numpy as np
import torch
import scipy.sparse as sp


def normalize_adj_coo(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adj_dense(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float16)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def heter_cam_normalization(k1_graph, cams):
        for i in range(len(k1_graph)):
            index = np.where(k1_graph[i] != 0.)
            weights = k1_graph[i][index]
            cd_c = cams[index]
            tag_c_set = set(cd_c)
            for c in tag_c_set:
                c_index = np.where(cd_c == c)
                w = weights[c_index]

                w = len(w) / len(cd_c) * w / np.sum(w)  
                k1_graph[i][index[0][c_index]] = w
        print(np.sum(k1_graph, axis=1))
        print('heter_cam_normalization')
        return k1_graph

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config):
		super(DataCenter, self).__init__()
		self.config = config
		
	def load_dataSet(self, dataSet='cora', top_indexs=None, st_sort_indexs=None,len_g=None, isVision=True, use_sparse=False):
		if dataSet == 'cora':
			cora_content_file = self.config['file_path.cora_content']
			cora_cite_file = self.config['file_path.cora_cite']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			label_map = {} # map label to Label_ID
			with open(cora_content_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:-1]])
					node_map[info[0]] = i
					if not info[-1] in label_map:
						label_map[info[-1]] = len(label_map)
					labels.append(label_map[info[-1]])
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(cora_cite_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 2
					paper1 = node_map[info[0]]
					paper2 = node_map[info[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)

			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)

		elif dataSet == 'pubmed':
			pubmed_content_file = self.config['file_path.pubmed_paper']
			pubmed_cite_file = self.config['file_path.pubmed_cites']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			with open(pubmed_content_file) as fp:
				fp.readline()
				feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
				for i, line in enumerate(fp):
					info = line.split("\t")
					node_map[info[0]] = i
					labels.append(int(info[1].split("=")[1])-1)
					tmp_list = np.zeros(len(feat_map)-2)
					for word_info in info[2:-1]:
						word_info = word_info.split("=")
						tmp_list[feat_map[word_info[0]]] = float(word_info[1])
					feat_data.append(tmp_list)
			
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(pubmed_cite_file) as fp:
				fp.readline()
				fp.readline()
				for line in fp:
					info = line.strip().split("\t")
					paper1 = node_map[info[1].split(":")[1]]
					paper2 = node_map[info[-1].split(":")[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)
			
			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)
		else:
			# cora_content_file = self.config['file_path.cora_content']
			# cora_cite_file = self.config['file_path.cora_cite']

			# feat_data = []
			labels = [] # label sequence of node
			# node_map = {} # map node to Node_ID
			# label_map = {} # map label to Label_ID
			# with open(cora_content_file) as fp:
			# 	for i,line in enumerate(fp):
			# 		info = line.strip().split()
			# 		feat_data.append([float(x) for x in info[1:-1]])
			# 		node_map[info[0]] = i
					# if not info[-1] in label_map:
					# 	label_map[info[-1]] = len(label_map)
					# labels.append(label_map[info[-1]])
			# feat_data = gallery_features
			# labels = np.asarray(labels, dtype=np.int64)

			# g = torch.eye(gscores.size(0))
			# gscores = torch.from_numpy(gscores)

			adj_lists = defaultdict(set)


				# if use_sparse:
				# # g = np.eye(len(gscores), dtype=np.float16)
				# 	g = sp.eye(len_g, format='lil')
				# 	for i in range(len(top_indexs)):
				# 		indexs = set(top_indexs[i][:15]).union(set(st_sort_indexs[i][:5]))
				# 		for j in indexs:
				# 			if gscores[i][j] != 0 and gscores[i][j] != -1 and gscores[i][j] != -2:
				# 				adj_lists[i].add(j)
				# 				adj_lists[j].add(i)
				# 				if i != j:
				# 					g[i,j] = gscores[i][j]
				# 					g[j,i] = gscores[j][i]
				# 	g = normalize_adj_coo(g)
				# 	g = sparse_mx_to_torch_sparse_tensor(g)
				# else:
					# g = np.eye(len_g, dtype=np.float16)
					# for i in range(len(top_indexs)):
					# 	# indexs = set(top_indexs[i][:15]).union(set([st_sort_indexs[i][0]]))
					# 	indexs = set(top_indexs[i][:15])
					# 	for j in indexs:
					# 		if gscores[i][j] != 0 and gscores[i][j] != -1 and gscores[i][j] != -2:
					# 			adj_lists[i].add(j)
					# 			adj_lists[j].add(i)
					# 			if i != j:
					# 				g[i][j] = gscores[i][j]
					# 				g[j][i] = gscores[j][i]
					
					# # g = heter_cam_normalization(g, cams)
					# g = normalize_adj_dense(g)
					# g = torch.from_numpy(g)

			# sum_scores = []
			# g = defaultdict()
			# for i in range(len(top_indexs)):
			# 	# indexs = set(top_indexs[i][:15]).union(set([st_sort_indexs[i][0]]))
			# 	indexs = set(top_indexs[i][:15])
			# 	for j in indexs:
			# 		if gscores[i][j] != 0 and gscores[i][j] != -1 and gscores[i][j] != -2:
			# 			if i != j:
			# 				adj_lists[i].add(j)
			# 				adj_lists[j].add(i)
			# 				g[i,j] = gscores[i][j]
			# 				g[j,i] = gscores[j][i]
			# 	adj_lists[i].add(i)
			# 	g[i,i] = 1
			
			# for i in range(len(top_indexs)):
			# 	indexs = adj_lists[i]
			# 	if i <= 10:
			# 		print("len_indexs:{}".format(len(indexs)))
			# 	sum_score = 0
			# 	for j in indexs:
			# 		sum_score += g[i,j]
			# 	sum_scores.append(sum_score)
			
			# for i in range(len(top_indexs)):
			# 	indexs = adj_lists[i]
			# 	for j in indexs:
			# 		g[i,j] /= sum_scores[i]

			

		
			
			# g = torch.from_numpy(g)
			# num_neigh = g.sum(1, keepdim=True)
			# g = g.div(num_neigh)

			# b = np.sum(g, axis = 1)
			# for i,_ in enumerate(g):
			# 	g[i] /= b[i]

			# print(np.sum(g, axis=1))
			# with open(cora_cite_file) as fp:
			# 	for i,line in enumerate(fp):
			# 		info = line.strip().split()
			# 		assert len(info) == 2
			# 		paper1 = node_map[info[0]]
			# 		paper2 = node_map[info[1]]
			# 		adj_lists[paper1].add(paper2)
			# 		adj_lists[paper2].add(paper1)

			# assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(len_g)

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			# setattr(self, dataSet+'_feats', feat_data)
			# setattr(self, dataSet+'_labels', labels)
			# setattr(self, dataSet+'_adj_lists', adj_lists)
			# setattr(self, dataSet+'_weight_lists', g)


	def _split_data(self, num_nodes, test_split = 8, val_split = 16):
		rand_indices = np.random.permutation(num_nodes)

		test_size = 7 * (num_nodes // test_split)
		val_size = num_nodes // val_split
		# test_size = 0
		# val_size = 0
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]
		
		return test_indexs, val_indexs, train_indexs


