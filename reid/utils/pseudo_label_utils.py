from sklearn.cluster import DBSCAN
from .cam_dist_util import compute_euclidean_dist
from collections import defaultdict
import torch
import numpy as np
import torch.nn.functional as F
import math

UNCLASSIFIED = False
NOISE = None

def _dist(p,q):
	return math.sqrt(np.power(p-q,2).sum())

def _eps_neighborhood(p,q,eps):
	return _dist(p,q) < eps

def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m[:,point_id], m[:,i], eps):
            seeds.append(i)
    return seeds

def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def dbscan(m, eps, min_points):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:,point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications



def pseudo_label_estimation(dist,eps, Min_Pts=4):
    print('clustering and labeling , eps : {}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=Min_Pts, metric='precomputed', n_jobs=-1)
    labels = cluster.fit_predict(dist)
    num_ids = len(set(labels)) - 1
    print('number of clusters : {}'.format(num_ids))
    return labels

def get_pseudo_label_dataset(pseudo_labels, trainset):
    ## 返回从第k个聚簇开始，每个聚簇只有一张图片
    pseudo_label_wo_noise_dataset = []
    pseudo2idxs = defaultdict(list) #defaultdict():当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值(list:[])
    num_clusters = len(set(pseudo_labels)) - 1 ## id 从 -1 开始， -1的都是噪声标签，所以总的聚簇个数需要减1
    ## 0 - （num_id-1）
    for idx, (p_label, info) in enumerate(zip(pseudo_labels, trainset)):
        ## info : img_path, pid, camid
        if p_label == -1: continue
        else:
            pseudo_label_wo_noise_dataset.append((info[0], int(p_label), info[2], idx)) #去除噪声数据集
            pseudo2idxs[p_label].append(idx) #该字典存储了每个聚簇中包含的图片的索引
    return pseudo_label_wo_noise_dataset, pseudo2idxs

def compute_silhouette_scores(dataset, pseudo2idxs=None, dist=None, tgt_features=None, cuda_device=None):
    '''
    param pseudo_label_wo_noise_dataset : dataset without noise
    dist : dist of each imgs' pair, should not in cuda device
    tgt_features : feats of each image
    '''
    
    if dist is None and tgt_features is None:
        raise RuntimeError("dist and tgt_features should not be None at the same time")
    if dist is None:
        dist = compute_euclidean_dist(tgt_features, tgt_features).clamp(min=1e-12).sqrt()

    if pseudo2idxs is None:
        pseudo2idxs = defaultdict(list)
        for _, pid, _, idx in dataset:
            pseudo2idxs[pid].append(idx)
    num_clusters = len(pseudo2idxs.keys())
    num_images = len(dataset)
    mask = torch.zeros((num_images, num_clusters), dtype=torch.float)
    if cuda_device is not None:
        mask = mask.cuda(device=cuda_device)
    num_images = 0
    pseudo_labels = sorted(pseudo2idxs.keys())
    all_idxs = []
    for pseudo_label in pseudo_labels:
        idxs = pseudo2idxs[pseudo_label]
        cur_cluster_images = len(idxs)
        mask[num_images : num_images + cur_cluster_images, pseudo_label] = 1.
        num_images += cur_cluster_images
        all_idxs = all_idxs + idxs
    members_each_cluster = mask.sum(dim=0) #将mask每一列的数据相加
    all_idxs = torch.tensor(all_idxs, dtype=torch.long)    
    sub_dist = dist[all_idxs][:, all_idxs]
    print("sub_dist[0]:{}".format(sub_dist[0]))
    Scores = torch.zeros(num_clusters)
    if cuda_device is not None:
        Scores = Scores.cuda(cuda_device)
    cur_idx = 0
    for pseudo_label in pseudo_labels:
        idxs = pseudo2idxs[pseudo_label]
        silhout_scores = torch.zeros(len(idxs))
        if cuda_device is not None:
            silhout_scores = silhout_scores.cuda(cuda_device)
        for i in range(len(idxs)):
            idx_in_cluster = cur_idx + i
            self_dist = sub_dist[idx_in_cluster, idx_in_cluster]
            i_dis_sim = sub_dist[idx_in_cluster].unsqueeze(0).mm(mask)[0]
            a_i = (i_dis_sim[pseudo_label] - self_dist) / (members_each_cluster[pseudo_label] - 1)
            i_inter_dis_sim_vector = i_dis_sim / members_each_cluster
            i_inter_dis_sim_vector[pseudo_label] = 1000
            b_i = torch.min(i_inter_dis_sim_vector)    
            if a_i < b_i:
                s_i = 1. - a_i / b_i
            elif a_i > b_i:
                s_i = b_i / a_i - 1.
            else:
                s_i = 0
            silhout_scores[i] = s_i
        
        cur_idx += len(idxs)
        Scores[pseudo_label] = torch.mean(silhout_scores)
    return Scores

def cluster_refine_intra(sil_scores, ratio, pseudo2idxs, dist, intra_eps):
    # num = max(int(len(sil_scores) * ratio), 1)
    # print('starting to refine the top {} sil_scores ids, ratio : {}'.format(num, ratio))
    
    for (pseudo_label, score) in sil_scores:
        if score >= ratio:
            break
        # pseudo_label = sil_scores[i][0] ## (k, v), k : label; v : score
        cur_num_labels = len(pseudo2idxs.keys())
        cur_idxs = pseudo2idxs[pseudo_label]
        cur_idxs = np.array(cur_idxs, dtype='int32')
        cur_dist = dist[cur_idxs][:, cur_idxs]
        y_pred = DBSCAN(eps=intra_eps, min_samples=4, metric='precomputed', n_jobs=-1).fit_predict(cur_dist)
        noise_num = np.where(y_pred == -1)[0].size
        if noise_num == y_pred.size:
            continue
        else:
            pseudo2idxs[pseudo_label] = []
            if -1 in y_pred:
                for idx, label in zip(cur_idxs, y_pred):
                    if label == -1: continue
                    elif label == 0: pseudo2idxs[pseudo_label].append(idx)
                    else:
                        new_label = cur_num_labels + label - 1
                        pseudo2idxs[new_label].append(idx)
            else:
                for idx, label in zip(cur_idxs, y_pred):
                    if label == -1: continue
                    elif label == 0: pseudo2idxs[pseudo_label].append(idx)
                    else:
                        new_label = cur_num_labels + label - 1
                        pseudo2idxs[new_label].append(idx)
    return pseudo2idxs

def compute_centroid(pseudo2idxs, feats):
    res = []
    for pl in sorted(pseudo2idxs.keys()):
        idxs = pseudo2idxs[pl]
        sub_feats = feats[idxs]
        res.append(torch.mean(sub_feats, dim=0).unsqueeze(0))    
    res = torch.cat(res, 0)
    res = F.normalize(res, p=2, dim=1)
    return res

def get_trainset(pseudo2idxs, unlabeled_data, with_noise=False, feats=None):
    if with_noise:
        assert feats is not None
    total_idxs = [i for i in range(len(unlabeled_data))]
    noise_idxs = set(total_idxs)
    trainset = []
    num_ids = len(pseudo2idxs.keys())
    for p_label, idxs in pseudo2idxs.items():
        noise_idxs = noise_idxs - set(idxs)
        for idx in idxs:
            info = unlabeled_data[idx]
            trainset.append((info[0], p_label, info[2], idx))
    if with_noise:
        centroids = compute_centroid(pseudo2idxs, feats)
        noise_idxs = list(noise_idxs)
        noise_feats = feats[noise_idxs]
        dist_matrix = compute_euclidean_dist(noise_feats, centroids)
        dist_matrix, sorted_idxs = torch.sort(dist_matrix, dim=1)
        for i, idx in enumerate(noise_idxs):
            cur_pseudo_label = sorted_idxs[i][0].item()
            info = unlabeled_data[idx]
            trainset.append((info[0], cur_pseudo_label, info[2], idx))
            pseudo2idxs[cur_pseudo_label].append(idx)
            # num_ids += 1
    print('{} ids and {} imgs for training ...'.format(num_ids, len(trainset)))
    return trainset, pseudo2idxs
        