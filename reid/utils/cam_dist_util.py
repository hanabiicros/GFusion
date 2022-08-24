import numpy as np
import copy
import torch

def compute_euclidean_dist(A, B):
    rows = A.size(0)
    cols = B.size(0)
    A_pow = torch.pow(A, 2).sum(dim=1, keepdim=True).expand(rows, cols)
    B_pow = torch.pow(B, 2).sum(dim=1, keepdim=True).expand(cols, rows)
    O_sum = A_pow + B_pow.t()
    # dist = O_sum.addmm_(1, -2, A, B.t())
    dist = O_sum.addmm_(A, B.t(), beta=1, alpha=-2)
    return dist

def compute_dist_matrix(inputs):
    N = inputs.size(0)
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(N, N)
    dist = dist + dist.t()
    dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def compute_eps(dist, rho, num=None):
    tri_mat = np.triu(dist, 1)  # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    tri_mat = np.sort(tri_mat, axis=None)
    if num is None:
        top_num = np.round(rho * tri_mat.size).astype(int)
    else:
        top_num = np.round(num).astype(int)
    eps = tri_mat[:top_num].mean()
    return eps

def compute_eps_cam(dist, cam2idxs, rho, gamma = 0.5, num=None):
    eps_list = []
    ## compute intra camera eps
    cp_dist = copy.deepcopy(dist)
    total_eps = compute_eps(dist, rho=0.0016)
    print('total eps : {}'.format(total_eps))
    for cam, idxs in cam2idxs.items():
        cur_dist = dist[idxs][:,idxs]
        cur_eps = compute_eps(cur_dist, rho)
        eps_list.append(cur_eps)

        ## set the intra camera dist as zero
        median = cp_dist[idxs]
        median[:,idxs] = 0
        cp_dist[idxs] = median

    ## compute inter camera eps
    eps_inter = compute_eps(cp_dist, rho, num)
    eps_intra_mean = np.mean(eps_list)
    eps_intra_min = np.min(eps_list)
    eps_intra_max = np.max(eps_list)
    eps_intra = (np.sum(eps_list) - eps_intra_min - eps_intra_max) / (len(eps_list) - 2)
    # eps_intra = eps_intra_mean
    print('eps_intra_cam : {}, eps_intra_avg : {}, eps_inter_cam : {}'.format(eps_list, eps_intra, eps_inter))
    eps = gamma * eps_intra + (1 - gamma) * eps_inter
    eps = [eps_intra, eps_inter]
    return eps

def compute_global_min_eps(dist, N=1):
    rows = dist.shape[0]
    res = np.zeros((rows, 1))
    max_dist = np.max(dist)
    dist[np.diag_indices_from(dist)] = max_dist + 1.
    dist.sort(axis=1)
    res[:, 0] = np.mean(dist[:, 0:N], axis=1)
    return res

def compute_intra_eps(dist, cam2idxs, intra_N=4):
    ## 输入：dist, cam2idxs， 
    ## 输出：按dist的顺序，每个图片的同摄像头最近邻，跨摄像头最近邻
    N = dist.shape[0]
    res = np.zeros((N, 1))
    for c in sorted(cam2idxs.keys()):
        idxs = cam2idxs[c]
        sub_dist = dist[idxs][:, idxs] ## 同摄像头距离
        cur_max = np.max(sub_dist)
        sub_dist[np.diag_indices_from(sub_dist)] = cur_max + 1. ## 将自己跟自己的距离最大化
        sub_dist.sort(axis=1) ## 每一行从小到大排序
        # intra_neighbor = np.max(sub_dist[:, 0 : intra_N], axis=1) 
        intra_neighbor = np.mean(sub_dist[:, 0 : intra_N], axis=1) 
        res[idxs, 0] = intra_neighbor ## 同摄像头间距离
    return res

def compute_inter_eps(dist, cam2idxs, inter_N=1):
    N = dist.shape[0]
    res = np.zeros((N, 1))
    total_idxs = set([x for x in range(N)])
    for c in sorted(cam2idxs.keys()):
        intra_idxs = cam2idxs[c] #该摄像机内部的图像的索引
        inter_idxs = list(total_idxs - set(intra_idxs))
        inter_idxs = np.array(inter_idxs, dtype='int32')
        sub_dist = dist[intra_idxs][:, inter_idxs]
        sub_dist.sort(axis=1)
        inter_neighbor = np.mean(sub_dist[:, 0 : inter_N], axis=1)
        res[intra_idxs, 0] = inter_neighbor
    return res

def compute_max_inter_eps(dist, cam2idxs, inter_N=1):
    ## 计算最大的跨摄像头对的距离
    ## 输入： dist, 所有图片对的距离
    ## 输出： eps, 每张图片的eps
    N = dist.shape[0]
    res = np.zeros((N, len(cam2idxs.keys())))
    for c_1 in sorted(cam2idxs.keys()):
        for c_2 in sorted(cam2idxs.keys()):
            if c_1 == c_2: continue
            c1_idx = cam2idxs[c_1]
            c2_idx = cam2idxs[c_2]
            sub_dist = dist[c1_idx][:, c2_idx]
            sub_dist.sort(axis=1)
            cur_inter_neighbor = np.mean(sub_dist[:, 0:inter_N], axis=1)
            res[c1_idx][c_2] = cur_inter_neighbor
    res = np.mean(res, axis=1)
    return res

def compute_intra_inter_dist(dist, cam2idxs, inter_N=1, intra_N=1):
    ## 归一化
    ## 计算每个摄像头跟另一个摄像头的最近邻的距离
    N = dist.shape[0]
    num_cams = len(cam2idxs.keys())
    res = [[0.0 for _ in range(num_cams)] for _ in range(num_cams)]
    for c_1 in sorted(cam2idxs.keys()):
        for c_2 in sorted(cam2idxs.keys()):
            if c_1 == c_2:
                ## 同摄像头距离
                idxs = cam2idxs[c_1]
                sub_dist = dist[idxs][:, idxs] ## 同摄像头距离
                cur_max = np.max(sub_dist)
                sub_dist[np.diag_indices_from(sub_dist)] = cur_max + 1. ## 将自己跟自己的距离最大化
                sub_dist.sort(axis=1) ## 每一行从小到大排序
                # intra_neighbor = np.max(sub_dist[:, 0 : intra_N], axis=1) 
                intra_neighbor = np.mean(sub_dist[:, 0 : intra_N], axis=1) 
                res[c_1][c_1] = np.mean(intra_neighbor)
            else:
                ## 跨摄像头距离
                c1_idx = cam2idxs[c_1]
                c2_idx = cam2idxs[c_2]
                sub_dist = dist[c1_idx][:, c2_idx]
                sub_dist.sort(axis=1)
                cur_inter_neighbor = np.mean(sub_dist[:, 0:inter_N], axis=1)
                res[c_1][c_2] = np.mean(cur_inter_neighbor)
                res[c_2][c_1] = res[c_1][c_2]
    res = np.array(res)
    return res
    
def normalize_dist(dist, cam2idxs, normalize_inter=False):
    inter_intra_dist = compute_intra_inter_dist(dist, cam2idxs)
    total_cams = len(cam2idxs.keys()) 
    N = dist.shape[0]
    total_idxs = {i for i in range(N)}
    if normalize_inter:
        for c in sorted(cam2idxs.keys()):
            ## 获取当前摄像头内图片的跨摄像头最近邻的均值
            cur_inter_intra_means = np.array(inter_intra_dist[c])
            cross_inter_means = (np.sum(cur_inter_intra_means) - inter_intra_dist[c][c]) / (total_cams - 1)
            idxs = cam2idxs[c]
            cross_idxs = list(total_idxs - set(idxs))
            sub_dist = dist[idxs][:, cross_idxs]
            sub_dist /= cross_inter_means
            num_rows = len(idxs)
            num_cols = len(cross_idxs)
            rows = list(idxs) * num_cols
            rows = np.array(rows).reshape(num_cols, num_rows).transpose().flatten()
            cols = np.array(list(cross_idxs) * num_rows) ## 列 ： 跨摄像头的图片下标
            indexs = (rows, cols) 
            sub_dist = sub_dist.flatten() 
            dist[indexs] = sub_dist           
    else:
        for c in sorted(cam2idxs.keys()):
            idxs = cam2idxs[c]
            sub_dist = dist[idxs][:, idxs]
            sub_dist /= inter_intra_dist[c][c]
            n = len(idxs)
            cols = np.array(list(idxs) * n)
            rows = cols.reshape(n, n).transpose().flatten()
            indexs = (rows, cols) 
            sub_dist = sub_dist.flatten()          
            dist[indexs] = sub_dist
    return dist


def normalize_all_dist(dist, cam2idxs):
    inter_intra_dist = compute_intra_inter_dist(dist, cam2idxs)
    total_cams = len(cam2idxs.keys()) 
    N = dist.shape[0]
    total_idxs = {i for i in range(N)}
    for c in sorted(cam2idxs.keys()):
        ## 获取当前摄像头内图片的跨摄像头最近邻的均值
        cur_inter_intra_means = np.array(inter_intra_dist[c])
        cross_inter_means = (np.sum(cur_inter_intra_means) - inter_intra_dist[c][c]) / (total_cams - 1)
        idxs = cam2idxs[c]
        cross_idxs = list(total_idxs - set(idxs))
        sub_dist = dist[idxs][:, cross_idxs]
        sub_dist /= cross_inter_means
        num_rows = len(idxs)
        num_cols = len(cross_idxs)
        rows = list(idxs) * num_cols
        rows = np.array(rows).reshape(num_cols, num_rows).transpose().flatten()
        cols = np.array(list(cross_idxs) * num_rows) ## 列 ： 跨摄像头的图片下标
        indexs = (rows, cols) 
        sub_dist = sub_dist.flatten() 
        dist[indexs] = sub_dist           

        idxs = cam2idxs[c]
        sub_dist = dist[idxs][:, idxs]
        sub_dist /= inter_intra_dist[c][c]
        n = len(idxs)
        cols = np.array(list(idxs) * n)
        rows = cols.reshape(n, n).transpose().flatten()
        indexs = (rows, cols) 
        sub_dist = sub_dist.flatten()          
        dist[indexs] = sub_dist
    return dist

def compute_intra_dist_means_invariance(cam2idxs, feats=None, dist=None, withCross=True):
    ## 同摄像头的均值与方差 ，输出：每个摄像头的摄像头内图片距离均值，图片距离方差，跨摄像头图片距离均值，跨摄像头图片距离方差
    ## withCross : 是否输出跨摄像头距离均值和方差，若是，则每一行的值为： 摄像头内图片均值，方差， 跨摄像头图片均值， 方差
    assert(feats is not None or dist is not None)
    if dist is None:
        dist = compute_dist_matrix(feats).data.cpu().numpy()
    N = dist.shape[0]
    cols = 4 if withCross else 2
    res = np.zeros((len(cam2idxs.keys()), cols))
    total_idxs = set([x for x in range(N)])
    
    for cam in sorted(cam2idxs.keys()):
        intra_idxs = cam2idxs[cam]
        ## 同摄像头距离
        sub_dist = dist[intra_idxs][:, intra_idxs] 
        sub_dist = np.triu(sub_dist, 1)
        tmp_mask = np.triu(np.ones_like(sub_dist), 1)
        intra_size = np.nonzero(tmp_mask)[0].shape[0]
        means = np.sum(sub_dist) / intra_size
        res[cam][0] = means
        sub_dist = np.power(sub_dist - means, 2)
        invariance = np.sum(sub_dist) / intra_size
        res[cam][1] = invariance
        if withCross:
            assert res.shape[1] == 4
            ## 跨摄像头距离
            inter_idxs = list(total_idxs - set(intra_idxs))
            inter_sub_dist = dist[intra_idxs][:, inter_idxs]
            inter_means = np.mean(inter_sub_dist)
            inter_sub_dist = np.power(inter_sub_dist - inter_means, 2)
            inter_invariance = np.mean(inter_sub_dist)
            res[cam][2] = inter_means
            res[cam][3] = inter_invariance
    return res

