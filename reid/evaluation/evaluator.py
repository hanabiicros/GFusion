from __future__ import print_function, absolute_import
import time
from collections import OrderedDict, defaultdict
import torch
import numpy as np
import os.path as osp
import os
import scipy.io as sci
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

from reid.evaluation.ranking import cmc, mean_ap, mean_ap2, map_cmc
from reid.evaluation.meters import AverageMeter
from reid.lib.serialization import to_torch, mkdir_if_missing


def extract_cnn_feature(backbone, inputs, layer=None):
    inputs = to_torch(inputs).cuda()
    with torch.no_grad():
        outputs = backbone(inputs)
        outputs = outputs.data.cpu()
    # if isinstance(layer, int):
    #     outputs = outputs[layer].data.cpu()
    # else:
    #     outputs = outputs[-1].data.cpu()  # norm feature
    return outputs


def extract_features(model, data_loader):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fpaths, pids, cams, indices) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fpath, output, pid in zip(fpaths, outputs, pids):
            features[fpath] = output
            labels[fpath] = pid

        batch_time.update(time.time() - end)
        end = time.time()

    return features, labels

def get_gause_sim(q_f, g_f, delta=1.):
    distance = euclidean_distances(q_f, g_f, squared=True)
    distance /= 2 *delta**2
    dist = np.exp(distance)
    return dist

def get_cossim(q_f, g_f):
    dist = -cosine_similarity(q_f, g_f)
    return dist

def get_euclidean(q_f, g_f):
    m, n = q_f.size(0), g_f.size(0)
    dist = torch.pow(q_f, 2)
    dist = dist.sum(dim=1, keepdim=True)
    dist = dist.expand(m, n)

    dist = torch.pow(q_f, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(g_f, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, q_f, g_f.t())
    # We use clamp to keep numerical stability
    dist = torch.clamp(dist, 1e-8, np.inf)
    return dist



def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    # fpath, pid, camera
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    # print("euclidean")
    # dist = get_euclidean(x, y)



    # print("get_cossim")
    # dist = get_cossim(x.numpy(), y.numpy())

    print("guase")
    dist = get_gause_sim(x.numpy(), y.numpy(), delta=1.)
    print(dist[0, :5])

    return dist, x.numpy(), y.numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), kl=50, is_train=True):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _,_,_ in query]
        gallery_ids = [pid for _, pid, _,_,_ in gallery]
        query_cams = [cam for _, _, cam,_,_ in query]
        gallery_cams = [cam for _, _, cam,_,_ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    query_features = torch.FloatTensor(query_features)
    gallery_features = torch.FloatTensor(gallery_features)
    # CMC = torch.IntTensor(len(gallery_ids)).zero_()
    # ap = 0.0
    scores, indexs = [], []
    gscores, gindexs = [], []

    if not is_train:
        # for i in range(len(gallery_ids)):
            # distance = euclidean_distances(gallery_features[i], gallery_features, squared=True)
        #     distance /= 2
        #     score = np.exp(-distance,dtype=np.float16)
        #     gindex = np.argsort(-score[0])[:kl]
        #     gscores.append(score[0][gindex])

        # for i in range(len(query_ids)):
        #     distance = euclidean_distances(query_features[i], gallery_features, squared=True)
        #     distance /= 2
        #     score = np.exp(-distance,dtype=np.float16)
        #     index = np.argsort(-score[0])[:kl]
        #     scores.append(score[0][index])

        for i in range(len(gallery_ids)):
            query = gallery_features[i].view(-1,1) 
            gscore = torch.mm(gallery_features,query) 
            gscore = gscore.squeeze(1).cpu()
            gscore = gscore.numpy()
            # predict index
            index = np.argsort(-gscore)[:kl]  #from small to large
            # index = index[::-1]
            gscore = gscore[index]
            gindexs.append(index)
            gscores.append(gscore)

        for i in range(len(query_ids)):
            query = query_features[i].view(-1,1) 
            score = torch.mm(gallery_features,query) 
            score = score.squeeze(1).cpu()
            score = score.numpy()
            # predict index
            index = np.argsort(-score)[:kl] 
            # index = index[::-1]
            score = score[index]
            indexs.append(index)
            scores.append(score)

        indexs = np.array(indexs, dtype=np.int32)
        gindexs = np.array(gindexs, dtype=np.int32)
        scores = np.array(scores, dtype=np.float16)
        gscores = np.array(gscores, dtype=np.float16)

    # scores = sp.coo_matrix(scores, dtype=np.float16)
    # gscores = sp.coo_matrix(gscores, dtype=np.float16)

    # if eval_path is not None:
    #         mkdir_if_missing(eval_path)
    # score_path = os.path.join(eval_path, 'score.txt')
    # pid_path = os.path.join(eval_path, 'pid.txt')
    # np.savetxt(score_path, scores, fmt='%.4f')
    # np.savetxt(pid_path, indexs, fmt='%d')

    # Evaluation
    mAP, all_cmc = map_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, all_cmc[k - 1]))
    return mAP, all_cmc[0], scores, gscores, indexs, gindexs, query_features, gallery_features


def save_feature(features, fpaths, labels, camids, save_name):
    result = {'features': np.squeeze(features),
              'fpaths': np.squeeze(fpaths),
              'labels': np.squeeze(labels),
              'camids': np.squeeze(camids)
              }
    save_dir = '/hdd/sdb/lsc/pytorch/PGM/logs/features_visualization'
    mkdir_if_missing(save_dir)
    sci.savemat(osp.join(save_dir, save_name+'_result.mat'),
                result)

def sort_by_score(qf, gf, ks):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(-score)[:ks]  #from small to large
    # index = index[::-1]
    score = score[index]
    return index, score

def get_score_pid_on_training(features, train=None, ks=4, logs_dir=""):

    x = torch.cat([features[f].unsqueeze(0) for f, _, _,_ ,_ in train], 0)
    m = x.size(0)
    x = x.view(m, -1)
    train_features = x.numpy()

    train_features = torch.FloatTensor(train_features)

    train_ids = [pid for _, pid, _ ,_ ,_ in train]
    indexs = []
    for i in range(len(train_ids)):
        index, score = sort_by_score(train_features[i], train_features, ks)
        indexs.append(index)

    indexs = np.array(indexs, dtype=np.int16)
    logs_dir = logs_dir + "/indexs.txt"
    np.savetxt(logs_dir, indexs, fmt='%d')
    #transfer_name = opt.name + '_' + target + '-train'

    return indexs

def result_eval(predict_path, score_path ,query=None, gallery=None):
    res = np.genfromtxt(predict_path, delimiter=' ',dtype = np.int32)
    score = np.genfromtxt(score_path, delimiter=' ',dtype = np.float32)
    print('predict info get, extract gallery info start')

    query_ids = np.asarray([pid for _, pid, _,_,_ in query])
    gallery_ids = np.asarray([pid for _, pid, _ ,_,_ in gallery])
    query_cams = np.asarray([cam for _, _, cam,_,_ in query])
    gallery_cams = np.asarray([cam for _, _, cam,_,_ in gallery])

    # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    # cmc1, mAP1 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    # print('3971:{}\t 14054:{}\t 9594:{}\t 9738:{}\t 2394:{}\t 8208:{}\t 4036:{}\t 4610:{}\t 14855:{}\t 8173:{}'.format(gallery_ids[3971],
    # gallery_ids[14054],gallery_ids[9594],gallery_ids[9738],gallery_ids[2394],gallery_ids[8208],gallery_ids[4036],gallery_ids[4610],gallery_ids[14855],gallery_ids[8173]))
    mAP = mean_ap2(score, res, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(score, res, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    cmc_topk=(1, 5, 10)
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))

class Evaluator(object):
    def __init__(self, backbone):
        super(Evaluator, self).__init__()
        self.backbone = backbone
        self.backbone.eval()  #模型测试时需固定住BN层和dropout层
        self.visualize = False

    # for train
    def evaluate(self, query_loader, gallery_loader, query, gallery, kl=50, is_train=True):
        query_features, _ = extract_features(self.backbone, query_loader)
        gallery_features, _ = extract_features(self.backbone, gallery_loader)
        distmat, query_features, gallery_features = pairwise_distance(query_features, gallery_features, query, gallery)

        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, kl=kl, is_train=is_train)

    def extract_tgt_train_features(self, train_loader, save_name=None, layer=None):
        self.backbone.eval()
        features, paths, labels, camids = [], [], [], []
        for i, (imgs, fpaths, pids, cams, _) in enumerate(train_loader):
            outputs = extract_cnn_feature(self.backbone, imgs, layer)
            for fpath, feature, pid, cam in zip(fpaths, outputs, pids, cams):
                features.append(feature.numpy())
                paths.append(fpath)
                labels.append(pid.numpy())
                camids.append(cam.numpy())
        features, fpaths, labels, camids = \
            np.array(features), np.array(paths), np.array(labels), np.array(camids)
        if save_name:
            save_feature(features, fpaths, labels, camids, save_name)
        return features, fpaths, labels, camids

    def evaluate_fusion(self, query, gallery,fusion_path):
        start_time = time.monotonic()

        predict_path = os.path.join(fusion_path,'cross_filter_pid.log')
        score_path = os.path.join(fusion_path,'cross_filter_score.log')
        # log_path = '/home/zyb/projects/TFusion/SpCL/eval_result.txt'
        result_eval(predict_path,score_path,query,gallery)

        end_time = time.monotonic()
        print('evaluate_on_fusion_time : {}'.format(end_time - start_time))

    def transfer(self, data_loader, train, ks=4, logs_dir=""):
        start_time = time.monotonic()
        features, _ = extract_features(self.backbone, data_loader)
        
        indexs = get_score_pid_on_training(features, train, ks, logs_dir)
        # results = evaluate_market1501(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
        end_time = time.monotonic()
        print('transfer_time : {}'.format(end_time - start_time))
        return indexs

    @staticmethod
    def extract_pids(data_loader):
        labels = []
        for i, (_, _, pids, _, _) in enumerate(data_loader):
            for pid in pids:
                labels.append(pid)

        labels = np.array(labels)
        return labels

    @staticmethod
    def load_feature(name):
        file_dir = '/hdd/sdb/lsc/pytorch/PGM/logs/features_visualization'

        result = sci.loadmat(osp.join(file_dir, name+'_result.mat'))
        features = result['features']
        print(features.shape)
        labels = result['labels']
        fpaths = result['fpaths']
        camids = result['camids']
        labels = np.squeeze(labels)
        fpaths = np.squeeze(fpaths)
        camids = np.squeeze(camids)
        return features, fpaths, labels, camids,
