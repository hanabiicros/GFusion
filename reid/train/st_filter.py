# coding=utf-8
from dataclasses import MISSING
import datetime

from torch import float32, int32
# from spcl.utils import data
from typing import Sequence
# from spcl.datasets import market1501

from torch.utils.data import dataset

from reid.evaluation.ranking import cmc, mean_ap, mean_ap2, map_cmc
from reid.post_process.track_prob import track_score
from reid.profile.fusion_param import get_fusion_param, ctrl_msg
from reid.utils.file_helper import read_lines, read_lines_and, write, safe_remove
from reid.utils.serialize import pickle_load
# from reid import datasets
from reid.dataloaders.dataset import Person
from reid.utils.osutils import mkdir_if_missing

import numpy as np
import os


def smooth_score(c1, c2, time1, time2, camera_delta_s, track_interval=20, filter_interval=1000):
    smooth_window_size = 10
    smooth_scores = [
        track_score(camera_delta_s, c1,
                    time1 - (smooth_window_size / 2 - 1) * track_interval + j * track_interval, c2, time2,
                    interval=track_interval, filter_interval=filter_interval)
        for j in range(smooth_window_size)]
    # filter
    for j in range(smooth_window_size):
        if smooth_scores[j] < 0.01:
            smooth_scores[j] = 0
    # smooth
    score = sum(smooth_scores) / len(smooth_scores)
    return score


def predict_track_scores(real_tracks, camera_delta_s, fusion_param, smooth=False, interval = 25, filter = 40000):
    # fusion_param = get_fusion_param()
    # persons_deltas_score = pickle_load(fusion_param['persons_deltas_path'])
    # if pickle_load(fusion_param['persons_deltas_path']) is not None:
    #     return persons_deltas_score
    # predict_path = fusion_param['renew_pid_path']
    # test_tracks.txt
    # print("len(real_tracks):{}".format(len(real_tracks)))
    len_track = len(real_tracks)
    top_cnt = 10
    # persons_deltas_score = list()
    sts_score = [[0 for i in range(len_track)] for j in range(len_track)]
    # todo 不用读第二遍
    # pids4probes = np.genfromtxt(predict_path, delimiter=' ', dtype = np.int32)
    # for probe_i, pids4probe in enumerate(pids4probes):
    for i in range(len_track):
        # person_deltas_score = list()
        # if probe_i % 100 == 0:
        #     print('%d of %d' % (probe_i, len(pids4probes)))
        temp = 0
        # for pid4probe in pids4probe:
        for j in range(len_track):
            # todo transfer: if predict by python, start from 0, needn't minus 1
            # pid4probe = int(pid4probe)
            # predict_idx = predict_idx - 1
            if len(real_tracks[j]) > 3:
                s1 = real_tracks[j][3]
                s2 = real_tracks[i][3]
                if s1 != s2:
                    # person_deltas_score.append(-1.0)
                    sts_score[i][j] = 0
                    continue
            time1 = real_tracks[j][2]
            # if track_score_idx == 3914:
            #    print('test'
            time2 = real_tracks[i][2]
            c1 = real_tracks[j][1]
            c2 = real_tracks[i][1]
            temp = temp + 1 
            if c1 == c2:
                sts_score[i][j] = 0
            # if smooth:
            #     # if '_dukequerytail' in predict_path:
            #     #     score = track_score(camera_delta_s, c1, time1, c2, time2, interval=500, moving_st=True, filter_interval=50000)
            #     # else:
            #     score = smooth_score(c1, c2, time1, time2, camera_delta_s)
            else:
                # 给定摄像头，时间，获取时空评分，这里camera_deltas如果是随机算出来的，则是随机评分
                # todo grid 需要 改区间大小
                # if '_dukequerytail' in predict_path:
                #     score = track_score(camera_delta_s, c1, time1, c2, time2,
                #                         interval = fusion_param['window_interval'], moving_st=True,
                #                         filter_interval = fusion_param['filter_interval'])
                # else:
                score = track_score(camera_delta_s, c1, time1, c2, time2, interval = interval, filter_interval = filter) #700，40000
                    #  X[probe_i][pid4probe] = score
                    # if(probe_i < 10 and temp < 100):
                    #     print("probe_i:{},real_tracks[probe_i]:{},pid4probe:{},real_tracks[pid4probe]:{},sp_score:{}".format(probe_i,real_tracks[probe_i],pid4probe,real_tracks[pid4probe],score))
            # if(score != -1.0):
                sts_score[i][j] = score
        #     person_deltas_score.append(score)
        # probe_i += 1
        # persons_deltas_score.append(person_deltas_score)
        
    # print("sts_score[0][0]:{},sts_score[0][4]:{}".format(sts_score[0][0],sts_score[0][4]))

    return sts_score


def predict_img_scores(fusion_param):
    # fusion_param = get_fusion_param()
    # final_persons_scores = pickle_load(fusion_param['persons_ap_path'])
    # if pickle_load(fusion_param['persons_ap_path']) is not None:
    #     return final_persons_scores
    predict_score_path = fusion_param['renew_ac_path']
    vision_persons_scores = np.genfromtxt(predict_score_path, delimiter=' ',dtype =np.float32)
    # pickle_save(fusion_param['persons_ap_path'], final_persons_scores)
    return vision_persons_scores


def predict_pids(fusion_param):
    # fusion_param = get_fusion_param()
    # predict_persons = pickle_load(fusion_param['predict_person_path'])
    # if pickle_load(fusion_param['predict_person_path']) is not None:
    #     return predict_persons
    predict_person_path = fusion_param['renew_pid_path']
    predict_persons = np.genfromtxt(predict_person_path, delimiter=' ', dtype = np.int32)
    # pickle_save(fusion_param['predict_person_path'], predict_persons)
    return predict_persons


def get_person_pids(predict_path):
    predict_person_path = predict_path
    predict_persons = np.genfromtxt(predict_person_path, delimiter=' ',dtype = np.int32)
    return predict_persons

def get_data(source, target, data_dir):
    # root = os.path.join(data_dir, name)
    # dataset = datasets.create(name, root)
    # return dataset
    dataset = Person(data_dir, target, source)
    return dataset

# def train_tracks(fusion_param, source, target):
#     # dataset = get_data('dukemtmc','/hdd/sdb/zyb/TFusion/SpCL/data')
#     # train = dataset.train
#     train = get_data(source,target,'/hdd/sdb/lsc/dataset_person')
#     real_tracks = [[pid,camid,frameid,sequenceid]for _, pid, camid,sequenceid,frameid in train]
#     return real_tracks
    # answer_path = fusion_param['answer_path']
    # answer_lines = read_lines(answer_path)
    # # 左图
    # real_tracks = list()
    # for answer in answer_lines:
    #     info = answer.split('_')
    #     if 'bmp' in info[2]:
    #         info[2] = info[2].split('.')[0]
    #     if len(info) > 4 and 'jpe' in info[6]:
    #         real_tracks.append([info[0], int(info[1][0]), int(info[2])])
    #     elif 'f' in info[2]:
    #         real_tracks.append([info[0], int(info[1][1]), int(info[2][1:-5]), 1])
    #     else:
    #         real_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    # return real_tracks


def fusion_st_img_ranker(fusion_param, source = 'DukeMTMC-reID', target = 'market', data_path = '/hdd/sdb/lsc/dataset_person', interval=25, scores = None):
    ep = fusion_param['ep']
    en = fusion_param['en']
    # print('ep:{}\t en:{}'.format(ep,en))
    # 从renew_pid和renew_ac获取预测的人物id和图像分数
    # persons_ap_scores = predict_img_scores(fusion_param)
    persons_ap_scores = scores
    # persons_ap_pids = predict_pids(fusion_param)
    # 从磁盘获取之前建立的时空模型，以及随机时空模型
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    # rand_delta_s = pickle_load(fusion_param['rand_distribution_pickle_path'])
    # diff_delta_s = pickle_load(fusion_param['rand_distribution_pickle_path'].replace('rand', 'diff'))

    dataset = get_data(source, target, data_path)
    train = dataset.target_train_real_s2
    real_tracks = [[pid,camid,frameid,sequenceid]for _, pid, camid,sequenceid,frameid in train]
    # 计算时空评分和随机时空评分
    print("时空评分计算...")
    if target == 'msmt17':
        persons_track_scores = predict_track_scores(real_tracks, camera_delta_s, fusion_param, interval = interval, filter = 1000)
    else:
        persons_track_scores = predict_track_scores(real_tracks, camera_delta_s, fusion_param, interval = interval)
    persons_track_scores = np.array(persons_track_scores)
    # eval_path = 'eval/market2duke-train'
    # mkdir_if_missing(eval_path)
    # score_path = os.path.join(eval_path, 'st_score.txt')
    # np.savetxt(score_path, sts_person, fmt='%.7f')
    return persons_track_scores

    print("随机时空评分计算...")
    rand_track_scores = predict_track_scores(real_tracks, rand_delta_s, fusion_param, interval = interval)
    # diff_track_scores = rand_track_scores # predict_track_scores(real_tracks, diff_delta_s, fusion_param)

    persons_cross_scores = [[0 for i in range(len(scores))] for j in range(len(scores))]
    # log_path = fusion_param['eval_fusion_path']
    # map_score_path = fusion_param['fusion_normal_score_path']
    # safe_remove(map_score_path)
    # safe_remove(log_path)
    # line_log_cnt = 10

    for i, person_ap_scores in enumerate(persons_ap_scores):
        cur_max_vision = max(person_ap_scores)
        cur_min_vision = min(person_ap_scores)
        persons_ap_scores[i] = (persons_ap_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
        persons_ap_scores[i] = np.exp(persons_ap_scores[i] * 3)
        cur_max_vision = max(persons_ap_scores[i])
        cur_min_vision = min(persons_ap_scores[i])
        persons_ap_scores[i] = (persons_ap_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)

    # for i, person_ap_pids in enumerate(persons_ap_pids):
    for i in range(len(scores)):
        # cross_scores = list()
        for j in range(len(scores[0])):
            if persons_track_scores[i][j] == -1 or persons_track_scores[i][j] == 0:
                continue
            elif rand_track_scores[i][j] < 0.00002:
                cross_score = (persons_track_scores[i][j]) * (persons_ap_scores[i][j]) / 0.00002
            else:
                cross_score = (persons_track_scores[i][j]) * (persons_ap_scores[i][j]) / rand_track_scores[i][j]
            persons_cross_scores[i][j] = cross_score
        #     cross_scores.append(cross_score)
        # persons_cross_scores.append(cross_scores)
    print('img score ready')
    max_score = max([max(predict_cross_scores) for predict_cross_scores in persons_cross_scores])
    print('max_cross_score %f' % max_score)

    persons_cross_scores = np.array(persons_cross_scores)

    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            if person_cross_score > 0: # diff camera
                # diff seq not sort and not normalize
                persons_cross_scores[i][j] /= max_score
            else: # same camera and (diff camere && rand score == -1    )
                persons_cross_scores[i][j] = persons_ap_scores[i][j]
            # if real_tracks[i][1] == real_tracks[persons_ap_pids[i][j]][1]:
            #     # print('%d, %d' % (i, j)
            #     persons_cross_scores[i][j] = 0
            if i < 5 and j < 20:
                print('st:%f st_rand%f vis:%f fus:%f' % (
                    persons_track_scores[i][j], rand_track_scores[i][j], persons_ap_scores[i][j], persons_cross_scores[i][j]))

    return persons_cross_scores

    person_score_idx_s = list()
    top1_scores = list()
    print('above person score ready')
    for i, person_cross_scores in enumerate(persons_cross_scores):
        # 单个probe的预测结果中按score排序，得到index，用于对pid进行排序
        sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
        person_score_idx_s.append(sort_score_idx_s)
        # 统计top1分布，后面计算中位数用
        top1_scores.append(person_cross_scores[sort_score_idx_s[0]])
    # 降序排，取前60%处的分数
    sorted_top1_scores = sorted(top1_scores, reverse=True)
    mid_score = sorted_top1_scores[int(len(sorted_top1_scores) * 0.5)]
    mid_score_path = fusion_param['mid_score_path']
    safe_remove(mid_score_path)
    write(mid_score_path, '%f\n' % mid_score)
    print(str(mid_score))
    sorted_persons_ap_pids = np.zeros([len(persons_ap_pids), len(persons_ap_pids[0])])
    sorted_persons_ap_scores = np.zeros([len(persons_ap_pids), len(persons_ap_pids[0])])
    for i, person_ap_pids in enumerate(persons_ap_pids):
        for j in range(len(person_ap_pids)):
            sorted_persons_ap_pids[i][j] = persons_ap_pids[i][person_score_idx_s[i][j]]
            sorted_persons_ap_scores[i][j] = persons_cross_scores[i][person_score_idx_s[i][j]]
    np.savetxt(log_path, sorted_persons_ap_pids, fmt='%d')
    np.savetxt(map_score_path, sorted_persons_ap_scores, fmt='%f')


def simple_fusion_st_img_ranker(fusion_param):
    ep = fusion_param['ep']
    en = fusion_param['en']
    # 从renew_pid和renew_ac获取预测的人物id和图像分数
    persons_ap_scores = predict_img_scores(fusion_param)
    persons_ap_pids = predict_pids(fusion_param)
    # 从磁盘获取之前建立的时空模型
    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])

    real_tracks = train_tracks(fusion_param)
    # 计算时空评分
    persons_track_scores = predict_track_scores(real_tracks, camera_delta_s, fusion_param)

    persons_cross_scores = list()
    log_path = fusion_param['eval_fusion_path']
    map_score_path = fusion_param['fusion_normal_score_path']
    safe_remove(map_score_path)
    safe_remove(log_path)
    line_log_cnt = 10

    for i, person_ap_pids in enumerate(persons_ap_pids):
        cross_scores = list()
        for j, person_ap_pid in enumerate(person_ap_pids):
            cross_score = persons_track_scores[i][j] * persons_ap_scores[i][j]
            cross_scores.append(cross_score)
        persons_cross_scores.append(cross_scores)
    print('img score ready')
    max_score = max([max(predict_cross_scores) for predict_cross_scores in persons_cross_scores])

    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            if person_cross_score > 0:
                # diff seq not sort and not normalize
                persons_cross_scores[i][j] /= max_score
            else:
                persons_cross_scores[i][j] *= -1.
            # if real_tracks[i][1] == real_tracks[persons_ap_pids[i][j]][1]:
            #     # print('%d, %d' % (i, j)
            #     persons_cross_scores[i][j] = 0
    person_score_idx_s = list()
    top1_scores = list()
    print('above person score ready')
    for i, person_cross_scores in enumerate(persons_cross_scores):
        # 单个probe的预测结果中按score排序，得到index，用于对pid进行排序
        sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
        person_score_idx_s.append(sort_score_idx_s)
        # 统计top1分布，后面计算中位数用
        top1_scores.append(person_cross_scores[sort_score_idx_s[0]])
    # 降序排，取前60%处的分数
    sorted_top1_scores = sorted(top1_scores, reverse=True)
    mid_score = sorted_top1_scores[int(len(sorted_top1_scores) * 0.5)]
    mid_score_path = fusion_param['mid_score_path']
    safe_remove(mid_score_path)
    write(mid_score_path, '%f\n' % mid_score)
    print(str(mid_score))
    sorted_persons_ap_pids = np.zeros([len(persons_ap_pids), len(persons_ap_pids[0])])
    sorted_persons_ap_scores = np.zeros([len(persons_ap_pids), len(persons_ap_pids[0])])
    for i, person_ap_pids in enumerate(persons_ap_pids):
        for j in range(len(person_ap_pids)):
            sorted_persons_ap_pids[i][j] = persons_ap_pids[i][person_score_idx_s[i][j]]
            sorted_persons_ap_scores[i][j] = persons_cross_scores[i][person_score_idx_s[i][j]]
    np.savetxt(log_path, sorted_persons_ap_pids, fmt='%d')
    np.savetxt(map_score_path, sorted_persons_ap_scores, fmt='%f')

def gallery_track_scores(query_tracks, gallery_tracks, camera_delta_s, indexs, fusion_param, smooth=False, interval=50, filter=50000):
    # predict_path = fusion_param['renew_pid_path']

    # persons_deltas_score = list()
    # len_query = len(query_tracks)
    # len_gallery = len(gallery_tracks)
    sts_scores = []
    # pids4probes = np.genfromtxt(predict_path, delimiter=' ', dtype = np.int32)
    for i in range(len(indexs)):
        st_scores = []
        for j in indexs[i]:
            # if i >= top_cnt:
            #     break
            # pid4probe = int(pid4probe)
            # if len(query_tracks[0]) > 3:
            #     # market index minus 1
            #     probe_i_tmp = probe_i # (probe_i + 1) % len(pids4probes)
            # else:
            #     probe_i_tmp = probe_i
            # # todo transfer: if predict by python, start from 0, needn't minus 1
            # # predict_idx = predict_idx - 1
            if len(query_tracks[i]) > 3:
                s1 = query_tracks[i][3]
                # print(predict_idx
                s2 = gallery_tracks[j][3]
                if s1 != s2:
                    # person_deltas_score[i] = -1.0
                    st_scores.append(0.)
                    continue
            time1 = query_tracks[i][2]
            # if track_score_idx == 3914:
            #     print('test'
            time2 = gallery_tracks[j][2]
            c1 = query_tracks[i][1]
            c2 = gallery_tracks[j][1]
            score = track_score(camera_delta_s, c1, time1, c2, time2, interval = interval, filter_interval = filter)
            # if(score != -1.0):
            #     sts_score[i][j] = score
            st_scores.append(score)
        sts_scores.append(st_scores)

    return sts_scores



def fusion_st_gallery_ranker(fusion_param, source = 'DukeMTMC-reID', target = 'market',data_path='/hdd/sdb/zyb/TFusion/SpCL/data',interval=50, scores=None, gscores=None, qgindexs=None, ggindexs=None):
    # ep = fusion_param['ep']
    # en = fusion_param['en']
    #print('ep:{}\t en:{}'.format(ep,en))
    # log_path = fusion_param['eval_fusion_path']
    # map_score_path = fusion_param['fusion_normal_score_path']  # fusion_param = get_fusion_param()
    
    dataset = get_data(source, target, data_path)
    query = dataset.target_query
    gallery = dataset.target_gallery

    query_tracks = [[pid,camid,frameid,sequenceid]for _, pid, camid,sequenceid,frameid in query]
    gallery_tracks = [[pid,camid,frameid,sequenceid]for _, pid, camid,sequenceid,frameid in gallery]
    len_query = len(query_tracks)
    len_gallery = len(gallery_tracks)
    print('probe and gallery tracks ready')
    # persons_ap_scores = predict_img_scores(fusion_param)
    # persons_ap_scores = scores
    # persons_gg_scores = gscores
    # persons_ap_pids = predict_pids(fusion_param)
    # print('read vision scores and pids ready')
    # if 'market_market' in log_path:
    #     scale = 10
    # else:
    scale = 3 # 1.5 for direct fusion
    
    # for i, person_ap_scores in enumerate(persons_ap_scores):
    #     cur_max_vision = max(person_ap_scores)
    #     cur_min_vision = min(person_ap_scores)
    #     persons_ap_scores[i] = (persons_ap_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
    #     persons_ap_scores[i] = np.exp(persons_ap_scores[i] * scale)
    #     cur_max_vision = max(persons_ap_scores[i])
    #     cur_min_vision = min(persons_ap_scores[i])
    #     persons_ap_scores[i] = (persons_ap_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
    
    # for i, person_gg_scores in enumerate(persons_gg_scores):
    #     cur_max_vision = max(person_gg_scores)
    #     cur_min_vision = min(person_gg_scores)
    #     persons_gg_scores[i] = (persons_gg_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
    #     persons_gg_scores[i] = np.exp(persons_gg_scores[i] * scale)
    #     cur_max_vision = max(persons_gg_scores[i])
    #     cur_min_vision = min(persons_gg_scores[i])
    #     persons_gg_scores[i] = (persons_gg_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)


    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])  #获得同一行人在不同摄像头之间迁移的时间差
    # camera_delta_s = pickle_load('true_market_probe.pck')
    print('load track deltas ready')
    # rand_delta_s = pickle_load(fusion_param['rand_distribution_pickle_path'])
    # print('load rand deltas ready')
    # diff_delta_s = pickle_load(fusion_param['rand_distribution_pickle_path'].replace('rand', 'diff'))
    # print('load diff deltas ready')
    # # todo tmp diff deltas
    # # diff_delta_s = rand_delta_s
    # rand_track_scores = gallery_track_scores(query_tracks, gallery_tracks, rand_delta_s, fusion_param, smooth=False,interval=interval)
    # rand_gg_scores = gallery_track_scores(gallery_tracks, gallery_tracks, rand_delta_s, fusion_param, smooth=False,interval=interval)
    # print('rand scores ready')
    # smooth = '_grid' in log_path
    # smooth = True
    if target == 'msmt17':
        persons_track_scores = gallery_track_scores(query_tracks, gallery_tracks, camera_delta_s, qgindexs, fusion_param, smooth=False, interval=interval, filter=1000)
        gg_track_scores = gallery_track_scores(gallery_tracks, gallery_tracks, camera_delta_s, ggindexs, fusion_param, smooth=False, interval=interval, filter=1000)
    else:
        persons_track_scores = gallery_track_scores(query_tracks, gallery_tracks, camera_delta_s, qgindexs, fusion_param, smooth=False,interval=interval)
        gg_track_scores = gallery_track_scores(gallery_tracks, gallery_tracks, camera_delta_s, ggindexs, fusion_param, smooth=False,interval=interval)
    
    persons_track_scores = np.array(persons_track_scores, dtype=np.float16)
    gg_track_scores = np.array(gg_track_scores, dtype=np.float16)
    print('track scores ready')
    return persons_track_scores, gg_track_scores

    # diff_track_scores = gallery_track_scores(query_tracks, gallery_tracks, diff_delta_s, fusion_param, smooth=smooth)
    # print('diff track score ready')
    # todo tmp diff scores
    # diff_track_scores = rand_track_scores

    persons_cross_scores = list()
    persons_cross_gg_scores = list()
    # safe_remove(map_score_path)
    # safe_remove(log_path)

    # fusion_track_scores = np.zeros([len(persons_ap_pids), len(persons_ap_pids[0])])
    # for i, person_ap_pids in enumerate(persons_ap_pids):
    #     for j, person_ap_pid in enumerate(person_ap_pids):
    #         cur_track_score = persons_track_scores[i][j]
    #         rand_track_score = rand_track_scores[i][j]
    #         if rand_track_score < 0.00002:
    #             rand_track_score = 0.00002
    #         fusion_track_scores[i][j] = (cur_track_score * (1 - ep) - en * diff_track_scores[i][j]) / rand_track_score
    # for i, person_ap_pids in enumerate(persons_ap_pids):
    #     cur_max_predict = max(persons_track_scores[i])
    #     cur_max_rand = max(rand_track_scores[i])
    #     for j in range(len(fusion_track_scores[i])):
    #         if fusion_track_scores[i][j] >= 0:
    #             fusion_track_scores[i][j] /= cur_max_predict/cur_max_rand
    #         else:
    #             fusion_track_scores[i][j] = 1.

    min_rand = 1e-3 # 0.00002
    # min_rand = 2e-5 # 0.00002
    # if 'market_market' in log_path:
    #     min_rand = 1e-2  # 0.00002
    
    for i in range(len_query):
        cross_scores = list()
        for j in range(len_gallery):
            cur_track_score = persons_track_scores[i][j]
            rand_track_score = rand_track_scores[i][j]
            if rand_track_score <  0:
                rand_track_score =min_rand
            elif rand_track_score < min_rand:
                rand_track_score = min_rand
                if cur_track_score != 0:
                    cur_track_score = -1

            cross_score = (cur_track_score * persons_ap_scores[i][j]) / rand_track_score
            # if i <= 3 and j < 30:
            #     print(cur_track_score, persons_ap_scores[i][j], rand_track_score, cross_score)
            if cur_track_score > 0 and cross_score < 0:
                cross_score = 0
            cross_scores.append(cross_score)
        # if max(cross_scores) == 0:
        #     print('max cross scores %d is 0' % i)
        persons_cross_scores.append(cross_scores)
    print('fusion scores ready')

    for i in range(len_gallery):
        cross_scores = list()
        for j in range(len_gallery):
            cur_track_score = gg_track_scores[i][j]
            rand_track_score = rand_gg_scores[i][j]
            if rand_track_score <  0:
                rand_track_score =min_rand
            elif rand_track_score < min_rand:
                rand_track_score = min_rand
                if cur_track_score != 0:
                    cur_track_score = -1

            cross_score = (cur_track_score * persons_gg_scores[i][j]) / rand_track_score
            # if i <= 3 and j < 30:
            #     print(cur_track_score, persons_ap_scores[i][j], rand_track_score, cross_score)
            if cur_track_score > 0 and cross_score < 0:
                cross_score = 0
            cross_scores.append(cross_score)
        # if max(cross_scores) == 0:
        #     print('max cross scores %d is 0' % i)
        persons_cross_gg_scores.append(cross_scores)
    print('fusion gg scores ready')
        # pickle_save(ctrl_msg['data_folder_path']+'viper_r-testpersons_cross_scores.pick', persons_cross_scores)
        # pickle_save(ctrl_msg['data_folder_path']+'viper_r-testpersons_ap_pids.pick', persons_ap_pids)

    max_score_s = [max(predict_cross_scores) for predict_cross_scores in persons_cross_scores]
    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            if persons_cross_scores[i][j] >= 0:
                # diff seq not sort, not rank for max, and not normalize
                if max_score_s[i] != 0:
                    # there exist probe track with same seq, diff camera but value > 1000
                    # print('max_score_s %d is 0' % i)
                # else:
                    persons_cross_scores[i][j] /= max_score_s[i]
                # persons_cross_scores[i][j] /= max_score
                # if persons_cross_scores[i][j] > 0.5:
                #     print('same'
                #     print(persons_cross_scores[i][j]
            else:
                # so diff seq is negative, normalize by minimum
                # persons_cross_scores[i][j] /= min_score_s[i]
                # persons_cross_scores[i][j] *= 1.0
                persons_cross_scores[i][j] *= -1 * min_rand
            if i == 0 and j % 100 == 0:
                print('track: %f vision: %f rand: %f final: %f' % (
                persons_track_scores[i][j],  persons_ap_scores[i][j], rand_track_scores[i][j], persons_cross_scores[i][j]))
    
    print('fusion scores normalized, diff seq use vision score to rank')
    
    max_score_s = [max(predict_cross_scores) for predict_cross_scores in persons_cross_gg_scores]
    for i, person_cross_scores in enumerate(persons_cross_gg_scores):
        for j, person_cross_score in enumerate(persons_cross_gg_scores):
            if persons_cross_gg_scores[i][j] >= 0:
                # diff seq not sort, not rank for max, and not normalize
                if max_score_s[i] != 0:
                    persons_cross_gg_scores[i][j] /= max_score_s[i]
            else:
                persons_cross_gg_scores[i][j] *= -1 * min_rand
            if i == 0 and j % 100 == 0:
                print('track: %f vision: %f rand: %f final: %f' % (
                gg_track_scores[i][j],  persons_gg_scores[i][j], rand_gg_scores[i][j], persons_cross_gg_scores[i][j]))
    
    persons_cross_scores = np.array(persons_cross_scores, dtype=np.float32)
    persons_cross_gg_scores = np.array(persons_cross_gg_scores, dtype=np.float32)
    print('fusion gg scores normalized, diff seq use vision score to rank')
    # person_score_idx_s = list()

    # for i, person_cross_scores in enumerate(persons_cross_scores):
    #     # 单个probe的预测结果中按score排序，得到index，用于对pid进行排序
    #     sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
    #     person_score_idx_s.append(sort_score_idx_s)
    # # sorted_persons_ap_pids = np.zeros([len_query, len_gallery])
    # # sorted_persons_ap_pids = np.array(person_score_idx_s, dtype=np.int32)
    # sorted_persons_ap_scores = np.zeros([len_query, len_gallery], dtype=np.float32)
    # for i in range(len_query):
    #     for j in range(len_gallery):
    #         # sorted_persons_ap_pids[i][j] = persons_ap_pids[i][person_score_idx_s[i][j]]
    #         # sorted_persons_ap_pids[i][j] = person_score_idx_s[i][j]
    #         sorted_persons_ap_scores[i][j] = persons_cross_scores[i][person_score_idx_s[i][j]]
    
    # Testing
    # query_ids = np.asarray([pid for _, pid, _,_,_ in query])
    # gallery_ids = np.asarray([pid for _, pid, _ ,_,_ in gallery])
    # query_cams = np.asarray([cam for _, _, cam,_,_ in query])
    # gallery_cams = np.asarray([cam for _, _, cam,_,_ in gallery])

    # # mAP = mean_ap2(sorted_persons_ap_scores, person_score_idx_s, query_ids, gallery_ids, query_cams, gallery_cams)
    # mAP, all_cmc = map_cmc(-persons_cross_scores, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))
    # print('CMC Scores')
    # for k in (1, 5, 10):
    #     print('  top-{:<4}{:12.1%}'
    #           .format(k, all_cmc[k - 1]))


    # print('sorted scores ready')
    # np.savetxt(log_path, sorted_persons_ap_pids, fmt='%d')
    # np.savetxt(map_score_path, sorted_persons_ap_scores, fmt='%f')
    # print('save sorted fusion scores')

    # for i, person_ap_pids in enumerate(persons_ap_pids):
    #     for j in range(len(person_ap_pids)):
    #         write(log_path, '%d ' % person_ap_pids[person_score_idx_s[i][j]])
    #         write(map_score_path, '%.3f ' % persons_cross_scores[i][person_score_idx_s[i][j]])
    #     write(log_path, '\n')
    #     write(map_score_path, '\n')
    return persons_cross_scores, persons_cross_gg_scores


def simple_fusion_st_gallery_ranker(fusion_param,dataset='market1501',data_path='/hdd/sdb/zyb/TFusion/SpCL/data',interval=50):
    ep = fusion_param['ep']
    en = fusion_param['en']
    log_path = fusion_param['eval_fusion_path']
    map_score_path = fusion_param['fusion_normal_score_path']  # fusion_param = get_fusion_param()

    dataset = get_data(dataset,data_path)
    query = dataset.query
    gallery = dataset.gallery
    query_tracks = [[pid,camid,frameid,sequenceid]for _, pid, camid,sequenceid,frameid in query]

    gallery_tracks = [[pid,camid,frameid,sequenceid]for _, pid, camid,sequenceid,frameid in gallery]
    # answer path is probe path
    # answer_path = fusion_param['answer_path']
    # answer_lines = read_lines(answer_path)
    # query_tracks = list()
    # for answer in answer_lines:
    #     info = answer.split('_')
    #     if 'bmp' in info[2]:
    #         info[2] = info[2].split('.')[0]
    #     if len(info) > 4 and 'jpe' in info[6]:
    #         query_tracks.append([info[0], int(info[1][0]), int(info[2])])
    #     elif 'f' in info[2]:
    #         query_tracks.append([info[0], int(info[1][1]), int(info[2][1:-5]), 1])
    #     else:
    #         query_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    # gallery_path = fusion_param['gallery_path']
    # gallery_lines = read_lines(gallery_path)
    # gallery_tracks = list()
    # for gallery in gallery_lines:
    #     info = gallery.split('_')
    #     if 'bmp' in info[2]:
    #         info[2] = info[2].split('.')[0]
    #     if len(info) > 4 and 'jpe' in info[6]:
    #         gallery_tracks.append([info[0], int(info[1][0]), int(info[2])])
    #     elif 'f' in info[2]:
    #         gallery_tracks.append([info[0], int(info[1][1]), int(info[2][1:-5]), 1])
    #     else:
    #         gallery_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    print('probe and gallery tracks ready')
    persons_ap_scores = predict_img_scores(fusion_param)
    persons_ap_pids = predict_pids(fusion_param)
    print('read vision scores and pids ready')
    if fusion_param['gt_fusion']:
        scale = 6.
    else:
        scale = 3.
    # if 'market_market' in log_path:
    if True:
        # if 'market_market' in log_path:
        for i, person_ap_scores in enumerate(persons_ap_scores):
            cur_max_vision = max(person_ap_scores)
            cur_min_vision = min(person_ap_scores)
            persons_ap_scores[i] = (persons_ap_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)
            persons_ap_scores[i] = np.exp(persons_ap_scores[i] * scale)
            cur_max_vision = max(persons_ap_scores[i])
            cur_min_vision = min(persons_ap_scores[i])
            persons_ap_scores[i] = (persons_ap_scores[i] - cur_min_vision) / (cur_max_vision - cur_min_vision)

    camera_delta_s = pickle_load(fusion_param['distribution_pickle_path'])
    # camera_delta_s = pickle_load('true_market_probe.pck')
    print('load track deltas ready')
    smooth = '_grid' in log_path
    # smooth = True
    persons_track_scores = gallery_track_scores(query_tracks, gallery_tracks, camera_delta_s, fusion_param, smooth=smooth,interval=interval)
    print('track scores ready')

    persons_cross_scores = list()
    safe_remove(map_score_path)
    safe_remove(log_path)

    for i, person_ap_pids in enumerate(persons_ap_pids):
        cross_scores = list()
        for j, person_ap_pid in enumerate(person_ap_pids):
            cur_track_score = persons_track_scores[i][j]
            cross_score = cur_track_score * persons_ap_scores[i][j]
            cross_scores.append(cross_score)
        if max(cross_scores) == 0:
            print('max_cross_scores %d is 0' % i)
        persons_cross_scores.append(cross_scores)
    print('fusion scores ready')
    max_score_s = [max(predict_cross_scores) for predict_cross_scores in persons_cross_scores]
    for i, person_cross_scores in enumerate(persons_cross_scores):
        for j, person_cross_score in enumerate(person_cross_scores):
            if persons_cross_scores[i][j] >= 0:
                # diff seq not sort, not rank for max, and not normalize
                if max_score_s[i] == 0:
                    # there exist probe track with same seq, diff camera but value > 1000
                    print('max_score_s %d is 0' % i)
                else:
                    persons_cross_scores[i][j] /= max_score_s[i]
            else:
                persons_cross_scores[i][j] *= -1.
    print('fusion scores normalized, diff seq use vision score to rank')
    person_score_idx_s = list()

    for i, person_cross_scores in enumerate(persons_cross_scores):
        # 单个probe的预测结果中按score排序，得到index，用于对pid进行排序
        sort_score_idx_s = sorted(range(len(person_cross_scores)), key=lambda k: -person_cross_scores[k])
        person_score_idx_s.append(sort_score_idx_s)
    sorted_persons_ap_pids = np.zeros([len(persons_ap_pids), len(persons_ap_pids[0])])
    sorted_persons_ap_scores = np.zeros([len(persons_ap_pids), len(persons_ap_pids[0])])
    for i, person_ap_pids in enumerate(persons_ap_pids):
        for j in range(len(person_ap_pids)):
            sorted_persons_ap_pids[i][j] = persons_ap_pids[i][person_score_idx_s[i][j]]
            sorted_persons_ap_scores[i][j] = persons_cross_scores[i][person_score_idx_s[i][j]]
    print('sorted scores ready')
    safe_remove(log_path)
    safe_remove(map_score_path)
    np.savetxt(log_path, sorted_persons_ap_pids, fmt='%d')
    np.savetxt(map_score_path, sorted_persons_ap_scores, fmt='%f')
    print('save sorted fusion scores')
    return person_score_idx_s

if __name__ == '__main__':
    # ctrl_msg['data_folder_path'] = 'cuhk_duke-train'
    ctrl_msg['data_folder_path'] = 'duke_market-test'
    ctrl_msg['ep'] = 0.0
    ctrl_msg['en'] = 0.0
    # fusion_param = get_fusion_param()
    # fusion_st_img_ranker(fusion_param, fusion_param['pos_shot_rate'], fusion_param['neg_shot_rate'])
    # eval_on_train_test(fusion_param, test_mode=True)
    fusion_param = get_fusion_param()
    # fusion_param['distribution_pickle_path'] = fusion_param['distribution_pickle_path'].replace('duketail', 'duke')
    # fusion_param['rand_distribution_pickle_path'] = fusion_param['rand_distribution_pickle_path'].replace('duketail', 'duke')
    # fusion_st_img_ranker(fusion_param)
    fusion_st_gallery_ranker(fusion_param)
    # os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #           + 'grid-cv-0 /home/cwh/coding/TrackViz/' + fusion_param['eval_fusion_path'])
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #           + 'duke /home/cwh/coding/TrackViz/' + fusion_param['eval_fusion_path'])
    # fusion_st_img_ranker(fusion_param)
    # delta_range, over_probs = fusion_curve(fusion_param)
    # viz_fusion_curve(delta_range, [over_probs])
    # pt = fusion_heat(fusion_param)
    # viz_heat_map(pt)

