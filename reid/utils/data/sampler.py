from __future__ import absolute_import
from collections import defaultdict
import math
import sys
import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

class MoreCameraSampler(Sampler):
    def __init__(self, data_source, num_instances=4, video=False):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances
        self.video = video

        if self.video:
            for index, (_, pid, cam, _) in enumerate(data_source):
                if (pid < 0): continue
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)
        else:
            for index, (_, pid, cam, _) in enumerate(data_source):
                if (pid < 0): continue
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            if self.video:
                _, i_pid, i_cam, _ = self.data_source[i]
            else:
                _, i_pid, i_cam, _ = self.data_source[i]

            cams = self.pid_cam[i_pid]
            index = self.pid_index[i_pid]

            unique_cams = set(cams)
            cams = np.array(cams)
            index = np.array(index)
            select_indexes = []
            for cam in unique_cams:
                select_indexes.append(np.random.choice(index[cams==cam], size=1, replace=False))
            select_indexes = np.concatenate(select_indexes)
            if len(select_indexes)< self.num_instances:
                diff_indexes = np.setdiff1d(index, select_indexes)
                if len(diff_indexes) == 0:
                    select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=True)
                elif len(diff_indexes) >= (self.num_instances-len(select_indexes)):
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances-len(select_indexes)), replace=False)
                else:
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances-len(select_indexes)), replace=True)
                select_indexes = np.concatenate([select_indexes, diff_indexes])
            else:
                select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=False)
            ret.extend(select_indexes)
        return iter(ret)

class RandomIdentityCameraSampler(Sampler):
    ## 每个行人至少挑2个摄像头的图片
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.pid_cid_index = defaultdict(dict)
        for index, (_, pid, cid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
            if cid not in self.pid_cid_index[pid].keys():
                self.pid_cid_index[pid][cid] = []
            self.pid_cid_index[pid][cid].append(index)

    def __len__(self):
        length = 0
        for p in self.index_dic.keys():
            length += len(self.index_dic[p]) // self.num_instances
        return length

    def check_valid(self, pid_index):
        valid = 0
        for p in pid_index.keys():
            if len(pid_index[p]) >= self.num_instances:
                valid += 1
        return valid >= 1
        
    def __iter__(self):
        ret = []
        pid_cid_index = copy.deepcopy(self.pid_cid_index)
        pid_index = copy.deepcopy(self.index_dic)
        while self.check_valid(pid_index):
            pids = torch.randperm(len(pid_index.keys())).tolist()
            random.shuffle(pids)
            for p in pids:
                if len(pid_index[p]) < self.num_instances:
                    continue
                else:
                    ele = []
                    cams = list(pid_cid_index[p].keys())
                    random.shuffle(cams)
                    cidx = 0
                    while len(ele) < self.num_instances:
                        ## 从一个cam里面取图片
                        cid = cams[cidx]
                        c_idxs = pid_cid_index[p][cid]
                        if len(c_idxs) > 0:
                            e = random.choice(c_idxs)
                            ele.append(e)
                            pid_cid_index[p][cid].remove(e)
                            pid_index[p].remove(e)
                        cidx += 1
                        cidx %= len(cams)
                    ret.extend(ele)
        return iter(ret)

class RandomCameraSampler(Sampler):
    ## 每个摄像头随机挑4张
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, _, cid) in enumerate(data_source):
            self.index_dic[cid].append(index)
        self.cids = list(self.index_dic.keys())
        max_idxs = 0
        for c in self.cids:
            random.shuffle(self.index_dic[c])
            max_idxs = max(max_idxs, len(self.index_dic[c]))
        self.max_idxs = max_idxs
        self.num_samples = len(self.cids)
        print('sampler : max_idxs {}'.format(self.max_idxs))

    def check_valid(self, idx_cids):
        valid_cids = 0
        for c in idx_cids.keys():
            if len(idx_cids[c]) > self.num_instances: 
                valid_cids += 1
        return valid_cids >= 2

    def __len__(self):
        length = 0 
        for c in self.cids:
            length += (len(self.index_dic[c]) // self.num_instances)
        return length

    def __iter__(self):
        ret = []
        idx_cids = copy.deepcopy(self.index_dic)
        while self.check_valid(idx_cids):
            # indices = torch.randperm(self.num_samples).tolist()
            random.shuffle(self.cids)
            for cid in self.cids:
                t = idx_cids[cid]
                if len(t) < self.num_instances:
                    continue
                else:
                    ele = np.random.choice(t, size=self.num_instances, replace=False)
                    ret.extend(ele)
                    ele = list(ele)
                    for e in ele:
                        idx_cids[cid].remove(e)
        return iter(ret)

class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances
        try:
            for index, (_, pid, cam) in enumerate(data_source):
                if (pid<0): continue
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)
        except Exception:
            for index, (_, pid, cam, _) in enumerate(data_source):
                if (pid<0): continue
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            try:
                _, i_pid, i_cam = self.data_source[i]
            except Exception:
                _, i_pid, i_cam, _ = self.data_source[i]
            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam) ## 跨摄像头的图片下标

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i) ## 其他非同图片的下标
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])
        return iter(ret)
