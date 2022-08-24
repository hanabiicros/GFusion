import os
import os.path as osp
import re
from glob import glob
from collections import defaultdict
import numpy as np


class Person(object):
    def __init__(self, data_dir, target, source):
        self.target = target
        self.source = source

        # source / target image root
        self.target_root = osp.join(data_dir, self.target)
        self.source_root = osp.join(data_dir, self.source)
        self.list_train_path = osp.join(self.target_root, 'list_train.txt')
        self.list_query_path = osp.join(self.target_root, 'list_query.txt')
        self.list_gallery_path = osp.join(self.target_root, 'list_gallery.txt')
        if self.target == 'msmt17':
            self.train_path = 'mask_train_v2'
            self.gallery_path = 'mask_test_v2'
            self.query_path = 'mask_test_v2'
        else:
            self.train_path = 'bounding_box_train'
            self.camstyle_train_path = 'bounding_box_train_camstyle'
            self.gallery_path = 'bounding_box_test'
            self.query_path = 'query'

        # source / target misc
        self.target_train_fake, self.target_gallery, self.target_query = [], [], []
        self.target_train_real = []
        self.source_train, self.source_gallery, self.source_query = [], [], []

        self.t_train_pids, self.t_gallery_pids, self.t_query_pids = 0, 0, 0
        self.s_train_pids, self.s_gallery_pids, self.s_query_pids = 0, 0, 0
        self.t_train_cam_2_imgs, self.s_train_cam_2_imgs = [], []
        self.target_cam_nums = self.set_cam_dict(self.target)
        self.source_cam_nums = self.set_cam_dict(self.source)

        self.load_dataset()

    @staticmethod
    def set_cam_dict(name):
        cam_dict = {}
        cam_dict['market'] = 6
        cam_dict['DukeMTMC-reID'] = 8
        cam_dict['msmt17'] = 15
        cam_dict['cuhk03'] = 2
        cam_dict['MSMT17_V1'] = 15
        return cam_dict[name]

    def preprocess_m(self, root, name_path, list_path): 
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        dir_path = osp.join(root, name_path)
        pids_dict = {}
        ret = []
        cam_2_imgs = defaultdict(int)

        for img_idx, img_info in enumerate(lines):
            fname = img_info.split()[0]
            # fname = fname.split('/')[1]
            info = fname.split('_')
            # pid = info[0]
            pid = img_info.split()[1]
            cam = info[2]
            sequenceid = info[3]
            frame = info[4]
            pid = int(pid)  # no need to relabel
            if pid not in pids_dict:
                pids_dict[pid] = pid
            # camid = int(cam.split('c')[-1]) - 1
            camid = int(cam) - 1
            frameid = int(frame)
            img_path = osp.join(dir_path, fname)
            ret.append((img_path, pid, camid, sequenceid, frameid))
            cam_2_imgs[camid] += 1

        if cam_2_imgs:
            cam_2_imgs = list(np.asarray(
                sorted(cam_2_imgs.items(), key=lambda e: e[0]))[:, 1])
            print(root, cam_2_imgs)
        return ret, len(pids_dict), cam_2_imgs

    def preprocess(self, root, name_path, relable=True):
        # pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        if osp.basename(root) == 'market':
            pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_(\d+)_')
        elif osp.basename(root) == 'DukeMTMC-reID':
            pattern = re.compile(r'([-\d]+)_c(\d)_f(\d+)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d)')
        pids_dict = {}
        ret = []
        cam_2_imgs = defaultdict(int)

        if 'cuhk03' in root:
            fpaths = sorted(glob(osp.join(root, name_path, '*.png')))
        else:
            fpaths = sorted(glob(osp.join(root, name_path, '*.jpg')))

        for fpath in fpaths:
            fname = osp.basename(fpath)
            if osp.basename(root) == 'market':
                pid, cam, sequenceid, frameid = map(int, pattern.search(fname).groups())
            elif osp.basename(root) == 'DukeMTMC-reID':
                pid, cam, frameid = map(int, pattern.search(fname).groups())
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            cam -= 1  # start from 0
            if pid == -1: continue
            if relable:
                # relable
                if pid not in pids_dict:
                    pids_dict[pid] = len(pids_dict)
                # for train
                cam_2_imgs[cam] += 1
            else:
                # not relabel
                if pid not in pids_dict:
                    pids_dict[pid] = pid

            pid = pids_dict[pid]
            if osp.basename(root) == 'market':
                ret.append([fpath, pid, cam, sequenceid, frameid])
            elif osp.basename(root) == 'DukeMTMC-reID':
                ret.append([fpath, pid, cam, 1, frameid])
            else:
                ret.append([fpath, pid, cam, 1, 1])

        if cam_2_imgs:
            cam_2_imgs = list(np.asarray(
                sorted(cam_2_imgs.items(), key=lambda e: e[0]))[:, 1])
            print(root, cam_2_imgs)
        return ret, len(pids_dict), cam_2_imgs
    
    def preprocess_msmt(self, root, name_path, list_path, fake, relable=True):

        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_ids = set()
        dir_path = osp.join(root, name_path)
        cam_2_imgs = defaultdict(list)
        pids_dict = {}
        ret = []

        for img_idx, img_info in enumerate(lines):
            fname = img_info.split()[0]
            # fname = fname.split('/')[1]
            info = fname.split('_')
            # pid = info[0]
            pid = img_info.split()[1]
            cam = info[2]
            sequenceid = info[3]
            frame = info[4]
            pid = int(pid)  # no need to relabel
            # camid = int(cam.split('c')[-1]) - 1
            camid = int(cam) - 1
            frameid = int(frame)
            img_path = osp.join(dir_path, fname)
            dataset.append((img_path, pid, camid,sequenceid,frameid))
            pid_container.add(pid)
            cam_ids.add(camid)
            cam_2_imgs[camid].append([img_path, pid, camid, sequenceid, frameid])
        num_imgs = len(dataset)
        num_pids = len(pid_container)

        # index start from camera 0-c
        if fake:
            # fake pid
            for c in sorted(cam_2_imgs.keys()):
                if c not in pids_dict:
                    pids_dict[c] = {}
                for i, (fpath, pid, cam, sequenceid, frameid) in enumerate(cam_2_imgs[c]):
                    # relable
                    pids_dict[cam][i] = len(pids_dict[cam]) # within camera, pid start from 0
                    pid = pids_dict[cam][i]
                    ret.append([fpath, pid, cam])
            cam_2_imgs = [len(pids_dict[e]) for e in sorted(pids_dict.keys())]
            print(root, cam_2_imgs)
        else:
            for c in sorted(cam_2_imgs.keys()):
                for i, (fpath, pid, cam, sequenceid, frameid) in enumerate(cam_2_imgs[c]):
                    if pid not in pids_dict:
                        pids_dict[pid] = len(pids_dict)
                    pid = pids_dict[pid]
                    ret.append([fpath, pid, cam, sequenceid, frameid])
        return ret, len(pids_dict), cam_2_imgs
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset, num_pids, num_imgs, len(cam_ids)

    def preprocess_cam(self, root, name_path, fake, relable=True):
        # pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        if osp.basename(root) == 'market':
            pattern = re.compile(r'([-\d]+)_c(\d)s(\d)_(\d+)_')
        elif osp.basename(root) == 'DukeMTMC-reID':
            pattern = re.compile(r'([-\d]+)_c(\d)_f(\d+)')
        else:
            pattern = re.compile(r'([-\d]+)_c(\d)')
        pids_dict = {}
        ret = []
        cam_2_imgs = defaultdict(list)

        if 'cuhk03' in root:
            fpaths = sorted(glob(osp.join(root, name_path, '*.png')))
        else:
            fpaths = sorted(glob(osp.join(root, name_path, '*.jpg')))

        for i, fpath in enumerate(fpaths):
            fname = osp.basename(fpath)
            # print(fpath)
            if osp.basename(root) == 'market':
                pid, cam, sequenceid, frameid = map(int, pattern.search(fname).groups())
            elif osp.basename(root) == 'DukeMTMC-reID':
                pid, cam, frameid = map(int, pattern.search(fname).groups())
            else:
                pid, cam = map(int, pattern.search(fname).groups())
            cam -= 1  # start from 0
            if pid == -1: continue
            if osp.basename(root) == 'market':
                cam_2_imgs[cam].append([fpath, pid, cam,sequenceid,frameid])
            elif osp.basename(root) == 'DukeMTMC-reID':
                cam_2_imgs[cam].append([fpath, pid, cam,1,frameid])
            else:
                cam_2_imgs[cam].append([fpath, pid, cam, 1, 1])

        # index start from camera 0-c
        if fake:
            # fake pid
            for c in sorted(cam_2_imgs.keys()):
                if c not in pids_dict:
                    pids_dict[c] = {}
                for i, (fpath, pid, cam,sequenceid,frameid) in enumerate(cam_2_imgs[c]):
                    # relable
                    pids_dict[cam][i] = len(pids_dict[cam]) # within camera, pid start from 0
                    pid = pids_dict[cam][i]
                    ret.append([fpath, pid, cam])
            cam_2_imgs = [len(pids_dict[e]) for e in sorted(pids_dict.keys())]
            print(root, cam_2_imgs)
        else:
            for c in sorted(cam_2_imgs.keys()):
                for i, (fpath, pid, cam,sequenceid,frameid) in enumerate(cam_2_imgs[c]):
                    if pid not in pids_dict:
                        pids_dict[pid] = len(pids_dict)
                    pid = pids_dict[pid]
                    ret.append([fpath, pid, cam, sequenceid, frameid])
        return ret, len(pids_dict), cam_2_imgs

    def load_dataset(self):
        # train dataset order is not different !!!!
        if self.target == 'msmt17':
            self.target_train_real, self.t_train_pids, _ = self.preprocess_msmt(self.target_root, self.train_path, self.list_train_path, fake=False, relable=True)
            self.target_train_fake, _, self.t_train_cam_2_imgs = self.preprocess_msmt(self.target_root, self.train_path, self.list_train_path, fake=True, relable=True)
            self.source_train, self.s_train_pids, self.s_train_cam_2_imgs = self.preprocess(self.source_root, 'bounding_box_train', True)
            self.target_gallery, self.t_gallery_pids, _ = self.preprocess_m(self.target_root, self.gallery_path, self.list_gallery_path)
            self.source_gallery, self.s_gallery_pids, _ = self.preprocess(self.source_root, 'bounding_box_test', False)
            self.target_query, self.t_query_pids, _ = self.preprocess_m(self.target_root, self.query_path, self.list_query_path)
            self.source_query, self.s_query_pids, _ = self.preprocess(self.source_root, 'query', False)
        else:
            self.target_train_real, self.t_train_pids, _ = self.preprocess_cam(self.target_root, self.train_path, fake=False, relable=True)
            self.target_train_fake, _, self.t_train_cam_2_imgs = self.preprocess_cam(self.target_root, self.train_path, fake=True, relable=True)
            self.source_train, self.s_train_pids, self.s_train_cam_2_imgs = self.preprocess(self.source_root, self.train_path, True)
            self.target_gallery, self.t_gallery_pids, _ = self.preprocess(self.target_root, self.gallery_path, False)
            self.source_gallery, self.s_gallery_pids, _ = self.preprocess(self.source_root, self.gallery_path, False)
            self.target_query, self.t_query_pids, _ = self.preprocess(self.target_root, self.query_path, False)
            self.source_query, self.s_query_pids, _ = self.preprocess(self.source_root, self.query_path, False)

        print("  subset     | # ids | # imgs")
        print('  --------------------------')
        print("{} train   | {:5d} | {:8d}".format(self.source, self.s_train_pids, len(self.source_train)))
        print("{} query   | {:5d} | {:8d}".format(self.source, self.s_query_pids, len(self.source_query)))
        print("{} gallery | {:5d} | {:8d}".format(self.source, self.s_gallery_pids, len(self.source_gallery)))
        print('  --------------------------')
        print("{} train   | {:5d} | {:8d}".format(self.target, self.t_train_pids, len(self.target_train_fake)))
        print("{} query   | {:5d} | {:8d}".format(self.target, self.t_query_pids, len(self.target_query)))
        print("{} gallery | {:5d} | {:8d}".format(self.target, self.t_gallery_pids, len(self.target_gallery)))


