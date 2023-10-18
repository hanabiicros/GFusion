from __future__ import print_function, absolute_import
import argparse
from email.policy import strict
import gc
import os.path as osp
import networkx as nx
import argparse
import sys
import time
import os
import pyhocon
import torch.nn as nn
import scipy.sparse as sp
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# from reid.models.walkbased import DeepWalk, WalkBasedEmbedding, Walker, NegativeSampling, NodeEmbedding
from reid.ge import DeepWalk,Node2Vec
from reid.evaluation.ranking import cmc, mean_ap, mean_ap2, map_cmc
from reid.lib.logging import Logger
from reid.lib.serialization import load_checkpoint, to_torch
from reid.utils.serialization import save_checkpoint
from reid.lib.lr_scheduler import WarmupMultiStepLR
from reid.dataloaders.loader import *
import reid.dataloaders.transforms as T
from reid.dataloaders.dataset import Person
from reid import models
from reid.models.cls_layer import *
from reid.models.backbone import resnet
from reid.evaluation.evaluator import Evaluator
from reid.trainer.trainer import Trainer
from reid.detach_enhancement.hetero_graph import HG
from reid.lib.serialization import copy_state_dict
from reid.img_st_fusion import stmain
from reid.graphsage.dataCenter import *
from reid.graphsage.models import *
from reid.graphsage.utils import *
from reid.models.dsbn import convert_dsbn


def get_data(opt):
    dataset = Person(opt.data_dir, opt.target, opt.source)
    t_pids = dataset.t_train_pids
    s_pids = dataset.s_train_pids
    s_train = dataset.source_train
    t_train_fake = dataset.target_train_fake
    t_train_real = dataset.target_train_real

    s_query = dataset.source_query
    t_query = dataset.target_query
    s_gallery = dataset.source_gallery
    t_gallery = dataset.target_gallery
    t_cam_2_imgs = dataset.t_train_cam_2_imgs
    s_cam_2_imgs = dataset.s_train_cam_2_imgs

    height, width = opt.height, opt.width
    batch_size, workers = opt.batch_size, opt.workers

    if opt.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer
    ])

    test_graph_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ColorJitter(brightness=opt.brightness, saturation=opt.saturation, hue=opt.hue),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    t_train_loader = DataLoader(
        dataset=CamStylePreprocessor(t_train_fake, num_cam=len(t_cam_2_imgs),
                                     transform=train_transformer, use_camstyle=opt.use_camstyle), # todo next try
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    t_traintest_loader = DataLoader(
        dataset=Preprocessor(t_train_real, test_graph_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )

    t_query_loader = DataLoader(
        dataset=Preprocessor(t_query, test_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    t_gallery_loader = DataLoader(
        dataset=Preprocessor(t_gallery, test_transformer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False
    )
    # s_train_loader = DataLoader(
    #     dataset=Preprocessor(s_train, train_transformer),
    #     batch_size=batch_size,
    #     num_workers=workers,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True
    # )
    # s_traintest_loader = DataLoader(
    #     dataset=Preprocessor(s_train, test_transformer),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=workers,
    #     pin_memory=False,
    #     drop_last=False
    # )
    # s_query_loader = DataLoader(
    #     dataset=Preprocessor(s_query, test_transformer),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=workers,
    #     pin_memory=False,
    #     drop_last=False
    # )
    # s_gallery_loader = DataLoader(
    #     dataset=Preprocessor(s_gallery, test_transformer),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=workers,
    #     pin_memory=False,
    #     drop_last=False
    # )

    return t_pids, s_pids, t_train_loader, t_train_real,\
           t_query_loader, t_gallery_loader,\
           t_query, t_gallery, [],\
           [], [], \
           s_query, s_gallery, \
           t_traintest_loader, [], \
           t_train_fake, s_train, t_cam_2_imgs, s_cam_2_imgs

# class WalkEmbedding(WalkBasedEmbedding):
#     def __init__(self, graph, dimension, iterations, walker, sampler, model):
#         self.graph = graph
#         self.dimension = dimension
#         self.iterations = iterations

#         self.walker = walker
#         self.sampler = sampler
#         self.model = model

#         self.train()

#     def train(self):
#         sequences = self.walker.walk(self.graph)  # todo modify LINE

#         for it in range(self.iterations):
#             for samples in self.sampler.sample(sequences):
#                 self.model.feed(*samples)
#             self.model.lr_decay()
#             print("===> epoch %d" % it)
#             # embedding = self.model.get_embeddings()
#             # test_gps2img(categories, labels, cams, nodes, embedding, is_save=False)
#             # test_img2img(categories, labels, cams, nodes, embedding, is_save=False)

#         return self

# def DeepWalk(graph, *,
#         # dimension = 128,
#         dimension = 128,
#         num_walks = 10,
#         # walk_length = 80,
#         # window_size = 10,
#         walk_length = 250, # 150
#         window_size = 20,
#         iterations = 3,
#         neg_ratio = 5,
#         learning_rate = 0.001,
#         batch_size = 10000,
#         down_sample_threshold = 1e-3,
#         weighted_walk=False):
#     return WalkEmbedding(graph,
#                          dimension,
#                          iterations,
#                          walker = Walker(num_walks, walk_length, weighted=weighted_walk),
#                          sampler = NegativeSampling(window_size, batch_size, neg_ratio=neg_ratio, down_sampling=down_sample_threshold),
#                          model = NodeEmbedding(graph.number_of_nodes(), dimension, learning_rate),
#                          )

def compute_euclidean_distance(
    features, others=None, cuda=False,
):

    if others is None:
        if cuda:
            features = features.cuda()

        n = features.size(0)
        x = features.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())

    else:
        if cuda:
            features = features.cuda()
            others = others.cuda()

        m, n = features.size(0), others.size(0)
        # print(m, n)
        dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist_m.addmm_(features, others.t(), beta=1, alpha=-2)

    return dist_m.cpu().numpy()

def symm_kneighbors(sim_matrix, knn, unifyLabel=None):
    k_sim = np.zeros_like(sim_matrix, dtype=np.float32)
    argpart = np.argpartition(-sim_matrix, knn)  # big move before knn，将前knn个相似度大的数排在前面并获得相应下标

    row_index = np.arange(sim_matrix.shape[0])[:, None]
    if unifyLabel:
        k_sim[row_index, argpart[:, 0:knn]] = unifyLabel
    else:
        k_sim[row_index, argpart[:, 0:knn]] = sim_matrix[row_index, argpart[:, 0:knn]]

    k_sim[row_index, row_index] = 1
    print(k_sim[:3,argpart[:3, 0:knn]])

    # b = np.sum(k_sim, axis = 1)
    # for i,_ in enumerate(k_sim):
    #     k_sim[i] /= b[i]
    # d = sp.diags(np.power(np.array(k_sim.sum(1)), -0.5).flatten(), 0)
    # k_norm = k_sim.dot(d).transpose().dot(d)

    # print(np.sum(k_norm, axis=1))
    # print(k_sim[0:3,:])
    k_sim = torch.from_numpy(k_sim)
    return k_sim

# def non_symm_kneighbors(sim_matrix, knn, unifyLabel=None):
#     k_sim = np.zeros_like((len(sim_matrix),len(sim_matrix[0])), dtype=np.float32)
#     argpart = np.argpartition(-sim_matrix, knn)  # big move before knn，将前knn个相似度大的数排在前面并获得相应下标

#     row_index = np.arange(sim_matrix.shape[0])[:, None]
#     if unifyLabel:
#         k_sim[row_index, argpart[:, 0:knn]] = unifyLabel
#     else:
#         k_sim[row_index, argpart[:, 0:knn]] = sim_matrix[row_index, argpart[:, 0:knn]]
    
#     b = np.sum(k_sim, axis = 1)
#     for i,_ in enumerate(k_sim):
#         k_sim[i] /= b[i]
#     k_norm = torch.from_numpy(k_norm)
#     return k_norm

def qg_kneighbors(sim_matrix, st_matrix, knn, cams, isVision=True):
    len_matrix = len(sim_matrix[0]) + len(sim_matrix)
    k_sim = np.zeros((len(sim_matrix),len_matrix), dtype=np.float32)
    argpart = np.argpartition(-st_matrix, knn)  # big move before knn，将前knn个相似度大的数排在前面并获得相应下标
    # st_argpart = np.argpartition(-st_matrix, 1)
    
    row_index = np.arange(len(sim_matrix))[:, None]

    k_sim[row_index, argpart[:, 0:knn]] = st_matrix[row_index, argpart[:, 0:knn]]
    # k_sim[row_index, st_argpart[:, 0]] = sim_matrix[row_index, st_argpart[:, 0]]

    # for i in range(len(k_sim)):
    #     index = np.where(k_sim[i] != 0.)
    #     weights = k_sim[i][index]
    #     cd_c = cams[index]
    #     tag_c_set = set(cd_c)
    #     for c in tag_c_set:
    #         c_index = np.where(cd_c == c)
    #         w = weights[c_index]

    #         w = len(w) / len(cd_c) * w / np.sum(w)  
    #         k_sim[i][index[0][c_index]] = w
    
    if isVision:
        for i in range(len(k_sim)):
            k_sim[i][len(sim_matrix[0])+i] = 1

    b = np.sum(k_sim, axis = 1)
    for i,_ in enumerate(k_sim):
        k_sim[i] /= b[i]

    k_sim = torch.from_numpy(k_sim)
    return k_sim

def qg_diff_kneighbors(sim_matrix, top_same_indexs, top_diff_indexs, isVision=True):
    len_matrix = len(sim_matrix[0]) + len(sim_matrix)
    k_sim = np.zeros((len(sim_matrix),len_matrix), dtype=np.float32)
    
    row_index = np.arange(len(sim_matrix))[:, None]

    k_sim[row_index, top_same_indexs[:, :]] = sim_matrix[row_index, top_same_indexs[:, :]]
    k_sim[row_index, top_diff_indexs[:, :]] = sim_matrix[row_index, top_diff_indexs[:, :]]
    
    if isVision:
        for i in range(len(k_sim)):
            k_sim[i][len(sim_matrix[0])+i] = 1

    b = np.sum(k_sim, axis = 1)
    for i,_ in enumerate(k_sim):
        k_sim[i] /= b[i]

    k_sim = torch.from_numpy(k_sim)
    return k_sim

def main(opt):
    start = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    if opt.seed != -1:
        SEED = opt.seed
        np.random.seed(SEED)
        # this set the model state of all parameters
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cudnn.deterministic = True


    # For fast training.
    cudnn.benchmark = True
    # torch.cuda.set_device(opt.gpu_ids)
    # Redirect print ot both console and log file
    # if not opt.evaluate:
    sys.stdout = Logger(osp.join(opt.logs_dir, 'log.txt'))
    print(time.ctime(), 'logs_dir=', opt.logs_dir)

    # Print logs
    print(opt)

    # Create data loaders
    t_pids, s_pids, t_train_loader, t_train_real, \
    t_query_loader, t_gallery_loader, \
    t_query, t_gallery, s_train_loader, \
    s_query_loader, s_gallery_loader, \
    s_query, s_gallery, \
    t_traintest_loader, s_traintest_loader, \
    t_train, s_train, t_cam_2_imgs, s_cam_2_imgs,\
    = get_data(opt)

    # Create model
    if 'resnet' in opt.arch:
        print('use resnet')
        backbone = models.create(opt.arch, num_features=opt.features, dropout=opt.dropout, num_classes=0)
    elif 'vit' in opt.arch:
        print('use vit')
        backbone = models.create(opt.arch,img_size=(opt.height,opt.width),drop_path_rate=opt.drop_path_rate, pretrained_path = opt.pre_train,hw_ratio=opt.hw_ratio, conv_stem=opt.conv_stem)
    elif 'ICE' in opt.arch:
        print('use ICE')
        backbone = resnet.ICEResNet(num_features=args.features, dropout=args.dropout, num_classes=0)
    elif 'SpCL' in opt.arch:
        print('use SpCL')
        backbone = resnet.ICEResNet(num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
        convert_dsbn(backbone)
    elif 'CAP' in opt.arch:
        print('use CAP')
        backbone = resnet.MemoryBankModel(out_dim=2048)
    elif 'MMT' in opt.arch:
        print('use MMT')
        backbone = resnet.MMTResNet(50, num_features=args.features, dropout=args.dropout, num_classes=0)

    old_cls_tgt = OldClsTarget(
                            len(t_train),
                            opt.features,
                            t_cam_2_imgs,
                            opt.alpha,
                            opt.tao,
                            opt.switch,
                            opt.arch)
    # hyper parameter
    old_cls_tgt.use_single = opt.use_single

    # Set model
    backbone.cuda()
    backbone = nn.DataParallel(backbone) #若使用model.state_dict()保存模型，加载模型时需要先把网络弄成分布式训练
    old_cls_tgt = old_cls_tgt.cuda()

    # Load from checkpoint
    start_epoch = 0
    best_map = 0
    
    # pre_path = osp.join('./logs/pretrain',args.arch)
    # if opt.source == 'market' and opt.target == 'duke':
    #     checkpoint = load_checkpoint(osp.join(pre_path, 'market2duke/checkpoint.pth.tar'))
    #     copy_state_dict(checkpoint['backbone_dict'], backbone)
    # elif opt.source == 'DukeMTMC-reID' and opt.target == 'market':
    #     checkpoint = load_checkpoint(osp.join(pre_path, 'DukeMTMC-reID2market/checkpoint.pth.tar'))
    #     # copy_state_dict(checkpoint['backbone_dict'], backbone)
    #     backbone.load_state_dict(checkpoint['backbone_dict'], strict=False)
    # elif opt.source == 'duke' and opt.target == 'msmt17':
    #     checkpoint = load_checkpoint(osp.join(pre_path, 'duke2msmt17/checkpoint.pth.tar'))
    #     copy_state_dict(checkpoint['backbone_dict'], backbone)
    # elif opt.source == 'market' and opt.target == 'msmt17':
    #     checkpoint = load_checkpoint(osp.join(pre_path, 'market2msmt17/checkpoint.pth.tar'))
    #     copy_state_dict(checkpoint['backbone_dict'], backbone)
    
    # if args.pre_train:
    #     checkpoint = load_checkpoint(args.pre_train)
    #     backbone.load_state_dict(checkpoint['backbone_dict'], strict=False)
    
    # if args.resume:
    #     checkpoint = load_checkpoint(args.resume)
    #     backbone.load_state_dict(checkpoint['backbone_dict'], strict=False)
    #     if 'epoch' in checkpoint.keys():
    #         start_epoch = checkpoint['epoch']
    #         print('checkpoint loaded, start epoch is : {}'.format(start_epoch))

    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        new_state_dict =  {}

        #MMT
        # for key, value in checkpoint['state_dict'].items():
        #     if 'base.0' in key:
        #         name = key.replace('base.0','base.conv1')
        #     elif 'base.1' in key:
        #         name = key.replace('base.1','base.bn1')
        #     elif 'base.3' in key:
        #         name = key.replace('base.3','base.layer1')
        #     elif 'base.4' in key:
        #         name = key.replace('base.4','base.layer2')
        #     elif 'base.5' in key:
        #         name = key.replace('base.5','base.layer3')
        #     elif 'base.6' in key:
        #         name = key.replace('base.6','base.layer4')
        #     else:
        #         name = key
        #     new_state_dict[name] = value

        # #CAP
        # for key,value in checkpoint.items():
        #     if 'embeding' in key:
        #         print('pretrained model has key= {}'.format(key))
        #     elif 'resnet_conv.0' in key:
        #         name = key.replace('resnet_conv.0', 'base.conv1')
        #     elif 'resnet_conv.1' in key:
        #         name = key.replace('resnet_conv.1', 'base.bn1')
        #     elif 'resnet_conv.3' in key:
        #         name = key.replace('resnet_conv.3', 'base.layer1')
        #     elif 'resnet_conv.4' in key:
        #         name = key.replace('resnet_conv.4', 'base.layer2')
        #     elif 'resnet_conv.5' in key:
        #         name = key.replace('resnet_conv.5', 'base.layer3')
        #     elif 'resnet_conv.6' in key:
        #         name = key.replace('resnet_conv.6', 'base.layer4')
        #     elif 'bottleneck' in key:
        #         name = key.replace('bottleneck', 'feat_bn')
        #     else:
        #         name = key
        #     new_state_dict[name] = value

        # #ICE
        # for key, value in checkpoint['state_dict'].items():
        #     if 'base.0' in key:
        #         name = key.replace('base.0','base.conv1')
        #     elif 'base.1' in key:
        #         name = key.replace('base.1','base.bn1')
        #     elif 'base.4' in key:
        #         name = key.replace('base.4','base.layer1')
        #     elif 'base.5' in key:
        #         name = key.replace('base.5','base.layer2')
        #     elif 'base.6' in key:
        #         name = key.replace('base.6','base.layer3')
        #     elif 'base.7' in key:
        #         name = key.replace('base.7','base.layer4')
        #     else:
        #         name = key
        #     new_state_dict[name] = value

        # for key, value in checkpoint['state_dict_1'].items():
        #     if 'net.backbone' in key:
        #         name = key.replace('net.backbone','base')
        #     else:
        #         name = key
        #     new_state_dict[name] = value
        # backbone.load_state_dict(new_state_dict, strict=False)
        if 'resnet' in opt.arch or opt.evaluate:
            backbone.load_state_dict(checkpoint['backbone_dict'], strict=False)
            print("load backbone dict")
        elif 'CAP' in opt.arch:
            backbone.load_state_dict(checkpoint, strict=False)
            print("load checkpoint")
        else:
            backbone.load_state_dict(checkpoint['state_dict'], strict=False)
            print("load state dict")
    
    if not opt.evaluate and opt.target != 'msmt17':
        evaluator = Evaluator(backbone)   
        map,top1,_,_,_,_,_,_ = evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery, opt.kl, False)
        best_map = map+top1
    # Evaluator
    if opt.evaluate:
        print("test fusion")
        # evaluator = Evaluator(backbone)
        # indexs = evaluator.transfer(t_traintest_loader, t_train_real, opt.ks, opt.logs_dir)
        # return
        # indexs = np.genfromtxt(opt.logs_dir+"/indexs.txt", dtype=np.int32)
        evaluator = Evaluator(backbone)
        best_map, _, scores, gscores, qgindexs, ggindexs, query_features, gallery_features = evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery, opt.kl, False)
        indexs = evaluator.transfer(t_traintest_loader, t_train_real, opt.ks, opt.logs_dir)
        st_scores, gst_scores = stmain(opt, indexs, t_train_real, t_query, t_gallery, qgindexs=qgindexs, ggindexs=ggindexs, flag = False)

        # # 对时空相似性进行max归一化 没用
        # for i in range(len(st_scores)):
        #     # index = np.where((st_scores[i] != 0.) & (st_scores[i] != -1))
        #     max_i = max(st_scores[i])
        #     for j,score in enumerate(st_scores[i]):
        #         if score != 0 and score != -1 and score != -2:
        #             st_scores[i][j] /= max_i
        
        # for i in range(len(gst_scores)):
        #     # index = np.where((st_scores[i] != 0.) & (st_scores[i] != -1))
        #     max_i = max(gst_scores[i])
        #     for j,score in enumerate(gst_scores[i]):
        #         if score != 0 and score != -1 and score != -2:
        #             gst_scores[i][j] /= max_i
                    
        sum_scores = []
        gg = defaultdict()
        gg_positive = defaultdict(set)
        gg_negetive = defaultdict(set)
        adj_lists = defaultdict(set)

        # vision + st
        for i in range(len(gscores)):
            for index, j in enumerate(ggindexs[i]):
                if index < opt.ks or gst_scores[i][index] > 0:
                # if index < opt.ks:
                    if i != j:
                        adj_lists[i].add(j)
                        adj_lists[j].add(i)
                        gg[i,j] = gscores[i][index] 
                        gg[j,i] = gscores[i][index] 

            adj_lists[i].add(i)
            gg[i,i] = 1                

        query_ids = np.asarray([pid for _, pid, _,_,_ in t_query])
        gallery_ids = np.asarray([pid for _, pid, _ ,_,_ in t_gallery])
        query_cams = np.asarray([cam for _, _, cam,_,_ in t_query])
        gallery_cams = np.asarray([cam for _, _, cam,_,_ in t_gallery])
        
        
        ## graphsage

        # load config file
        config = pyhocon.ConfigFactory.parse_file(args.config)
        # load data
        ds = args.target
        dataCenter = DataCenter(config)

        # 构建gallery视觉图
        device = torch.device("cuda", 0)
        dataCenter.load_dataSet(ds, len_g=len(gscores), isVision=True, use_sparse=opt.use_sparse)
        del gscores
        del ggindexs
        # del gst_scores
        # del indexs
        gc.collect()
        torch.cuda.empty_cache()

        # gscores = getattr(dataCenter, ds+'_weight_lists').to(device)
        # gscores = getattr(dataCenter, ds+'_weight_lists')
        gallery_features = gallery_features.to(device)
        query_features = query_features.to(device)
        # gscores = gscores.to(device)
        # scores = scores.to(device)
        # st_scores = st_scores.to(device)

        graphSage1 = GraphSage(config['setting.num_layers'], gallery_features.size(1), config['setting.hidden_emb_size'], gallery_features, gg, adj_lists, device, gcn=True, agg_func=args.agg_func)
        # graphSage1 = nn.DataParallel(graphSage1 ,device_ids=[0])
        graphSage1.to(device)

        # num_labels = 0
        # classification = Classification(config['setting.hidden_emb_size'], num_labels)
        # classification.to(device)


        # unsupervised_loss = UnsupervisedLoss(adj_lists, getattr(dataCenter, ds+'_train'), gg_positive, gg_negetive, device)


        print('GraphSage with vision and spatial-temporal Net Unsupervised Learning')

        # for epoch in range(args.g_epochs):
        #     print('----------------------EPOCH %d-----------------------' % epoch)
        #     graphSage1, classification = apply_model(dataCenter, ds, graphSage1, classification, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method)

        
        vg_features = get_gnn_embeddings(graphSage1, len(adj_lists))
        vg_features = vg_features.cpu()
        vg_dir = os.path.join(opt.logs_dir, 'g_features.pth')
        torch.save(vg_features, vg_dir)

        # sum_scores = []
        for i in range(len(scores)):
            # sum_score = 0
            # indexs = np.argsort(-scores[i])[:5]
            for index, j in enumerate(qgindexs[i]):
                if index < opt.ks or (st_scores[i][index] > 0):
                # if index < opt.ks:
                    if i != j:
                        adj_lists[len(gallery_features) + i].add(j)
                        gg[len(gallery_features) + i,j] = scores[i][index] 
                            # sum_score += scores[i][index]
            adj_lists[len(gallery_features) + i].add(len(gallery_features) + i)
            gg[len(gallery_features) + i,len(gallery_features) + i] = 1
            # sum_score += 1
            # sum_scores.append(sum_score)

        graphSage1.raw_weight = gg
        features = torch.cat((gallery_features, query_features), dim=0)
        graphSage1.adj_lists = adj_lists
        graphSage1.raw_features = features
        

        vq_embeddings = get_query_embeddings(graphSage1, len(gallery_features), len(adj_lists))
        vq_embeddings = vq_embeddings.cpu()
        vq_dir = os.path.join(opt.logs_dir, 'q_embeddings.pth')
        
        torch.save(vq_embeddings, vq_dir)
        torch.cuda.empty_cache()

        q_embeddings =  vq_embeddings
        g_embeddings = vg_features
        distmat = compute_euclidean_distance(q_embeddings, g_embeddings)

        mAP, all_cmc = map_cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('CMC Scores')
        for k in (1, 5, 10):
            print('  top-{:<4}{:12.1%}'
                .format(k, all_cmc[k - 1]))

        return

    # best_map = 0
    # if opt.evaluate:
    #     # print('Test source {} : '.format(opt.source))
    #     # evaluator.evaluate(s_query_loader, s_gallery_loader, s_query, s_gallery)
    #     print('Test target {} : '.format(opt.target))
    #     evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery)
    #     return

    # Optimizer
    params = []
    for key, value in backbone.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)

    # if args.resume:
    #     optimizer.load_state_dict(checkpoint['optimizer_dict'], strict=False)
    
    scheduler = WarmupMultiStepLR(optimizer,
                                  [6],
                                  gamma=0.1,
                        warmup_factor=0.01,
                            warmup_iters=6)

    
    # Start training
    evaluator = Evaluator(backbone)
    t_train_loader = IterLoader(t_train_loader, length=len(t_train_loader))

    for epoch in range(start_epoch, opt.epochs):
        features, fpaths, labels, camids = \
            evaluator.extract_tgt_train_features(
                                t_traintest_loader)
        print("features len:{}".format(len(features)))


        graph = HG(opt.lamd, features, labels, camids)

        # hyper parameter
        graph.general_graph = opt.general_graph
        graph.homo_ap = opt.homo_ap

        start1 = time.time()

        # if opt.only_graph:
        #     graph_target = graph.only_graph(ks=opt.ks, kd=opt.kd, k2=opt.k2)
        # else:
        #     if opt.cos_sim:
        #         graph_target = graph.old_cos_propagation(ks=opt.ks, kd=opt.kd, k2=opt.k2)
        #     else:
        #         if opt.tracklet:
        #             graph_target = graph.old_tracklet_propagation(ks=opt.ks, kd=opt.kd, k2=opt.k2)
        #         else:
        graph_target = graph.old_delta_propagation(k2 = opt.k2, opt = opt, t_train = t_train_real)

        print('###graph time: ', time.time()-start1)
        # Trainer
        trainer = Trainer(backbone, old_cls_tgt)
        trainer.graph_target = graph_target

        t_train_loader.new_epoch()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        trainer.target_train(len(t_train_loader), optimizer,
                        t_train_loader, epoch, opt.arch)
        scheduler.step()

        if (epoch+1)%opt.eval_step==0:
            mAP ,top1 ,_ ,_,_,_,_,_ = evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery)
            is_best = (mAP  + top1 > best_map)
            best_map = max(mAP  + top1, best_map)

            save_checkpoint({
                'backbone_dict': backbone.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, is_best,fpath=osp.join(opt.logs_dir, 'checkpoint.pth.tar'))
        # print('\n * Finished epoch {:3d} \n'.
        #     format(epoch))

    print('Best mAP:{} : '.format(best_map))
    # evaluator.evaluate(t_query_loader, t_gallery_loader, t_query, t_gallery, eval_path=args.eval_dir)
    # evaluator.evaluate_fusion(t_query, t_gallery, args.fusion_dir)
    print(time.ctime(), ' Need Time:', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PGM")
    # gpu ids
    parser.add_argument('--gpu_ids', type=str, required=True)
    # random seed
    parser.add_argument('--seed', type=int, default=1)
    # source
    parser.add_argument('-s', '--source', type=str, default='DukeMTMC-reID',
                        choices=['market', 'DukeMTMC-reID', 'msmt17',])
    # target
    parser.add_argument('-t', '--target', type=str, default='market',
                        choices=['market', 'DukeMTMC-reID', 'msmt17', ])
    # imgs setting 
    parser.add_argument('-b', '--batch-size', type=int, default=64)   ## 64
    parser.add_argument('-j', '--workers', type=int, default=4)  ## 4
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    #vit
    parser.add_argument('--drop-path-rate', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=1)
    parser.add_argument('--self-norm', action="store_true")
    parser.add_argument('--conv-stem', action="store_true")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH', default='')
    parser.add_argument('-pp','--pre_train', type=str, metavar='PATH', default='')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help="evaluation only")
    parser.add_argument('--wo-st-train', action='store_true', default=False,
                        help="without spatial-temporal training")
    parser.add_argument('--wo-gaussian-sim', action='store_true', default=False,
                        help="without gaussian sim for training")
    parser.add_argument('--use-old-hgo', action='store_true', default=False,
                        help="without gaussian sim for training")
    parser.add_argument('--iters', type=int, default=202)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--epochs_decay', type=int, default=40)
    #st
    parser.add_argument('--interval', default=25, type=int, help='')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/hdd/sdb/zyb/TFusion/SpCL/data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/old/d2m'))
    # parser.add_argument('--eval_dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'eval','duke2market'))
    # parser.add_argument('--fusion_dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'data','DukeMTMC-reID_market-test'))
    parser.add_argument('--re', type=float, default=0.5)

    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--tao', type=float, default=0.05)
    parser.add_argument('--switch', type=int, default=1)
    parser.add_argument('--lamd', type=float, default=0.99)
    # parser.add_argument('--ratio', type=float, default=0.65)
    # parser.add_argument('--vision_weight', type=float, default=0.9)
    parser.add_argument('--ks', type=int, default=3)
    parser.add_argument('--kl', type=int, default=60)
    parser.add_argument('--k2', type=int, default=14)
    parser.add_argument('--use-camstyle', action='store_true', default=False)
    parser.add_argument('--use-sparse', action='store_true', default=False)
    parser.add_argument('--general-graph', action='store_true', default=False)
    parser.add_argument('--use-single', action='store_true', default=False)
    parser.add_argument('--only-graph', action='store_true', default=False)
    parser.add_argument('--cos-sim', action='store_true', default=False)
    parser.add_argument('--homo-ap', action='store_true', default=False)

    parser.add_argument('--brightness', type=float, default=0.2)
    parser.add_argument('--saturation', type=float, default=0.0)
    parser.add_argument('--hue', type=float, default=0.0)

    parser.add_argument("--tracklet", action="store_true", default=False)

    parser.add_argument('--describe', type=str)

    ##graphsage
    # parser.add_argument('--dataSet', type=str, default='market')
    parser.add_argument('--agg_func', type=str, default='WEIGHT')
    parser.add_argument('--g_epochs', type=int, default=1)
    parser.add_argument('--b_sz', type=int, default=20)
    parser.add_argument('--gcn', action='store_true')
    parser.add_argument('--learn_method', type=str, default='unsup')
    parser.add_argument('--unsup_loss', type=str, default='margin')
    parser.add_argument('--max_vali_f1', type=float, default=0)
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--config', type=str, default='/home/zyb/projects/h-go/reid/graphsage/experiments.conf')
    args = parser.parse_args()
    main(args)
