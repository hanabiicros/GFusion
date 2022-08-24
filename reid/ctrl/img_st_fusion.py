#coding=utf-8
import shutil

import os

from reid.profile.fusion_param import get_fusion_param, ctrl_msg
from reid.train.st_estim import get_predict_delta_tracks, prepare_rand_folder, prepare_diff_folder
from reid.train.st_filter import fusion_st_img_ranker, fusion_st_gallery_ranker, simple_fusion_st_img_ranker, \
    simple_fusion_st_gallery_ranker

# need to run on src directory
from reid.utils.file_helper import safe_remove, safe_mkdir

def test_fusion(fusion_param,opt,ep=0, en=0.01):
    # copy sort pickle
    safe_remove(fusion_param['distribution_pickle_path'])
    try:
        # 直接使用训练集的时空模型
        shutil.copy(fusion_param['src_distribution_pickle_path'], fusion_param['distribution_pickle_path'])
        print('copy train track distribute pickle done')
    except shutil.Error:
        print('pickle ready')
    # merge visual probability and track distribution probability
    if fusion_param['gt_fusion']:
        simple_fusion_st_gallery_ranker(fusion_param,dataset=opt.target_dataset ,data_path= opt.data_dir,interval=opt.interval)
    else:
        fusion_st_gallery_ranker(fusion_param,dataset=opt.target_dataset ,data_path= opt.data_dir,interval=opt.interval)
    # evaluate



def train_fusion(fusion_param, opt, sim, ep=0, en=0):
    # 这里不需要再做一次时空模型建立
    # get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    # store_sorted_deltas(fusion_param)
    if fusion_param['gt_fusion']:
        simple_fusion_st_img_ranker(fusion_param)
    else:
        return fusion_st_img_ranker(fusion_param, source = opt.source, target=opt.target, data_path= opt.data_dir, interval=opt.interval, sim = sim)


def init_strict_img_st_fusion(opt, sim):
    # 全局调度入口，会同时做训练集和测试集上的融合与评分
    fusion_param = get_fusion_param()
    safe_mkdir('data/' + ctrl_msg['data_folder_path'])
    get_predict_delta_tracks(fusion_param, source = opt.source, target=opt.target, data_path= opt.data_dir, useful_predict_limit = opt.useful_cnt, use_real_st=fusion_param['gt_fusion'])

    # only get rand model for train dataset
    # prepare_rand_folder(fusion_param)
    # # prepare_diff_folder(fusion_param)

    # ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    # fusion_param = get_fusion_param()
    # # # 生成随机时空点的时空模型
    # get_predict_delta_tracks(fusion_param, source = opt.source_dataset, target=opt.target_dataset, data_path= opt.data_dir, useful_predict_limit = opt.useful_cnt, random=True)

    # # #ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'].replace('rand', 'diff')
    # # #fusion_param = get_fusion_param()
    # # #get_predict_delta_tracks(fusion_param, diff_person=True)

    # # 改回非随机的train目录
    # ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]

    # # has prepared more accurate ep, en
    print('fusion on training dataset')
    return iter_strict_img_st_fusion(on_test=False, opt=opt, sim = sim)
    # # # 改成测试目录
    # print('fusion on test dataset')
    # ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-4] + 'est'
    # safe_mkdir('data/' + ctrl_msg['data_folder_path'])
    # iter_strict_img_st_fusion(on_test=True,opt=opt)


# def init_strict_img_st_fusion():
#     # 全局调度入口，会同时做训练集和测试集上的融合与评分
#     fusion_param = get_fusion_param()
#     print('init predict tracks into different class files')
#     # pick predict tracks into different class file
#     get_predict_tracks(fusion_param)
#     # get distribution sorted list for probability compute
#     store_sorted_deltas(fusion_param)
#
#     # # only get rand model for train dataset
#     print('generate random predict')
#     write_rand_pid(fusion_param)
#     ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
#     fusion_param = get_fusion_param()
#     # 生成随机时空点的时空模型
#     gen_rand_st_model(fusion_param)
#
#     # 改回非随机的train目录
#     ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]
#
#     # has prepared more accurate ep, en
#     print('fusion on training dataset')
#     iter_strict_img_st_fusion(on_test=False)
#     # 改成测试目录
#     print('fusion on test dataset')
#     ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-4] + 'est'
#     iter_strict_img_st_fusion(on_test=True)


def iter_strict_img_st_fusion(on_test=False, opt=None, sim = None):
    """
    call after img classifier update, train with new vision score and ep en
    :param on_test:
    :return:
    """
    fusion_param = get_fusion_param()
    # ep, en = get_shot_rate()
    if on_test:
        test_fusion(fusion_param,opt=opt)
    else:
        return train_fusion(fusion_param, opt = opt, sim = sim)
        # update_epen(fusion_param, True)


if __name__ == '__main__':
    # for i in range(5):
    #     for j in range(5):
    #         ctrl_msg['window_interval'] = (i+1) * 200
    #         ctrl_msg['filter_interval'] = (j+1) * 40000
    #         ctrl_msg['data_folder_path'] = 'market_dukequerytail-train'
    #         fusion_param = get_fusion_param()
    #         init_strict_img_st_fusion()
    #         ctrl_msg['data_folder_path'] = 'market_dukequerytail-test'
    #         fusion_param = get_fusion_param()
    #         os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    #         os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #                   + 'dukequerytail /home/cwh/coding/TrackViz/' + fusion_param['eval_fusion_path'])
    #
    #
    # src = 'grid'
    # dst = 'market'
    # ctrl_msg['data_folder_path'] = src + '_'+ dst +'-train'
    # init_strict_img_st_fusion()
    # ctrl_msg['data_folder_path'] = src + '_'+ dst +'-test'
    # fusion_param = get_fusion_param()
    # python_path = '/home/cwh/anaconda3/bin/python'
    # eval_sh_path = '/home/cwh/coding/taudl_pyt/baseline/eval_on_result.py'
    # target_dataset_path = '/home/cwh/coding/dataset/' + dst
    # pid_path = '/home/cwh/coding/TrackViz/' + fusion_param['eval_fusion_path']
    # log_path = src + '_'+ dst +'sense_eval.log'
    # os.system(
    #     'export PYTHONPATH=/home/cwh/coding/taudl_pyt; %s %s --target_dataset_path %s --pid_path %s --result_path %s ' % (
    #     python_path, eval_sh_path, target_dataset_path, pid_path, log_path))
    for i in range(0, 4):
        for j in range(0, 4 - i):
            ctrl_msg['ep'] = i * 0.25
            ctrl_msg['en'] = j * 0.25
            ctrl_msg['data_folder_path'] = 'grid_market-train'
            init_strict_img_st_fusion()
            ctrl_msg['data_folder_path'] = 'grid_market-test'
            fusion_param = get_fusion_param()
            os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
            # PYTHON eval_on_result.py --target_dataset_path $data_dir --pid_path $pid_path --result_path $log_path
            python_path = '/home/cwh/anaconda3/bin/python'
            eval_sh_path = '/home/cwh/coding/taudl_pyt/baseline/eval_on_result.py'
            target_dataset_path = '/home/cwh/coding/dataset/market'
            pid_path = '/home/cwh/coding/TrackViz/' + fusion_param['eval_fusion_path']
            log_path = 'grid2marketsense_eval.log'
            os.system('export PYTHONPATH=/home/cwh/coding/taudl_pyt; %s %s --target_dataset_path %s --pid_path %s --result_path %s ' % (python_path, eval_sh_path, target_dataset_path, pid_path, log_path))
    #
    # for cv_num in range(10):
    #     for i in range(0, 4):
    #         for j in range(0, 4 - i):
    #             ctrl_msg['ep'] = i * 0.25
    #             ctrl_msg['en'] = j * 0.25
    #             ctrl_msg['data_folder_path'] = 'market_grid-cv%d-train' % cv_num
    #             fusion_param = get_fusion_param()
    #             init_strict_img_st_fusion()
    #             ctrl_msg['data_folder_path'] = 'market_grid-cv%d-test' % cv_num
    #             fusion_param = get_fusion_param()
    #             os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    #             os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #                       + ('grid-cv%d' % cv_num) + ' ' + fusion_param['eval_fusion_path'])

    # ctrl_msg['ep'] = 0.25
    # ctrl_msg['en'] = 0.5
    # ctrl_msg['data_folder_path'] = 'duke_market-train'
    # ctrl_msg['data_folder_path'] = 'market_grid-cv-1-train'
    # fusion_param = get_fusion_param()
    # init_strict_img_st_fusion()
    # ctrl_msg['data_folder_path'] = 'grid_market-test'
    # fusion_param = get_fusion_param()
    # os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #           + 'market' + ' ' + fusion_param['eval_fusion_path'])
    # ctrl_msg['ep'] = 0.5
    # ctrl_msg['en'] = 0.25
    # ctrl_msg['data_folder_path'] = 'grid_market-train'
    # fusion_param = get_fusion_param()
    # init_strict_img_st_fusion()
    # ctrl_msg['data_folder_path'] = 'grid_market-test'
    # fusion_param = get_fusion_param()
    # os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #           + 'market' + ' ' + fusion_param['eval_fusion_path'])

    # delta_range, raw_probs, rand_probs, over_probs = fusion_curve(fusion_param)
    # viz_fusion_curve(delta_range, [raw_probs, rand_probs, over_probs])

    # viz smooth dist
    # viz_market_distribution(fusion_param)
