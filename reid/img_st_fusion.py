import argparse
import os
import os.path as osp
import sys
sys.path.append(os.getcwd())
from reid.ctrl.img_st_fusion import init_strict_img_st_fusion
from reid.profile import fusion_param
from reid.profile.fusion_param import get_fusion_param
from reid.utils.file_helper import safe_link


# def arg_parse():
#     parser = argparse.ArgumentParser(description='Training')
#     parser.add_argument('--data_folder_path', default='duke_market-train', type=str, help='fusion data output dir')
#     parser.add_argument('--cv_num', default=0, type=int, help='0...9, for grid cross validation')
#     parser.add_argument('--ep', default=0, type=float, help='[0,1], error of position sample')
#     parser.add_argument('-ds', '--source_dataset', type=str, default='DukeMTMC-reID')
#     parser.add_argument('-dt', '--target_dataset', type=str, default='market')
#     working_dir = osp.dirname(osp.abspath(__file__))
#     parser.add_argument('--data_dir', type=str, metavar='PATH',
#                         default='/hdd/sdb/zyb/TFusion/SpCL/data')
#     parser.add_argument('--useful_cnt', default=10, type=int, help='')
#     parser.add_argument('--interval', default=25, type=int, help='')
#     parser.add_argument('--en', default=0, type=float, help='[0,1], error of negative sample')
#     parser.add_argument('--window_interval', default=500, type=int, help='')
#     parser.add_argument('--filter_interval', default=80000, type=int, help='')
#     parser.add_argument('--vision_folder_path', default='/home/zyb/projects/h-go/eval/duke2market-train', type=str, help='')


#     opt = parser.parse_args()
#     return opt

def build_param(opt):
    fusion_param.ctrl_msg['data_folder_path'] = opt.source + '_' + opt.target + '-train'
    # fusion_param.ctrl_msg['cv_num'] = opt.cv_num
    # fusion_param.ctrl_msg['ep'] = opt.ep
    # fusion_param.ctrl_msg['en'] = opt.en
    # fusion_param.ctrl_msg['window_interval'] = opt.window_interval
    # fusion_param.ctrl_msg['filter_interval'] = opt.filter_interval
    working_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    source_dir = os.path.join(working_dir, opt.eval_dir)
    param = get_fusion_param()
    safe_link(source_dir + '-train/pid.txt', param['renew_pid_path'])
    # safe_link(opt.vision_folder_path + '/score.txt', param['renew_ac_path'])
    # fusion_param.ctrl_msg['data_folder_path'] = opt.data_folder_path.replace('train', 'test')
    # param = get_fusion_param()
    # safe_link(opt.vision_folder_path.replace('train', 'test') + '/pid.txt', param['renew_pid_path'])
    # safe_link(opt.vision_folder_path.replace('train', 'test') + '/score.txt', param['renew_ac_path'])
    # fusion_param.ctrl_msg['data_folder_path'] = opt.data_folder_path

def stmain(sim, opt):
    # opt = arg_parse()
    build_param(opt)
    return init_strict_img_st_fusion(opt, sim)

if __name__ == '__main__':
    stmain()