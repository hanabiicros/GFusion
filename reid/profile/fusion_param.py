from reid.utils.file_helper import safe_mkdir
import os
import sys
sys.path.append(os.getcwd())
ctrl_msg = {
    'data_folder_path': 'market_market-test',
    'cv_num': 0,
    'ep': 0,
    'en': 0,
    'window_interval': 500,
    'filter_interval': 80000
}

update_msg = {}


def get_fusion_param():
    origin_dict = {
        'gt_fusion': False,
        'renew_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_pid.log',
        'renew_ac_path': 'data/' + ctrl_msg['data_folder_path'] + '/renew_ac.log',
        'predict_pid_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_pid.log',
        'answer_path': 'data/' + ctrl_msg['data_folder_path'] + '/test_tracks.txt',

        'probe_path': '',
        'train_path': '',
        'gallery_path': '',

        'distribution_pickle_path': 'data/' + ctrl_msg['data_folder_path'] + '/sorted_deltas.pickle',
        'src_distribution_pickle_path': 'data/' + ctrl_msg['data_folder_path'].replace('test', 'train') + '/sorted_deltas.pickle',
        'persons_deltas_path': 'data/' + ctrl_msg['data_folder_path'] + '/persons_deltas_score.pickle',
        'persons_ap_path': 'data/' + ctrl_msg['data_folder_path'] + '/persons_ap_scores.pickle',
        'predict_person_path': 'data/' + ctrl_msg['data_folder_path'] + '/predict_persons.pickle',

        'mid_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/cross_mid_score.log',
        'eval_fusion_path': 'data/' + ctrl_msg['data_folder_path'] + '/cross_filter_pid.log',
        'fusion_normal_score_path': 'data/' + ctrl_msg['data_folder_path'] + '/cross_filter_score.log',
        'ep': ctrl_msg['ep'],
        'en': ctrl_msg['en'],
        'window_interval': ctrl_msg['window_interval'],
        'filter_interval': ctrl_msg['filter_interval']
    }
    safe_mkdir('data/' + ctrl_msg['data_folder_path'])

    if '_grid' in ctrl_msg['data_folder_path'] and '_grid_' not in ctrl_msg['data_folder_path']:
        origin_dict['probe_path'] = 'data/grid/grid-cv' + str(ctrl_msg['cv_num']) + '-probe.txt'
        origin_dict['train_path'] = 'data/grid/grid-cv' + str(ctrl_msg['cv_num']) + '-train.txt'
        origin_dict['gallery_path'] = 'data/grid/grid-cv' + str(ctrl_msg['cv_num']) + '-gallery.txt'
    elif '_market' in ctrl_msg['data_folder_path'] and '_market_' not in ctrl_msg['data_folder_path']:
        origin_dict['probe_path'] = 'data/market/probe.txt'
        origin_dict['train_path'] = 'data/market/train.txt'
        origin_dict['gallery_path'] = 'data/market/gallery.txt'
    elif '_dukehead' in ctrl_msg['data_folder_path']:
        origin_dict['probe_path'] = 'data/dukehead/probe.list'
        origin_dict['train_path'] = 'data/dukehead/train.list'
        origin_dict['gallery_path'] = 'data/dukehead/test.list'
    elif '_duketail' in ctrl_msg['data_folder_path']:
        origin_dict['probe_path'] = 'data/duketail/probe.list'
        origin_dict['train_path'] = 'data/duketail/train.list'
        origin_dict['gallery_path'] = 'data/duketail/test.list'
    elif '_duketqtail' in ctrl_msg['data_folder_path']:
        origin_dict['probe_path'] = 'data/duketqtail/probe.list'
        origin_dict['train_path'] = 'data/duketqtail/train.list'
        origin_dict['gallery_path'] = 'data/duketqtail/test.list'
    elif '_dukequerytail' in ctrl_msg['data_folder_path']:
        origin_dict['probe_path'] = 'data/dukequerytail/probe.list'
        origin_dict['train_path'] = 'data/dukequerytail/train.list'
        origin_dict['gallery_path'] = 'data/dukequerytail/test.list'
    elif '_duke' in ctrl_msg['data_folder_path']:
        origin_dict['probe_path'] = 'data/duke/probe.list'
        origin_dict['train_path'] = 'data/duke/train.list'
        origin_dict['gallery_path'] = 'data/duke/test.list'
    if 'train' in ctrl_msg['data_folder_path']:
        origin_dict['answer_path'] = origin_dict['train_path']
    else:
        origin_dict['answer_path'] = origin_dict['probe_path']
    if 'r-' in origin_dict['src_distribution_pickle_path']:
        # use track info before increment
        origin_dict['rand_distribution_pickle_path'] = origin_dict['src_distribution_pickle_path'].replace('r-train', 'train_rand')
    else:
        # use track info after increment
        origin_dict['rand_distribution_pickle_path'] = origin_dict['src_distribution_pickle_path'].replace('train',
                                                                                                           'train_rand')
    origin_dict['rand_distribution_pickle_path'] = origin_dict['src_distribution_pickle_path'].replace('train',
                                                                                                       'train_rand')
    for (k, v) in update_msg.items():
        origin_dict[k] = v
    return origin_dict

