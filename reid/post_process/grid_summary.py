from reid.utils.file_helper import read_lines, write


def avg_acc(grid_eval_path):
    grid_infos = read_lines(grid_eval_path)
    before_vision_accs = [0.0, 0.0, 0.0]
    before_fusion_accs = [0.0, 0.0, 0.0]
    after_vision_accs = [0.0, 0.0, 0.0]
    after_fusion_accs = [0.0, 0.0, 0.0]
    i_cv_cnt = 0
    for i, grid_info in enumerate(grid_infos):
        if i % 2 != 0:
            accs = grid_info.split()
            if i_cv_cnt % 4 == 0:
                for j in range(3):
                    before_vision_accs[j] += float(accs[j])
            if i_cv_cnt % 4 == 1:
                for j in range(3):
                    before_fusion_accs[j] += float(accs[j])
            if i_cv_cnt % 4 == 2:
                for j in range(3):
                    after_vision_accs[j] += float(accs[j])
            if i_cv_cnt % 4 == 3:
                for j in range(3):
                    after_fusion_accs[j] += float(accs[j])
            i_cv_cnt += 1
    write('grid_eval.log', '\n' + grid_eval_path + '\n')
    write('grid_eval.log', 'before_retrain_vision\n& %.2f & %.2f & %.2f\n' % (before_vision_accs[0]*10, before_vision_accs[1]*10, before_vision_accs[2]*10))
    write('grid_eval.log', 'before_retrain_fusion\n& %.2f & %.2f & %.2f\n' % (before_fusion_accs[0]*10, before_fusion_accs[1]*10, before_fusion_accs[2]*10))
    write('grid_eval.log', 'after_retrain_vision\n& %.2f & %.2f & %.2f\n' % (after_vision_accs[0]*10, after_vision_accs[1]*10, after_vision_accs[2]*10))
    write('grid_eval.log', 'after_retrain_fusion\n& %.2f & %.2f & %.2f\n' % (after_fusion_accs[0]*10, after_fusion_accs[1]*10, after_fusion_accs[2]*10))


def avg_acc2(grid_eval_path):
    grid_infos = read_lines(grid_eval_path)
    before_vision_accs = [0.0, 0.0, 0.0, 0.0]
    before_fusion_accs = [0.0, 0.0, 0.0]
    after_vision_accs = [0.0, 0.0, 0.0]
    after_fusion_accs = [0.0, 0.0, 0.0]
    i_cv_cnt = 0
    for i, grid_info in enumerate(grid_infos):
        if i % 2 != 0:
            accs = grid_info.split()
            for j in range(4):
                before_vision_accs[j] += float(accs[j])

            i_cv_cnt += 1
    write('grid_eval.log', '\n' + grid_eval_path + '\n')
    write('grid_eval.log', '& %.2f & %.2f & %.2f & %.2f\n' % (before_vision_accs[0]*10, before_vision_accs[1]*10, before_vision_accs[2]*10, before_vision_accs[3]*10))

if __name__ == '__main__':
    avg_acc2('market_grid.log')
    avg_acc2('duke_grid.log')
    avg_acc2('viper_grid.log')
    avg_acc2('cuhk_grid.log')
    # avg_acc('cuhk_grid.log')
    # avg_acc('viper_grid.txt')
    # avg_acc('grid_grid.txt')