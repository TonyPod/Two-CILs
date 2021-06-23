# -*- coding:utf-8 -*-  

""" 
@time: 12/17/19 9:22 PM 
@author: Chen He 
@site:  
@file: vis.py
@description:  
"""

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from functools import partial

# specify the colors of different methods you want (if not set, then the "color_list" below is used)
color_map = {
}

color_list = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
              '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#a9a9a9', '#ffffff', '#000000']


def vis_loss(data_dict, result_dir, filename):
    keys = sorted(data_dict.keys())
    values = [data_dict[key] for key in keys]

    plt.figure()
    plt.xlabel('Step')
    plt.ylabel(filename)
    plt.plot(keys, values, marker='.')
    plt.savefig(os.path.join(result_dir, '%s.pdf' % filename))
    plt.close()


def vis_conf_mat(test_conf_mat_filename, class_names, all_classes, vmax):
    test_conf_mat = np.load(test_conf_mat_filename)
    plt.figure()
    sns.heatmap(test_conf_mat, vmax=vmax, square=True, xticklabels=[class_names[i] for i in all_classes],
                yticklabels=[class_names[i] for i in all_classes])
    plt.savefig(os.path.splitext(test_conf_mat_filename)[0] + '.pdf')
    plt.close()


def vis_acc_curve(args, middle_folder, type):
    filename_dict = {
        'top1': 'top1_acc',
        'top5': 'top5_acc',
        'harmonic_mean': 'harmonic_mean',
    }

    name_dict = {
        'top1': 'Top-1 Accuracy',
        'top5': 'Top-5 Accuracy',
        'harmonic_mean': 'Harmonic Mean',
    }

    filename = filename_dict[type]
    name = name_dict[type]

    output_result_folder = os.path.join(args.OUTPUT_FOLDER, 'result_curve', middle_folder)
    if not os.path.exists(output_result_folder):
        os.makedirs(output_result_folder)

    # load
    accs = []
    for group_idx in range((args.total_cl - args.base_cl) // args.nb_cl + 1):
        result_filename = os.path.join(args.OUTPUT_FOLDER, 'group_%d' % (group_idx + 1), 'result', middle_folder,
                                       '%s.txt' % filename)
        if not os.path.exists(result_filename):
            break
        acc = float(open(result_filename, 'r').readline().strip())
        accs.append(acc)
    accs = np.array(accs)

    # txt
    with open(os.path.join(output_result_folder, '%s_curve.txt' % filename), 'w') as fout:
        for acc in accs:
            fout.write('%.2f' % acc + os.linesep)

    # plot
    plt.figure()
    plt.title('%s %s Curve' % (args.dataset, name))
    plt.xlabel('#Classes')
    plt.ylabel('Accuracy')
    plt.plot(range(args.base_cl, args.total_cl + args.nb_cl, args.nb_cl)[:len(accs)], accs, marker='.')
    for a, b in zip(range(args.base_cl, args.total_cl + args.nb_cl, args.nb_cl)[:len(accs)], accs):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    plt.savefig(os.path.join(output_result_folder, '%s_curve.pdf' % filename))
    plt.close()


vis_top1_acc_curve = partial(vis_acc_curve, type='top1')
vis_top5_acc_curve = partial(vis_acc_curve, type='top5')
vis_harmonic_mean_curve = partial(vis_acc_curve, type='harmonic_mean')


def vis_old_new_acc_curve(result_dir, accs, old_accs, new_accs):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if isinstance(accs, dict):
        epochs = sorted(accs.keys())
    else:
        epochs = range(len(accs))

    assert len(old_accs) == len(new_accs)
    plt.figure()
    plt.xlabel('#Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, [accs[epoch] for epoch in epochs], marker='.', label='avg', color='#3574B2')
    if len(old_accs) > 0:
        plt.plot(epochs, [old_accs[epoch] for epoch in epochs], marker='.', label='old', color='#36A221')
        plt.plot(epochs, [new_accs[epoch] for epoch in epochs], marker='.', label='new', color='#F67D06')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'old_new_acc_curve.pdf'))
    plt.close()


def calc_mean_std(result):
    mean_result = np.mean(result, axis=0)
    std_result = np.std(result, axis=0)
    return mean_result, std_result


def vis_multiple(result_dir_dict, total_cl, nb_cl, keys, output_name, title='CIFAR-100', MIDDLE_FOLDER='', ylim=None):
    print('[Top-1 ACC]')
    fontsize = 14

    x = [i + nb_cl for i in range(0, total_cl, nb_cl)]
    x_names = [str(i) for i in x]
    y = range(0, 110, 10)
    y_names = [str(i) + '%' for i in range(0, 110, 10)]

    plt.figure(figsize=(10, 10), dpi=220)

    plt.gca().set_autoscale_on(False)

    plt.xlim(0, total_cl)
    plt.ylim(0, 100)

    plt.xticks(x, x_names, rotation=45, fontsize=fontsize)
    plt.yticks(y, y_names, fontsize=fontsize)
    plt.margins(0)

    plt.xlabel("Number of classes", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.title(title)

    # Horizontal reference lines
    for i in range(10, 100, 10):
        plt.hlines(i, 0, total_cl, colors="lightgray", linestyles="dashed")

    for key_idx, key in enumerate(keys):

        result_dirs = result_dir_dict[key]
        aver_acc_over_time_mul = []

        for result_dir in result_dirs:
            accs = []
            for group_idx in range((total_cl - nb_cl) // nb_cl + 1):
                conf_mat_filename = os.path.join(result_dir, 'group_%d' % (group_idx + 1), MIDDLE_FOLDER,
                                                 'conf_mat.npy')
                if not os.path.exists(conf_mat_filename):
                    break
                conf_mat = np.load(conf_mat_filename)
                acc = np.mean(np.diag(conf_mat) * 100. / np.sum(conf_mat, axis=1))
                accs.append(acc)
            accs = np.array(accs)
            aver_acc_over_time_mul.append(accs)

        y_mean, y_std = calc_mean_std(np.array(aver_acc_over_time_mul))
        try:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_map[key])
        except:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_list[key_idx])

        print('%s: %.2f' % (key, y_mean[-1]))

    plt.legend(fontsize=fontsize)
    if ylim is not None:
        plt.ylim(ylim)

    plt.savefig('%s.pdf' % output_name)


def get_running_time(result_dir_dict, total_cl, nb_cl, keys):
    print('[Running time]')
    for key_idx, key in enumerate(keys):

        result_dirs = result_dir_dict[key]
        aver_acc_over_time_mul = []

        for result_dir in result_dirs:
            time_arr = []
            for group_idx in range((total_cl - nb_cl) // nb_cl + 1):
                time_filename = os.path.join(result_dir, 'stat_time.txt')
                if not os.path.exists(time_filename):
                    break
                times = [float(line.strip()) for line in open(time_filename, 'r').readlines()]
                time = np.mean(times)
                time_arr.append(time)
            time_arr = np.array(time_arr)
            aver_acc_over_time_mul.append(time_arr)

        y_mean, y_std = calc_mean_std(np.array(aver_acc_over_time_mul))
        print('%s: %.2f' % (key, y_mean[-1]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Class Incremental Learning')

    # settings (these are almost the same as in main.py)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--resolution', '-r', type=str, default='')
    parser.add_argument('--order_subset', type=str, default='')
    parser.add_argument('--order_type', type=str, default='random')
    parser.add_argument('--network', type=str, default='resnet18', help='[resnet18]')

    parser.add_argument('--base_cl', type=int, default=10)
    parser.add_argument('--nb_cl', type=int, default=10)
    parser.add_argument('--total_cl', type=int, default=100)

    parser.add_argument('--num_orders', type=int, default=1)
    parser.add_argument('--stage', type=str, default='2nd_stage')

    args = parser.parse_args()

    # parse the arguments
    dataset = ('%s%s' % (args.dataset, args.resolution) + (
        '_%s' % args.order_subset if args.order_subset != '' else ''))
    network = args.network
    class_order = args.order_type
    stage = args.stage
    inc_protocol = 'base_%d_inc_%d_total_%d' % (args.base_cl, args.nb_cl, args.total_cl)
    num_order = args.num_orders

    if args.dataset == 'cifar100':
        title = 'CIFAR-100'
        if network == 'lenet':
            base_lr = 0.001
        elif network == 'resnet':
            base_lr = 0.005
    elif args.dataset == 'imagenet':
        title = 'Group ImageNet'
        if network == 'mobilenet':
            base_lr = 0.005
        elif network == 'resnet18':
            base_lr = 0.005

    dirname = os.path.dirname(__file__)

    result_dir_dict = {
        'Lowerbound': [
            os.path.join(dirname, '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001/' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in range(num_order)
        ],
        'LwF': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0/' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'iCaRL': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_no_final_relu_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_embedding_cosine' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'EEIL': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_eeil_30' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'LSIL': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adj_w_bic_epochs_2_w_0.1_ratio_0.1_aug' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'IL2M': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_il2m' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'MDFCIL': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'GDumb': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_random_undersampling_init_all' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'Re-weighting': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_reweighting' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'Post-scaling': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_post_scaling' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'Under-sampling': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_random_undersampling' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'Over-sampling': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_random_oversampling' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'SMOTE': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_smote' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'ADASYN': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adasyn' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'K-means': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_kmeans' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'K-medoids': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_kmedoids' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ],
        'Joint Training': [
            os.path.join(dirname,
                         '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_70_adam_%s_aug_wd_0.0001_joint_training' % (
                             dataset, class_order, i + 1, inc_protocol, network, str(base_lr))) for i in
            range(num_order)
        ]

    }
    keys = ['Lowerbound', 'LwF', 'iCaRL', 'EEIL', 'LSIL', 'IL2M', 'MDFCIL', 'GDumb', 'Re-weighting', 'Post-scaling',
            'Under-sampling', 'Over-sampling', 'SMOTE', 'ADASYN', 'K-means', 'K-medoids', 'Joint Training']
    total_cl, nb_cl = 100, 10
    output_name = '%s_%s_%s' % (dataset, class_order, network)
    print('%s %s: %s (%s)' % (dataset, class_order, network, stage))
    vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name,
                 MIDDLE_FOLDER='result/%s' % stage, title=title, ylim=(10, 100))
    get_running_time(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys)
