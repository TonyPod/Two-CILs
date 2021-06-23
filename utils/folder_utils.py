# -*- coding:utf-8 -*-  

""" 
@time: 11/30/20 10:44 PM 
@author: Chen He 
@site:  
@file: folder_utils.py
@description:  
"""
import os

import yaml


def gen_base_param_str(args):
    params_str = '%s' % args.network
    params_str += '_%d_%s_%s' % (args.epochs, args.optimizer, str(args.base_lr))
    params_str += '_sigmoid' if args.sigmoid else ''
    params_str += ('_no_final_relu' if args.no_final_relu else '')
    params_str += '_aug' if not args.no_aug else ''
    params_str += ('_wd_' + str(args.weight_decay))
    params_str += ('_warmup_%d' % args.warmup_epochs if args.warmup else '')

    return params_str


def gen_inc_param_str(args):
    # 1. baseline
    params_str = gen_base_param_str(args)
    params_str = (('skip_first_' if (not args.no_skip_base_training and not args.base_model) else '') + params_str)
    if not args.base_model:
        params_str += ('_ada_wd' if args.ada_weight_decay else '')
    params_str += ('_lr_drop_%s' % args.lr_drop_ratio if args.lr_drop else '')

    # 2. rehearsal
    if args.joint_training:
        params_str += '_joint_training'

    if not args.base_model:
        if args.memory_type == 'episodic':
            params_str += '_%s_%d_%s' % (args.selection_strategy,
                                         args.num_exemplars,
                                         args.fixed_budget)

    # 3. regularization
    if args.reg_type == 'lwf':
        params_str += '_lwf_%.1f_temp_%.1f' % (args.reg_loss_weight, args.lwf_loss_temp)
        params_str += '_adj_w' if args.adjust_lwf_w else ''

    # 5. embedding learning
    if args.embedding:
        params_str += '_embedding'
        params_str += '_cosine' if args.test_cosine else ''

    # 6. bias rectification
    if not args.bias_rect == 'none':
        params_str += '_%s' % args.bias_rect
        if args.bias_rect == 'eeil':
            params_str += '_%d' % args.finetune_epochs
        if args.bias_rect == 'bic':
            params_str += '_epochs_%d_w_%s_ratio_%s' % (
                args.bias_epochs, str(args.bias_w), str(args.val_exemplars_ratio))
            params_str += '_aug' if args.bias_aug else ''

    if not args.init_type == 'final' and not args.base_model:
        params_str += '_init_%s' % args.init_type

    return params_str


def gen_dataset_str(args):
    if args.dataset == 'imagenet':
        dataset_str = args.dataset + args.resolution
        dataset_str += '_%s_%s_%d' % (args.order_subset, args.order_type,
                                      args.order_idx)
    elif args.dataset == 'cifar100':
        dataset_str = args.dataset
        dataset_str += '_%s_%d' % (args.order_type, args.order_idx)
    else:
        raise Exception()

    return dataset_str


def create_folder(args):
    params_str = gen_inc_param_str(args)
    dataset_str = gen_dataset_str(args)

    # output folder
    OUTPUT_FOLDER = os.path.join('result', dataset_str,
                                 'base_%d' % args.base_cl if args.base_model else 'base_%d_inc_%d_total_%d' % (
                                     args.base_cl, args.nb_cl, args.total_cl), 'seed_%d' % args.random_seed, params_str)

    print('Result folder: %s' % OUTPUT_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    with open(os.path.join(OUTPUT_FOLDER, 'config.yaml'), 'w') as fout:
        yaml.dump(vars(args), fout, default_flow_style=False)

    args.OUTPUT_FOLDER = OUTPUT_FOLDER

    return args


def create_subfolder(args, class_inc):
    # group status
    new_classes_idx = range(len(class_inc.old_wnids), len(class_inc.cumul_wnids))
    print('Group %d: (%d-%d)' % (class_inc.group_idx + 1, min(new_classes_idx), max(new_classes_idx)))

    # define sub-folder
    if class_inc.group_idx > 0:
        assert hasattr(args, 'SUB_OUTPUT_FOLDER')
        args.PREV_SUBFOLDER = args.SUB_OUTPUT_FOLDER
    args.SUB_OUTPUT_FOLDER = os.path.join(args.OUTPUT_FOLDER, 'group_%d' % (class_inc.group_idx + 1))
    if not os.path.exists(args.SUB_OUTPUT_FOLDER):
        os.makedirs(args.SUB_OUTPUT_FOLDER)

    with open(os.path.join(args.SUB_OUTPUT_FOLDER, 'new_wnids.txt'), 'w') as fout:
        if args.dataset == 'imagenet':
            fout.writelines([wnid + os.linesep for wnid in class_inc.cur_wnids])

    with open(os.path.join(args.SUB_OUTPUT_FOLDER, 'new_classes.txt'), 'w') as fout:
        if args.dataset == 'imagenet':
            fout.writelines([name + os.linesep for name in class_inc.cur_names])

    args.CHECKPOINT_FOLDER = os.path.join(args.SUB_OUTPUT_FOLDER, 'checkpoints')

    args.SUB_RESULT_FOLDER = os.path.join(args.SUB_OUTPUT_FOLDER, 'result')
    if not os.path.exists(args.SUB_RESULT_FOLDER):
        os.makedirs(args.SUB_RESULT_FOLDER)

    return args
