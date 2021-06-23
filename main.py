# -*- coding:utf-8 -*-  

""" 
@time: 11/30/20 10:09 PM 
@author: Chen He 
@site:  
@file: main.py.py
@description:  this file is the entrance of the training process
"""

import argparse
import os
import random
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
import wandb

import trainer
from datasets.dataset import Dataset, CILProtocol
from networks import output_layer
from networks.backbone import get_backbone
from utils import folder_utils, memory_utils
from utils.stats import ModelStat

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Class Incremental Learning')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--resolution', '-r', type=str, default='64x64')
    parser.add_argument('--order_subset', type=str, default='10x10')
    parser.add_argument('--order_type', type=str, default='random')
    parser.add_argument('--order_idx', type=int, default=1)

    # incremental protocol
    parser.add_argument('--base_model', action='store_true', help='whether it is the first class increment')
    parser.add_argument('--base_cl', type=int, default=10, help='number of classes in the first class increment')
    parser.add_argument('--nb_cl', type=int, default=10, help='number of classes added at later class increments')
    parser.add_argument('--total_cl', type=int, default=100, help='total number of classes')
    parser.add_argument('--to_group_idx', type=int, default=-1,
                        help='stop training at which class increment (for debug)')
    parser.add_argument('--random_seed', type=int, default=1993)

    # 1. baseline
    parser.add_argument('--network', type=str, default='resnet18',
                        help='lenet and resnet for CIFAR-100; resnet18 and mobilenet for Group ImageNet')
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs for training')
    parser.add_argument('--optimizer', type=str, default='adam', help='[adam|sgd]')
    parser.add_argument('--baseline', type=str, default='fc', help='the final layer')
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--sigmoid', action='store_true', help='sigmoid or softmax')
    parser.add_argument('--no_final_relu', action='store_true', help='no final ReLU in the final FC layer (for iCaRL)')
    parser.add_argument('--no_aug', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ada_weight_decay', action='store_true', help='adaptive weight decay')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--warmup', action='store_true', help='whether to use warmup in the training process')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='epochs in the warmup')

    # lr in later increments
    parser.add_argument('--lr_drop', action='store_true', help='whether to drop lr in later increments')
    parser.add_argument('--lr_drop_ratio', type=float, default=0.2, help='learning rate ratio in later increments')

    # 2. rehearsal
    parser.add_argument('--memory_type', type=str, default='episodic', choices=['episodic', 'none'])

    # 2.1 episodic
    parser.add_argument('--num_exemplars', type=int, default=20)
    parser.add_argument('--selection_strategy', type=str, default='random')
    parser.add_argument('--fixed_budget', type=str, default='total', help='[total|each]')
    parser.add_argument('--show_exemplars', action='store_true')

    # 3. regularization
    parser.add_argument('--reg_type', type=str, default='lwf', choices='[lwf|none]')
    parser.add_argument('--reg_loss_weight', type=float, default=1.)
    parser.add_argument('--lwf_loss_temp', type=float, default=2.)
    parser.add_argument('--adjust_lwf_w', action='store_true')

    # 5. embedding
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument('--test_cosine', action='store_true')

    # 6. bias rectification
    parser.add_argument('--bias_rect', type=str, default='post_scaling', choices=['post_scaling', 'reweighting',
                                                                                  'il2m', 'eeil', 'bic',
                                                                                  'weight_aligning',
                                                                                  'weight_aligning_no_bias',
                                                                                  'none', 'smote', 'adasyn',
                                                                                  'random_oversampling',
                                                                                  'random_undersampling',
                                                                                  'kmeans',
                                                                                  'kmedoids',
                                                                                  'near_miss_1', 'near_miss_2',
                                                                                  'near_miss_3', 'smote_tomek',
                                                                                  'smote_enn'])
    # eeil
    parser.add_argument('--finetune_epochs', type=int, default=30)

    # bic (lsil)
    parser.add_argument('--bias_epochs', type=int, default=2)
    parser.add_argument('--bias_w', type=float, default=0.1)
    parser.add_argument('--val_exemplars_ratio', type=float, default=0.1, help='0.1 for 1:9 as the paper suggests')
    parser.add_argument('--bias_aug', action='store_true')

    # 7. other settings
    parser.add_argument('--init_type', type=str, default='final')
    parser.add_argument('--joint_training', action='store_true')

    # debug switch
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb_flag', action='store_true')
    parser.add_argument('--stat_boundary', action='store_true')
    parser.add_argument('--geometric_view', action='store_true')

    # use the same base model for fair comparison
    parser.add_argument('--no_skip_base_training', action='store_true')

    args = parser.parse_args()

    '''
    Check arguments
    '''
    assert args.memory_type == 'none' and args.bias_rect == 'none' and args.reg_type == 'none' if args.joint_training else True
    assert args.num_exemplars > 0 if args.memory_type == 'episodic' else True

    if args.embedding:
        args.no_final_relu = True

    if args.dataset == 'imagenet':
        if args.resolution == '64x64':
            args.batch_size = 128
            if args.network == 'resnet18':
                args.base_lr = 0.005
            elif args.network == 'mobilenet':
                args.base_lr = 0.005
            else:
                raise Exception()
        else:
            raise Exception()

    elif args.dataset == 'cifar100':
        args.resolution = '32x32'
        args.order_type = 'random'
        args.order_subset = ''
        if args.network == 'lenet':
            args.base_lr = 0.001
        elif args.network == 'resnet':
            args.base_lr = 0.005
        else:
            raise Exception()
    else:
        raise Exception()

    if args.base_model:
        args.memory_type = 'none'
        args.reg_type = 'none'
        args.hat = False
        args.bias_rect = 'none'

    if args.bias_rect == 'il2m':
        args.reg_type = 'none'

    # assert args.no_aug == False
    assert args.memory_type == 'episodic' if not args.bias_rect == 'none' else True
    assert not args.memory_type == 'none' if args.embedding else True

    # print hyperparameters
    pprint(vars(args), width=1)

    # start time
    program_start_wall_time = time.time()

    # some settings
    args.AUTOTUNE = tf.data.experimental.AUTOTUNE
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.debug:
        tf.config.experimental_run_functions_eagerly(True)

    # random seed
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    '''
    CIL Training
    '''

    # create parent folder
    args = folder_utils.create_folder(args)

    # init dataset
    dataset = Dataset.get(args)

    # init protocol
    protocol = CILProtocol(args, dataset)

    # init backbone
    backbone = get_backbone(args)

    # init memory
    memory = memory_utils.get_memory(args)

    # init stats
    stats = ModelStat(args)

    # init weights & biases
    if args.wandb_flag:
        wandb.init(project='revisitcil' if not args.base_model else 'revisitcil_base', config=vars(args))

    # start incremental process
    print('Start training...')
    for class_inc in protocol:

        # create subfolder
        args = folder_utils.create_subfolder(args, class_inc)

        # get old model
        if class_inc.group_idx > 0:
            assert 'model' in locals()
            old_model = tf.keras.models.clone_model(model)
            old_model.set_weights(model.get_weights())
            old_model.trainable = False
        else:
            old_model = None

        # get current model (borrow knowledge from before)
        if args.init_type == 'all' and class_inc.group_idx > 0:
            backbone = get_backbone(args)
        model = output_layer.get_output_layer(args, backbone, class_inc)
        if class_inc.group_idx > 0:
            output_layer.special_init(model, old_model, args.init_type)

        # start training
        trainer.train(args, model, old_model, class_inc, memory, stats)

        # exit when at defined group_idx
        if class_inc.group_idx == args.to_group_idx:
            break

    # calculate time
    program_stop_wall_time = time.time()
    print('TOTAL RUNNING WALL TIME: %.2f' % (program_stop_wall_time - program_start_wall_time))

    # upload experiment info to wandb
    if args.wandb_flag:
        stats.upload_wandb()
