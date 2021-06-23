# -*- coding:utf-8 -*-  

""" 
@time: 2020/5/31 22:03 
@author: Chen He 
@site:  
@file: dataset.py
@description:  
"""

import os

import tensorflow as tf


class ClassIncrement:
    def __init__(self, args, dataset, cur_wnids, cur_names, old_wnids, old_names, cumul_wnids, cumul_names, wnid_order,
                 class_names, group_idx, nb_groups, is_final_inc):
        self.dataset = dataset

        self.cur_wnids = cur_wnids
        self.cur_names = cur_names
        self.cumul_wnids = cumul_wnids
        self.cumul_names = cumul_names
        self.old_wnids = old_wnids
        self.old_names = old_names

        self.wnid_order = wnid_order
        self.name_order = class_names
        self.args = args

        self.group_idx = group_idx
        self.nb_groups = nb_groups
        self.is_final_inc = is_final_inc

        self.load_train()

    def load_train(self):

        if self.args.dataset == 'imagenet':
            # train dataset
            if self.args.resolution == '64x64':
                self.train_dataset = self.dataset.load_train(self.cur_wnids,
                                                             self.wnid_order,
                                                             num_samples=-1,
                                                             use_shuffle=False,
                                                             parallel_calls=not self.args.debug)
            else:
                raise Exception()

            # test set
            self.test_dataset = self.dataset.load_test(self.cumul_wnids, self.wnid_order)
        elif self.args.dataset in 'cifar100':
            self.train_dataset = self.dataset.load_train(self.cur_wnids, self.wnid_order, num_samples=-1)
            self.test_dataset = self.dataset.load_test(self.cumul_wnids, self.wnid_order)
        else:
            raise Exception()

        self.test_images_now = self.test_dataset.map(lambda img, label: img).batch(self.args.batch_size).prefetch(
            self.args.AUTOTUNE)
        self.test_labels_now = tf.stack(list(self.test_dataset.map(lambda img, label: label)))


class CILProtocol:
    def __init__(self, args, dataset):

        # load class order
        if args.dataset == 'imagenet':
            order_file = os.path.join(os.path.dirname(__file__), args.dataset,
                                      'order_%s_%s_%d_wnid.txt' % (
                                          args.order_subset, args.order_type, args.order_idx))
            wnid_order = []
            with open(order_file, 'r') as fin:
                for line in fin.readlines():
                    wnid_order.append(line.strip())
        elif args.dataset == 'cifar100':
            order_file = os.path.join(os.path.dirname(__file__), args.dataset,
                                      'order_%d.txt' % args.order_idx)
            wnid_order = []
            with open(order_file, 'r') as fin:
                for line in fin.readlines():
                    wnid_order.append(int(line.strip()))
        else:
            raise Exception('Invalid dataset')

        # load dataset
        class_names = [dataset.label_names_dict[wnid] for wnid in wnid_order]

        class_name_groups = []
        class_wnid_groups = []
        class_name_groups.append([class_names[idx] for idx in range(args.base_cl)])
        class_wnid_groups.append([wnid_order[idx] for idx in range(args.base_cl)])
        if 'base_model' not in args or not args.base_model:
            num_classes = min(len(wnid_order), args.total_cl)
            NUM_INC = (num_classes - args.base_cl) // args.nb_cl
            for inc_idx in range(NUM_INC):
                class_name_groups.append([class_names[idx] for idx in
                                          range(args.base_cl + inc_idx * args.nb_cl,
                                                args.base_cl + (inc_idx + 1) * args.nb_cl)])
            for inc_idx in range(NUM_INC):
                class_wnid_groups.append([wnid_order[idx] for idx in
                                          range(args.base_cl + inc_idx * args.nb_cl,
                                                args.base_cl + (inc_idx + 1) * args.nb_cl)])
        if args.debug:
            for group_idx in range(len(class_wnid_groups)):
                print('Group %d: ' % (group_idx + 1))
                print(os.linesep.join(
                    ['%s: %s' % (wnid, dataset.label_names_dict[wnid]) for wnid in class_wnid_groups[group_idx]]))

        self.wnid_order, self.class_names, self.class_wnid_groups, self.class_name_groups = wnid_order, class_names, class_wnid_groups, class_name_groups
        self.dataset = dataset
        self.group_idx = -1
        self.args = args
        self.order_file = order_file

    def __iter__(self):
        return self

    def __next__(self):
        self.group_idx += 1
        if self.group_idx >= len(self.class_wnid_groups):
            raise StopIteration()

        is_final_inc = self.group_idx == len(self.class_wnid_groups) - 1

        cumul_wnids = sum(self.class_wnid_groups[:(self.group_idx + 1)], [])
        cumul_names = sum(self.class_name_groups[:(self.group_idx + 1)], [])
        if self.args.joint_training:
            cur_wnids = cumul_wnids
            cur_names = cumul_names
        else:
            cur_wnids = self.class_wnid_groups[self.group_idx]
            cur_names = self.class_name_groups[self.group_idx]

        class_inc = ClassIncrement(self.args, self.dataset,
                                   cur_wnids, cur_names,
                                   sum(self.class_wnid_groups[:self.group_idx], []),
                                   sum(self.class_name_groups[:self.group_idx], []),
                                   cumul_wnids, cumul_names,
                                   self.wnid_order, self.class_names, self.group_idx, len(self.class_wnid_groups),
                                   is_final_inc)
        return class_inc


class Dataset:

    def __init__(self, args):
        self.args = args

    @staticmethod
    def get(args):
        if args.dataset == 'imagenet':
            if args.resolution == '64x64':
                from datasets.imagenet64x64 import ImageNet64x64
                dataset = ImageNet64x64(args)
            else:
                raise Exception('Invalid resolution')
        elif args.dataset == 'cifar100':
            from datasets.cifar100 import CIFAR100
            dataset = CIFAR100(args)
        else:
            raise Exception('Invalid dataset name')

        return dataset

    def aug_fn(self, image, label):
        raise NotImplementedError

    def preprocess(self, images, **kwargs):
        raise NotImplementedError
