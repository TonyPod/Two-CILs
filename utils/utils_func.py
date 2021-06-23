# -*- coding:utf-8 -*-  

""" 
@time: 12/5/19 10:10 PM 
@author: Chen He 
@site:  
@file: utils_func.py
@description:  
"""

import os
import pickle

import numpy as np
import wandb

from utils.vis import vis_loss


class MySummary:
    def __init__(self):
        self.log = dict()

    def add(self, name, step, value):
        if name not in self.log:
            self.log[name] = dict()

        self.log[name][step] = value

    def dump(self, filename):
        pickle.dump(self.log, open(filename, 'wb'))

    def vis(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for key in self.log:
            vis_loss(self.log[key], folder, key)

    def reset(self):
        self.log.clear()


def wandb_log(args, stats):
    if args.wandb_flag:
        wandb.log(stats)


def get_top5_acc(logits, labels):
    top5_hit = 0
    for img_idx, single_logits in enumerate(logits):
        top5_hit += 1 if labels[img_idx] in np.argsort(-single_logits)[:5] else 0
    top5_acc = top5_hit * 100. / len(labels)
    return top5_acc


def get_harmonic_mean(per_class_accs, num_old_classes):
    new_acc = np.mean(per_class_accs[num_old_classes:])
    if num_old_classes == 0:
        hm = new_acc
    else:
        old_acc = np.mean(per_class_accs[:num_old_classes])
        hm = (2 * new_acc * old_acc / (new_acc + old_acc))
    return hm


def post_scaling_il2m(logits, group_idx, nb_cl, old_num_classes, init_classes_means, current_classes_means,
                      models_confidence):
    pred_to_new_class_indices = np.where(np.argmax(logits, axis=1) >= old_num_classes)[0]
    for old_class_idx in range(old_num_classes):
        logits[pred_to_new_class_indices, old_class_idx] *= \
            init_classes_means[old_class_idx] / current_classes_means[old_class_idx] * \
            models_confidence[group_idx] / models_confidence[old_class_idx // nb_cl]
    return logits


def get_folder_size(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if
               os.path.isfile(os.path.join(path, f)))


def lr_scheduler(args, class_inc):
    if args.epochs > 100:
        lr_desc_epochs = [int(0.6 * args.epochs), int(0.8 * args.epochs), int(0.9 * args.epochs)]
    else:
        lr_desc_epochs = [int(0.7 * args.epochs), int(0.9 * args.epochs)]

    lrs = []
    if class_inc.group_idx > 0 and args.lr_drop:
        base_lr = args.base_lr * args.lr_drop_ratio
    else:
        base_lr = args.base_lr
    lr = base_lr

    for epoch in range(args.epochs):
        if epoch in lr_desc_epochs:
            lr *= 0.1
        lrs.append(lr)

    if args.warmup:
        warmup_lrs = list(base_lr / np.arange(20, 0, -1))
        lrs = warmup_lrs + lrs

    return lrs
