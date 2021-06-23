# -*- coding:utf-8 -*-  

""" 
@time: 11/3/20 11:36 AM 
@author: Chen He 
@site:  
@file: evaluator.py
@description:  
"""

import os

import numpy as np
import tensorflow as tf

from utils.utils_func import get_top5_acc, get_harmonic_mean
from utils.vis import vis_conf_mat, vis_top1_acc_curve, vis_top5_acc_curve, vis_harmonic_mean_curve


def get_preds(model, num_classes, test_images, args, use_embedding=False, memory=None):
    """
    Get the prediction scores of the some test images
    :param model:
    :param num_classes:
    :param test_images:
    :param args:
    :param use_embedding:
    :param memory:
    :return:
    """
    assert memory is not None if use_embedding else True

    if use_embedding:

        backbone = tf.keras.Model(model.input, model.layers[args.feat_layer_idx].output)
        images, labels = memory.load_raw(old_or_new='new')

        class_means_feats = []
        for class_idx in range(num_classes):
            class_exemplars_feats = backbone.predict(images[labels == class_idx])
            if args.test_cosine:
                class_exemplars_feats = class_exemplars_feats / np.expand_dims(
                    np.linalg.norm(class_exemplars_feats, axis=1),
                    axis=1)
                class_mean_feats = np.mean(class_exemplars_feats, axis=0)
                class_mean_feats = class_mean_feats / np.linalg.norm(class_mean_feats)
            else:
                class_mean_feats = np.mean(class_exemplars_feats, axis=0)

            class_means_feats.append(class_mean_feats)
        class_means_feats = np.stack(class_means_feats)

        test_feats = backbone.predict(test_images)
        if args.test_cosine:
            test_feats = test_feats / np.expand_dims(np.linalg.norm(test_feats, axis=1), axis=1)
        test_logits = []
        for feat_idx in range(0, len(test_feats), args.batch_size):
            test_logits.extend(-np.sum(np.square(
                np.expand_dims(class_means_feats, axis=0) - np.expand_dims(
                    test_feats[feat_idx:feat_idx + args.batch_size],
                    axis=1)),
                axis=-1))
        test_logits = tf.stack(test_logits)
        if args.test_cosine:
            test_scores = (test_logits + 2) / 2
        else:
            test_scores = tf.exp(test_logits)
    else:
        test_logits = model.predict(test_images)
        test_scores = tf.nn.sigmoid(test_logits) if args.sigmoid else tf.nn.softmax(test_logits)

    test_scores = test_scores.numpy()

    return test_scores


def evaluate(model, class_inc, args, STAGE_FOLDER='1st_stage', use_embedding=False, memory=None):
    test_scores = get_preds(model, len(class_inc.cumul_wnids), class_inc.test_images_now, args, use_embedding, memory)
    test_preds = tf.argmax(test_scores, axis=1, output_type=tf.int32)

    SUB_RESULT_FOLDER = os.path.join(args.SUB_RESULT_FOLDER, STAGE_FOLDER)
    if not os.path.exists(SUB_RESULT_FOLDER):
        os.makedirs(SUB_RESULT_FOLDER)

    with open(os.path.join(SUB_RESULT_FOLDER, 'preds.txt'), 'w') as fout:
        for i in range(len(class_inc.test_labels_now)):
            fout.write('%s-%d\t%s\t%d\t%s\t%s' % (class_inc.wnid_order[class_inc.test_labels_now[i]],
                                                  i + 1, class_inc.name_order[class_inc.test_labels_now[i]],
                                                  test_preds[i],
                                                  class_inc.name_order[test_preds[i]],
                                                  'Y' if class_inc.test_labels_now[i] == test_preds[
                                                      i] else 'N') + os.linesep)
    test_conf_mat = tf.math.confusion_matrix(class_inc.test_labels_now, test_preds).numpy()
    test_conf_mat_filename = os.path.join(SUB_RESULT_FOLDER, 'conf_mat.npy')
    np.save(test_conf_mat_filename, test_conf_mat)

    # per class acc
    test_per_class_acc = np.diag(test_conf_mat) * 1. / np.sum(test_conf_mat, axis=1)
    with open(os.path.join(SUB_RESULT_FOLDER, 'test_per_class_acc.txt'), 'w') as f:
        f.write(os.linesep.join([str(elem) for elem in test_per_class_acc]))

    # top-1 acc
    top1_acc = np.mean(test_per_class_acc)
    with open(os.path.join(SUB_RESULT_FOLDER, 'top1_acc.txt'), 'w') as fout:
        fout.write('%f' % top1_acc)

    # top-5 acc
    top5_acc = get_top5_acc(test_scores, class_inc.test_labels_now)
    with open(os.path.join(SUB_RESULT_FOLDER, 'top5_acc.txt'), 'w') as fout:
        fout.write('%f' % top5_acc)

    # harmonic mean
    harmonic_mean = get_harmonic_mean(test_per_class_acc, len(class_inc.old_wnids))
    with open(os.path.join(SUB_RESULT_FOLDER, 'harmonic_mean.txt'), 'w') as fout:
        fout.write('%f' % harmonic_mean)

    # vis result
    print('INCREMENT %d (%s): Top-1 ACC: %.2f, Top-5 ACC: %.2f, Harmonic Mean: %.2f' % (
        class_inc.group_idx + 1, STAGE_FOLDER, top1_acc, top5_acc, harmonic_mean))
    vis_conf_mat(test_conf_mat_filename, class_inc.name_order, range(len(class_inc.cumul_wnids)),
                 class_inc.dataset.NUM_MAX_TEST_SAMPLES_PER_CLASS)

    if not args.base_model:
        vis_top1_acc_curve(args, middle_folder=STAGE_FOLDER)
        vis_top5_acc_curve(args, middle_folder=STAGE_FOLDER)
        vis_harmonic_mean_curve(args, middle_folder=STAGE_FOLDER)

    return top1_acc, top5_acc, harmonic_mean
