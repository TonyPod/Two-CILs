# -*- coding:utf-8 -*-  

""" 
@time: 12/29/20 10:53 AM 
@author: Chen He 
@site:  
@file: episodic_memory.py
@description:  
"""

import os
import pickle

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from utils import imagenet_preprocessing
from utils.memory import Memory


class EpisodicMemory(Memory):

    def __init__(self, args):
        super().__init__(args)
        self.stats = []

    def _get_exemplar_file(self):
        return os.path.join(self.args.SUB_OUTPUT_FOLDER, 'exemplars.pkl')

    def _get_old_exemplar_file(self):
        return os.path.join(self.args.PREV_SUBFOLDER, 'exemplars.pkl')

    def size(self):
        return os.stat(self._get_exemplar_file()).st_size

    def load_prev(self, shuffle=False):
        images, labels = self.load_raw(old_or_new='old')
        if shuffle:
            p = np.random.permutation(len(images))
            images = images[p]
            labels = labels[p]
        rehearsal_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return rehearsal_dataset

    def load_raw(self, old_or_new):
        if old_or_new == 'old':
            exemplars = pickle.load(open(self._get_old_exemplar_file(), 'rb'))
        elif old_or_new == 'new':
            exemplars = pickle.load(open(self._get_exemplar_file(), 'rb'))
        else:
            raise Exception()

        images = np.concatenate(exemplars)
        labels = sum([len(exems) * [i] for i, exems in enumerate(exemplars)], [])
        labels = np.array(labels, dtype=np.int32)
        return images, labels

    def save(self, model, class_inc):

        # check exist
        if os.path.exists(self._get_exemplar_file()):
            return

        if class_inc.group_idx > 0:
            assert os.path.exists(self._get_old_exemplar_file())
            old_exemplars = pickle.load(open(self._get_old_exemplar_file(), 'rb'))
        else:
            old_exemplars = []

        old_classes_idx = range(len(class_inc.old_wnids))
        new_classes_idx = range(len(class_inc.old_wnids), len(class_inc.cumul_wnids))
        all_classes_idx = range(len(class_inc.cumul_wnids))

        if self.args.fixed_budget == 'each':
            num_exemplars_per_class = self.args.num_exemplars
        elif self.args.fixed_budget == 'total':
            num_exemplars_per_class = self.args.num_exemplars * len(class_inc.wnid_order) // len(all_classes_idx)
        else:
            raise Exception()

        backbone = tf.keras.Model(model.input, model.layers[self.args.feat_layer_idx].output)

        # update old exemplars
        if len(old_classes_idx) > 0:
            updated_exemplars = []
            for old_class_idx in old_classes_idx:
                updated_exemplars.append(old_exemplars[old_class_idx][:num_exemplars_per_class])

            # convert to ndarray
            old_exemplars = updated_exemplars

        # get new exemplars
        images_new_classes = class_inc.train_dataset.filter(
            lambda img, label: tf.reduce_any(label == np.array(new_classes_idx)))
        images_new_classes = np.stack(list(images_new_classes))

        for new_class_idx in tqdm(new_classes_idx, desc='Select exemplars'):
            images_cur_class = np.stack(images_new_classes[images_new_classes[:, 1] == new_class_idx, 0])

            # if num_exemplars_per_class is larger than all samples of the certain class, then save all
            if num_exemplars_per_class >= len(images_cur_class):
                old_exemplars.append(images_cur_class)
                continue

            if self.args.selection_strategy == 'random':
                exemplar_indices_cur_class = np.random.choice(range(len(images_cur_class)),
                                                              num_exemplars_per_class,
                                                              replace=False)
                old_exemplars.append(images_cur_class[exemplar_indices_cur_class])
            else:
                raise Exception('Invalid option')

            # print('Class %d complete' % (new_class_idx + 1))

        # save exemplars
        pickle.dump(old_exemplars, open(self._get_exemplar_file(), 'wb'))
        if self.args.show_exemplars:
            for class_idx in all_classes_idx:
                EXEMPLARS_FOLDER = os.path.join(self.args.SUB_OUTPUT_FOLDER, 'exemplars', 'class_%d' % (class_idx + 1))
                if not os.path.exists(EXEMPLARS_FOLDER):
                    os.makedirs(EXEMPLARS_FOLDER)
                exemplars_cur_class = old_exemplars[class_idx]
                exemplars_cur_class_restored = imagenet_preprocessing.restore_images(exemplars_cur_class)
                for exemplar_idx, exemplar in enumerate(exemplars_cur_class_restored):
                    Image.fromarray(exemplar).save(os.path.join(EXEMPLARS_FOLDER, '%d.jpg' % (exemplar_idx + 1)))
