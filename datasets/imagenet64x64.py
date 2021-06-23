import os
import random

import numpy as np
import scipy.io
import tensorflow as tf

from datasets.dataset import Dataset
from utils import imagenet_preprocessing as imagenet_preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE
DEVKIT_PATH = '/home/hechen/ILSVRC/ILSVRC2012_devkit_t12'
TRAIN_DIR = '/home/hechen/Datasets/ImageNet_64x64/train_classified'
TEST_DIR = '/home/hechen/Datasets/ImageNet_64x64/val_classified'


def _parse_devkit_meta():
    meta_mat = scipy.io.loadmat(os.path.join(DEVKIT_PATH, 'data/meta.mat'))
    labels_dic = dict(
        (m[0][1][0], m[0][0][0][0] - 1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names_dic = dict(
        (m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names = [tup[1] for tup in
                   sorted([(v, label_names_dic[k]) for k, v in labels_dic.items()], key=lambda x: x[0])]
    fval_ground_truth = open(os.path.join(DEVKIT_PATH, 'data/ILSVRC2012_validation_ground_truth.txt'), 'r')
    validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
    fval_ground_truth.close()

    return labels_dic, label_names, label_names_dic, validation_ground_truth


class ImageNet64x64(Dataset):
    NUM_CLASSES = 1000
    NUM_MAX_TRAIN_SAMPLES_PER_CLASS = 1300
    NUM_MAX_TEST_SAMPLES_PER_CLASS = 50

    def __init__(self, args):
        super().__init__(args)
        self.labels_dict, self.label_names, self.label_names_dict, self.validation_ground_truth = _parse_devkit_meta()

    def _map_fn(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.image.decode_image(img, channels=3)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.ensure_shape(img, (64, 64, 3))
        img = img - tf.broadcast_to(imagenet_preprocessing.CHANNEL_MEANS, tf.shape(img))
        return img, label

    def aug_fn(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.pad_to_bounding_box(image, 4, 4, 64 + 4 * 2, 64 + 4 * 2)
        image = tf.image.random_crop(image, [64, 64, 3])
        return image, label

    def preprocess(self, images, **kwargs):
        images = images.astype(np.float32)
        if images.shape[1] == 3:
            images = images.transpose((0, 2, 3, 1))
        images = images - np.array(imagenet_preprocessing.CHANNEL_MEANS, dtype=np.float32)
        return images

    def load_train(self, wnids, order_wnid, num_samples=-1, parallel_calls=True, random_selection=False,
                   use_shuffle=False):
        filenames = []
        labels = []
        for wnid in wnids:
            sub_dir = os.path.join(TRAIN_DIR, wnid)
            files_cur_cl = [os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir)]
            if num_samples > 0:
                if random_selection:
                    random_indices = list(range(len(files_cur_cl)))
                    random.shuffle(random_indices)
                    files_cur_cl = [files_cur_cl[idx] for idx in random_indices[:num_samples]]
                else:
                    files_cur_cl = files_cur_cl[:num_samples]
            filenames.extend(files_cur_cl)
            labels.extend([order_wnid.index(wnid)] * len(files_cur_cl))

        if use_shuffle:
            indices = list(range(len(filenames)))
            random.shuffle(indices)
            filenames = [filenames[i] for i in indices]
            labels = [labels[i] for i in indices]

        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        if parallel_calls:
            dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).map(self._map_fn,
                                                                                  num_parallel_calls=AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).map(self._map_fn)

        return dataset

    def load_test(self, wnids, order_wnid, num_samples=-1):
        filenames = []
        labels = []
        for wnid in wnids:
            sub_dir = os.path.join(TEST_DIR, wnid)
            files_cur_cl = [os.path.join(sub_dir, filename) for filename in os.listdir(sub_dir)]
            if num_samples > 0:
                files_cur_cl = files_cur_cl[:num_samples]
            filenames.extend(files_cur_cl)
            labels.extend([order_wnid.index(wnid)] * len(files_cur_cl))

        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).map(self._map_fn, num_parallel_calls=AUTOTUNE)
        return dataset
