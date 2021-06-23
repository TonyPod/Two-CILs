# -*- coding:utf-8 -*-  

""" 
@time: 11/30/20 10:28 PM 
@author: Chen He 
@site:  
@file: trainer.py
@description:  
"""
import os
import time

import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import psutil
import tensorflow as tf
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss
from sklearn.utils import class_weight
from tqdm import tqdm

from utils import evaluator
from utils.bias_rectifier import bias_rect
from utils.folder_utils import gen_base_param_str, gen_dataset_str
from utils.utils_func import get_top5_acc, MySummary, get_folder_size
from utils.utils_func import lr_scheduler
from utils.utils_func import wandb_log


def train(args, model, old_model, class_inc, memory, stats):
    # start timing
    inc_start_time = time.process_time()

    # check model exist
    FIRST_STAGE_CHECKPOINT_FOLDER = os.path.join(args.CHECKPOINT_FOLDER, '1st_stage')
    if os.path.exists(os.path.join(FIRST_STAGE_CHECKPOINT_FOLDER, 'final.index')):
        print('Model exist. Skipping...')
        model.load_weights(os.path.join(FIRST_STAGE_CHECKPOINT_FOLDER, 'final'))

        print('Start testing...')
        test_only = True
    elif not args.no_skip_base_training and class_inc.group_idx == 0 and not args.base_model:

        dataset_str = gen_dataset_str(args)
        base_model_dir = 'result/%s/base_%d/seed_%d/' % (dataset_str, args.base_cl, args.random_seed)
        params_str = gen_base_param_str(args)

        base_model_sub_dir = os.path.join(base_model_dir, params_str, 'group_1', 'checkpoints', '1st_stage')

        if not os.path.exists(base_model_sub_dir):
            raise Exception('Base model not found: %s' % base_model_sub_dir)

        print('Using base model: %s' % base_model_sub_dir)
        model.load_weights(os.path.join(base_model_sub_dir, 'final'))

        print('Loading statistics of base model...')
        stats.load(os.path.join(base_model_dir, params_str, 'stats.pkl'))

        print('Start testing...')
        test_only = True
    else:
        print('Model not exist. Start training...')
        test_only = False

    # load old memory and combine with new data
    train_dataset = class_inc.train_dataset
    rehearsal_dataset = None
    if args.memory_type == 'episodic':
        if class_inc.group_idx > 0 and memory is not None:
            rehearsal_dataset = memory.load_prev()
            num_samples_old_classes = tf.data.experimental.cardinality(rehearsal_dataset)
            num_samples_per_old_class = num_samples_old_classes / len(class_inc.old_wnids)
            num_samples_new_classes = tf.data.experimental.cardinality(train_dataset)
            num_samples_per_new_class = num_samples_new_classes / len(class_inc.cur_wnids)

            train_dataset = train_dataset.concatenate(rehearsal_dataset)

            if args.bias_rect in ['smote', 'adasyn', 'random_oversampling', 'kmeans', 'kmedoids',
                                  'random_undersampling',
                                  'near_miss_1', 'near_miss_2', 'near_miss_3', 'smote_tomek', 'smote_enn']:
                X, y = zip(*list(train_dataset))
                X = tf.stack(X)
                X_flatten = tf.reshape(X, [X.shape[0], -1])
                start_time = time.time()
                if args.bias_rect == 'smote':
                    X_flatten_resampled, y_resampled = SMOTE(n_jobs=16).fit_resample(X_flatten, y)
                elif args.bias_rect == 'adasyn':  # sampling_strategy='ratio' does not work in Group ImageNet
                    X_flatten_resampled, y_resampled = ADASYN(n_jobs=16, sampling_strategy='minority').fit_resample(
                        X_flatten, y)
                elif args.bias_rect == 'random_oversampling':
                    X_flatten_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X_flatten, y)
                elif args.bias_rect == 'kmeans':
                    X_flatten_resampled, y_resampled = ClusterCentroids(random_state=0).fit_resample(X_flatten, y)
                elif args.bias_rect == 'kmedoids':
                    from pyclustering.cluster.kmedoids import kmedoids
                    from collections import Counter

                    backbone = tf.keras.Model(model.input, model.layers[args.feat_layer_idx].output)
                    y_flatten = tf.stack(y)
                    k = min(list(Counter(y_flatten.numpy()).values()))
                    X_flatten_resampled = []
                    for class_idx in range(model.output_shape[-1]):
                        X_ = X[y_flatten == class_idx].numpy()
                        X_feats = backbone.predict(X_)
                        kmedoids_instance = kmedoids(X_feats, np.random.choice(len(X_feats), k, replace=False))
                        kmedoids_instance.process()
                        X_flatten_resampled.extend(X_[kmedoids_instance.get_medoids()])
                    y_resampled = np.repeat(range(model.output_shape[-1]), k).astype(np.int32)
                    X_flatten_resampled = np.stack(X_flatten_resampled)
                elif args.bias_rect == 'random_undersampling':
                    X_flatten_resampled, y_resampled = RandomUnderSampler(random_state=0).fit_resample(X_flatten, y)
                elif args.bias_rect == 'near_miss_1':
                    X_flatten_resampled, y_resampled = NearMiss(version=1).fit_resample(X_flatten, y)
                elif args.bias_rect == 'near_miss_2':
                    X_flatten_resampled, y_resampled = NearMiss(version=2).fit_resample(X_flatten, y)
                elif args.bias_rect == 'near_miss_3':
                    X_flatten_resampled, y_resampled = NearMiss(version=3).fit_resample(X_flatten, y)
                elif args.bias_rect == 'smote_tomek':
                    X_flatten_resampled, y_resampled = SMOTETomek(random_state=0).fit_resample(X_flatten, y)
                elif args.bias_rect == 'smote_enn':
                    X_flatten_resampled, y_resampled = SMOTEENN(random_state=0).fit_resample(X_flatten, y)
                else:
                    raise Exception()
                print('Time for %s: %.2f' % (args.bias_rect, (time.time() - start_time)))

                X_resampled = tf.reshape(X_flatten_resampled, [-1] + list(X.shape[1:]))
                train_dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))

            if args.bias_rect == 'bic':
                num_skip_samples = int(np.floor(args.val_exemplars_ratio * num_samples_per_old_class))
                for class_idx in range(len(class_inc.cumul_wnids)):
                    if class_idx == 0:
                        bic_train_dataset = train_dataset.filter(lambda img, label: tf.equal(label, class_idx)).skip(
                            num_skip_samples)
                    else:
                        bic_train_dataset = bic_train_dataset.concatenate(
                            train_dataset.filter(lambda img, label: tf.equal(label, class_idx)).skip(num_skip_samples))

        if args.bias_rect == 'undersampling' and class_inc.group_idx > 0:
            nb_train_samples = tf.cast(num_samples_per_old_class * len(class_inc.cumul_wnids), tf.int64)
        else:
            nb_train_samples = tf.data.experimental.cardinality(train_dataset)
        if args.bias_rect == 'bic' and class_inc.group_idx > 0:
            nb_train_samples -= num_skip_samples * len(class_inc.cumul_wnids)
            train_dataset = bic_train_dataset
    else:
        print('No memory is leveraged!')
        nb_train_samples = tf.data.experimental.cardinality(train_dataset)

    # calculate sample weights
    if args.bias_rect == 'reweighting':
        all_labels = np.array(list(train_dataset.map(lambda img, label: label)))
        class_weights = tf.convert_to_tensor(class_weight.compute_class_weight('balanced', np.unique(all_labels),
                                                                               all_labels).astype(np.float32))
    else:
        class_weights = tf.convert_to_tensor(np.array([1.] * len(class_inc.cumul_wnids), np.float32))

    # dataset processing
    train_dataset = train_dataset.cache().shuffle(nb_train_samples)

    if not args.no_aug:
        train_dataset = train_dataset.map(map_func=class_inc.dataset.aug_fn,
                                          num_parallel_calls=args.AUTOTUNE)
    train_dataset = train_dataset.batch(args.batch_size).prefetch(args.AUTOTUNE)

    # train model
    if not test_only:

        # losses
        if args.sigmoid:
            loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        lr = tf.Variable(args.base_lr)
        if args.optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif args.optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            raise Exception()
        train_loss_logger = tf.keras.metrics.Mean(name='train_loss')
        ce_loss_logger = tf.keras.metrics.Mean(name='ce_loss')
        reg_loss_logger = tf.keras.metrics.Mean(name='reg_loss')
        train_acc_logger = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

        if args.reg_type == 'lwf':
            lwf_loss_logger = tf.keras.metrics.Mean(name='lwf_loss')
            if args.sigmoid:
                lwf_loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                lwf_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        @tf.function
        def train_step(images, labels, weights):

            with tf.GradientTape() as tape:
                loss = 0.0

                # cross entropy
                logits = model(images)
                if args.sigmoid:
                    ce_loss = loss_obj(tf.one_hot(labels, len(class_inc.cumul_wnids)), logits, sample_weight=weights)
                else:
                    ce_loss = loss_obj(labels, logits, sample_weight=weights)

                if args.reg_type == 'lwf' and args.adjust_lwf_w:
                    ce_loss = ce_loss * len(class_inc.cur_wnids) / len(class_inc.cumul_wnids)

                loss += ce_loss

                # regularizer
                reg_loss = tf.add_n(model.losses)
                if args.ada_weight_decay and not args.base_model:
                    reg_loss = reg_loss * class_inc.nb_groups / (class_inc.group_idx + 1) / 10
                loss += reg_loss

                # learning without forgetting
                if args.reg_type == 'lwf':
                    lwf_loss = 0.0
                    if class_inc.group_idx > 0:
                        if args.sigmoid:
                            distill_gt = tf.nn.sigmoid(old_model(images) / args.lwf_loss_temp)
                        else:
                            distill_gt = tf.nn.softmax(old_model(images) / args.lwf_loss_temp)
                        distill_logits = logits[:, :len(class_inc.old_wnids)] / args.lwf_loss_temp
                        lwf_loss += args.reg_loss_weight * lwf_loss_obj(distill_gt, distill_logits)

                    if args.reg_type == 'lwf' and args.adjust_lwf_w:
                        lwf_loss = lwf_loss * len(class_inc.old_wnids) / len(class_inc.cumul_wnids)
                    loss += lwf_loss

            # optimize
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            # log losses
            train_loss_logger(loss)
            ce_loss_logger(ce_loss)
            reg_loss_logger(reg_loss)
            train_acc_logger(labels, logits)
            if args.reg_type == 'lwf':
                lwf_loss_logger(lwf_loss)

        # init vars
        global_step = tf.Variable(0)
        lr.assign(args.base_lr)
        my_summary = MySummary()

        lrs = lr_scheduler(args, class_inc)
        for epoch, cur_lr in enumerate(lrs):

            if not np.float32(cur_lr) == lr.numpy():
                print('New learning rate: %f' % cur_lr)
            lr.assign(cur_lr)

            # training
            tf.keras.backend.set_learning_phase(True)
            starttime = time.time()
            for batch_images, batch_labels in train_dataset:
                global_step.assign_add(1)
                weights = tf.gather(class_weights, batch_labels)
                train_step(batch_images, batch_labels, weights)
                # with writer.as_default():
                #     tf.summary.trace_on(profiler=True)
                #     train_step(batch_images, batch_labels, weights)
                #     tf.summary.trace_export(name='my_trace', step=0, profiler_outdir=logdir)

                step = int(global_step.numpy())
                my_summary.add('train_loss', step, train_loss_logger.result())
                my_summary.add('ce_loss', step, ce_loss_logger.result())
                my_summary.add('reg_loss', step, reg_loss_logger.result())
                my_summary.add('train_acc', step, train_acc_logger.result())
                if args.reg_type == 'lwf':
                    my_summary.add('lwf_loss', step, lwf_loss_logger.result())

            # testing
            if epoch < 5 or (epoch + 1) % 5 == 0:
                test_time = time.time()
                tf.keras.backend.set_learning_phase(False)
                test_logits = model.predict(class_inc.test_images_now)
                test_scores = tf.nn.sigmoid(test_logits) if args.sigmoid else tf.nn.softmax(test_logits)
                test_scores = test_scores.numpy()
                test_preds = tf.argmax(test_scores, axis=1, output_type=tf.int32)
                test_conf_mat = tf.math.confusion_matrix(class_inc.test_labels_now, test_preds).numpy()
                test_acc_per_class = np.diag(test_conf_mat) * 100. / np.sum(test_conf_mat, axis=1)
                test_acc_avg = np.mean(test_acc_per_class)

                # top-5 accuracy
                top5_acc = get_top5_acc(test_scores, class_inc.test_labels_now)

                lwf_str = ''
                if args.reg_type == 'lwf':
                    lwf_str = 'lwf %.4f, ' % lwf_loss_logger.result()
                print(
                    'Epoch %d: Loss %.4f (ce %.4f, %sreg %.4f), Train Acc %.2f, Test Acc %.2f, Top-5 Acc %.2f, Time %.2f (%.2f)' % (
                        epoch + 1,
                        train_loss_logger.result(),
                        ce_loss_logger.result(),
                        lwf_str,
                        reg_loss_logger.result(),
                        train_acc_logger.result() * 100.,
                        test_acc_avg,
                        top5_acc,
                        time.time() - starttime,
                        time.time() - test_time))

            train_loss_logger.reset_states()
            ce_loss_logger.reset_states()
            reg_loss_logger.reset_states()
            train_acc_logger.reset_states()

        my_summary.vis(os.path.join(args.SUB_OUTPUT_FOLDER, 'loss'))
        my_summary.reset()

    # save model
    if not test_only or not os.path.exists(os.path.join(FIRST_STAGE_CHECKPOINT_FOLDER, 'final.index')):
        model.save_weights(os.path.join(FIRST_STAGE_CHECKPOINT_FOLDER, 'final'))

    # evaluate model before bias rectification
    top1_acc, top5_acc, harmonic_mean = evaluator.evaluate(model, class_inc, args)
    wandb_log(args, {'top1_acc_before': top1_acc, 'top5_acc_before': top5_acc, 'harmonic_mean_before': harmonic_mean})
    if class_inc.is_final_inc:
        wandb_log(args, {'final_top1_acc_before': top1_acc, 'final_top5_acc_before': top5_acc,
                         'final_harmonic_mean_before': harmonic_mean})

    # store the lambda for future use
    if args.bias_rect == 'weight_aligning_no_bias' and class_inc.group_idx == class_inc.nb_groups - 1:
        fc_weights = model.get_layer('fc').trainable_variables[0].numpy()
        fc_weights_norms = np.linalg.norm(fc_weights, axis=0)
        old_mean, new_mean = np.mean(fc_weights_norms[:len(class_inc.old_wnids)]), np.mean(
            fc_weights_norms[len(class_inc.old_wnids):])
        lamda = old_mean / new_mean
        args.lamda = lamda

    # check second model exists
    SECOND_STAGE_CHECKPOINT_FOLDER = os.path.join(args.CHECKPOINT_FOLDER, '2nd_stage')
    if os.path.exists(os.path.join(SECOND_STAGE_CHECKPOINT_FOLDER, 'final.index')):
        print('Model exist. Skipping...')
        model.load_weights(os.path.join(SECOND_STAGE_CHECKPOINT_FOLDER, 'final'))

        print('Start testing...')
        test_only_2nd_stage = True
    else:
        print('Model not exist. Start training...')
        test_only_2nd_stage = False

    # bias rectification
    if not test_only_2nd_stage:
        bias_rect(args, model, old_model, class_inc, rehearsal_dataset)

        # save model
        model.save_weights(os.path.join(SECOND_STAGE_CHECKPOINT_FOLDER, 'final'))

    # save memory
    if args.memory_type == 'episodic':
        memory.save(model, class_inc)

    # evaluate model after bias rectification (or without bias rectification)
    top1_acc, top5_acc, harmonic_mean = evaluator.evaluate(model, class_inc, args, STAGE_FOLDER='2nd_stage',
                                                           use_embedding=True if args.embedding else False,
                                                           memory=memory if args.embedding else None)

    wandb_log(args, {'top1_acc': top1_acc, 'top5_acc': top5_acc, 'harmonic_mean': harmonic_mean})
    if class_inc.is_final_inc:
        wandb_log(args, {'final_top1_acc': top1_acc, 'final_top5_acc': top5_acc, 'final_harmonic_mean': harmonic_mean})

    # update stats
    ckpt_file = os.path.join(SECOND_STAGE_CHECKPOINT_FOLDER)

    # end timing
    inc_stop_time = time.process_time()

    # update statistics
    stats.put('time', class_inc.group_idx, inc_stop_time - inc_start_time)
    stats.put('rehearsal_memory_size', class_inc.group_idx, memory.size() if memory is not None else 0)
    stats.put('model_size', class_inc.group_idx, get_folder_size(ckpt_file))
    stats.put('internal_memory', class_inc.group_idx, psutil.Process(os.getpid()).memory_info().rss)
    stats.save()

    # tf.profiler.experimental.stop()
    calc_stats(args, class_inc, model, rehearsal_dataset, stage_str='2nd_stage')


def calc_stats(args, class_inc, model, rehearsal_dataset, stage_str='1st_stage'):
    if args.stat_boundary and class_inc.group_idx == class_inc.nb_groups - 1:
        import pandas
        import matplotlib
        matplotlib.use('Agg')

        STAT_PLOT_DIR = os.path.join(args.SUB_RESULT_FOLDER, stage_str, 'stat')
        if not os.path.exists(STAT_PLOT_DIR):
            os.makedirs(STAT_PLOT_DIR)

        backbone = tf.keras.Model(model.input, model.layers[args.feat_layer_idx].output)

        train_feats_now = backbone.predict(
            class_inc.train_dataset.concatenate(rehearsal_dataset).batch(args.batch_size).prefetch(args.AUTOTUNE))
        train_labels_now = tf.stack(
            list(class_inc.train_dataset.concatenate(rehearsal_dataset).map(lambda img, label: label)))

        old_classes = list(range(len(class_inc.old_wnids)))
        new_classes = list(range(len(class_inc.old_wnids), len(class_inc.cumul_wnids)))
        all_classes = list(range(len(class_inc.cumul_wnids)))

        train_cumul_dataset = class_inc.dataset.load_train(class_inc.cumul_wnids, class_inc.wnid_order, num_samples=-1)
        train_feats_cumul = backbone.predict(train_cumul_dataset.batch(args.batch_size).prefetch(args.AUTOTUNE))
        train_labels_cumul = tf.stack(list(train_cumul_dataset.map(lambda img, label: label)))

        test_feats_now = backbone.predict(class_inc.test_images_now)

        short_names = [n.split(', ')[0] for n in class_inc.name_order]

        if args.bias_rect == 'post_scaling':

            if args.geometric_view:
                PLOT_DIR = os.path.join(args.SUB_RESULT_FOLDER, 'geometric_view')
                if not os.path.exists(PLOT_DIR):
                    os.makedirs(PLOT_DIR)

            train_dist_mat_now = np.zeros([len(class_inc.cumul_wnids), len(class_inc.cumul_wnids)])
            train_dist_mat_cumul = np.zeros([len(class_inc.cumul_wnids), len(class_inc.cumul_wnids)])
            test_dist_mat_now = np.zeros([len(class_inc.cumul_wnids), len(class_inc.cumul_wnids)])

            train_var_vec_now = np.zeros([len(class_inc.cumul_wnids)])
            train_var_vec_cumul = np.zeros([len(class_inc.cumul_wnids)])
            test_var_vec_now = np.zeros([len(class_inc.cumul_wnids)])

            last_fc_w = model.get_layer('fc').variables[0].numpy()
            last_fc_b = model.get_layer('fc').variables[1].numpy()

            for class_a in tqdm(range(len(class_inc.cumul_wnids))):
                class_a_w = last_fc_w[:, class_a]
                class_a_b = last_fc_b[class_a]

                train_feats_now_class_a = train_feats_now[train_labels_now == class_a]
                train_feats_cumul_class_a = train_feats_cumul[train_labels_cumul == class_a]
                test_feats_class_a = test_feats_now[class_inc.test_labels_now == class_a]

                train_var_vec_now[class_a] = np.std(train_feats_now_class_a)
                train_var_vec_cumul[class_a] = np.std(train_feats_cumul_class_a)
                test_var_vec_now[class_a] = np.std(test_feats_class_a)

                for class_b in range(class_a):

                    if args.geometric_view:
                        # check dst file exists
                        if class_a in old_classes and class_b in old_classes:
                            SUB_VIOLIB_PLOT_DIR = os.path.join(PLOT_DIR, 'old_old')
                        elif class_a in new_classes and class_b in old_classes:
                            SUB_VIOLIB_PLOT_DIR = os.path.join(PLOT_DIR, 'new_old')
                        elif class_a in new_classes and class_b in new_classes:
                            SUB_VIOLIB_PLOT_DIR = os.path.join(PLOT_DIR, 'new_new')
                        else:
                            raise Exception()
                        if not os.path.exists(SUB_VIOLIB_PLOT_DIR):
                            os.makedirs(SUB_VIOLIB_PLOT_DIR)

                        violin_graph_file = os.path.join(SUB_VIOLIB_PLOT_DIR, 'violin_%d-%d.pdf' % (class_a, class_b))
                        # if os.path.exists(violin_graph_file):
                        #     print('File existings: %s' % violin_graph_file)
                        #     continue

                    class_b_w = last_fc_w[:, class_b]
                    class_b_b = last_fc_b[class_b]

                    rect_values = model.layers[-1].variables[0].numpy()
                    bias = (rect_values[class_b] - rect_values[class_a])

                    # train cumul
                    train_feats_cumul_class_b = train_feats_cumul[train_labels_cumul == class_b]
                    dist_a = ((np.matmul(train_feats_cumul_class_a, np.expand_dims(class_a_w - class_b_w,
                                                                                   1)) + class_a_b - class_b_b - bias) / np.linalg.norm(
                        class_a_w - class_b_w))
                    dist_b = ((np.matmul(train_feats_cumul_class_b, np.expand_dims(class_a_w - class_b_w,
                                                                                   1)) + class_a_b - class_b_b - bias) / np.linalg.norm(
                        class_a_w - class_b_w))

                    df1 = pandas.DataFrame(data=zip(np.concatenate((np.squeeze(dist_a), np.squeeze(dist_b))),
                                                    ['%d_%s' % (class_a, short_names[class_a])] * len(dist_a) + [
                                                        '%d_%s' % (class_b, short_names[class_b])] * len(dist_b),
                                                    ['training (all)'] * (len(dist_a) + len(dist_b))),
                                           index=range(len(dist_a) + len(dist_b)), columns=['dist', 'class', 'split'])

                    train_dist_mat_cumul[class_a][class_b] = np.abs(np.mean(dist_a)) / (
                            np.abs(np.mean(dist_a)) + np.abs(np.mean(dist_b)))
                    train_dist_mat_cumul[class_b][class_a] = np.abs(np.mean(dist_b)) / (
                            np.abs(np.mean(dist_a)) + np.abs(np.mean(dist_b)))

                    # train now
                    train_feats_class_b = train_feats_now[train_labels_now == class_b]
                    dist_a = ((np.matmul(train_feats_now_class_a, np.expand_dims(class_a_w - class_b_w,
                                                                                 1)) + class_a_b - class_b_b - bias) / np.linalg.norm(
                        class_a_w - class_b_w))
                    dist_b = ((np.matmul(train_feats_class_b, np.expand_dims(class_a_w - class_b_w,
                                                                             1)) + class_a_b - class_b_b - bias) / np.linalg.norm(
                        class_a_w - class_b_w))

                    df2 = pandas.DataFrame(data=zip(np.concatenate((np.squeeze(dist_a), np.squeeze(dist_b))),
                                                    ['%d_%s' % (class_a, short_names[class_a])] * len(dist_a) + [
                                                        '%d_%s' % (class_b, short_names[class_b])] * len(dist_b),
                                                    ['training'] * (len(dist_a) + len(dist_b))),
                                           index=range(len(dist_a) + len(dist_b)), columns=['dist', 'class', 'split'])

                    train_dist_mat_now[class_a][class_b] = np.abs(np.mean(dist_a)) / (
                            np.abs(np.mean(dist_a)) + np.abs(np.mean(dist_b)))
                    train_dist_mat_now[class_b][class_a] = np.abs(np.mean(dist_b)) / (
                            np.abs(np.mean(dist_a)) + np.abs(np.mean(dist_b)))

                    # test
                    test_feats_class_b = test_feats_now[class_inc.test_labels_now == class_b]
                    dist_a = (
                            (np.matmul(test_feats_class_a, np.expand_dims(class_a_w - class_b_w,
                                                                          1)) + class_a_b - class_b_b - bias) / np.linalg.norm(
                        class_a_w - class_b_w))
                    dist_b = (
                            (np.matmul(test_feats_class_b, np.expand_dims(class_a_w - class_b_w,
                                                                          1)) + class_a_b - class_b_b - bias) / np.linalg.norm(
                        class_a_w - class_b_w))
                    test_dist_mat_now[class_a][class_b] = np.abs(np.mean(dist_a)) / (
                            np.abs(np.mean(dist_a)) + np.abs(np.mean(dist_b)))
                    test_dist_mat_now[class_b][class_a] = np.abs(np.mean(dist_b)) / (
                            np.abs(np.mean(dist_a)) + np.abs(np.mean(dist_b)))

                    df3 = pandas.DataFrame(data=zip(np.concatenate((np.squeeze(dist_a), np.squeeze(dist_b))),
                                                    ['%d_%s' % (class_a, short_names[class_a])] * len(dist_a) + [
                                                        '%d_%s' % (class_b, short_names[class_b])] * len(dist_b),
                                                    ['test'] * (len(dist_a) + len(dist_b))),
                                           index=range(len(dist_a) + len(dist_b)), columns=['dist', 'class', 'split'])
                    df = pandas.concat([df1, df2, df3])

                    if args.geometric_view:
                        if class_a in new_classes and class_b in old_classes:
                            plt.figure()
                            # for i in [-10, -5, 5, 10]:
                            #     plt.hlines(i, -1, 3, colors='gray', linestyles='dashed', linewidth=1)
                            plt.hlines(0, -1, 3, colors='r', linestyles='solid')

                            new_decision_boundary = bias / np.linalg.norm(class_a_w - class_b_w)
                            plt.hlines(new_decision_boundary, -1, 3, colors='r', linestyles='dashed')

                            sns.violinplot(data=df, x='split', y='dist', hue='class', palette='muted', bw=.2,
                                           split=True,
                                           inner='point')
                            plt.savefig(violin_graph_file)
                            plt.close()

            with open(os.path.join(STAT_PLOT_DIR, 'train_feats_std.txt'), 'w') as f1, open(
                    os.path.join(STAT_PLOT_DIR, 'test_feats_std.txt'), 'w') as f2, open(
                os.path.join(STAT_PLOT_DIR, 'train_feats_cumul_std.txt'), 'w') as f3:
                for class_a in range(len(class_inc.cumul_wnids)):
                    f1.write(str(train_var_vec_now[class_a]) + os.linesep)
                    f2.write(str(test_var_vec_now[class_a]) + os.linesep)
                    f3.write(str(train_var_vec_cumul[class_a]) + os.linesep)
            with open(os.path.join(STAT_PLOT_DIR, 'train_boundary_dist_ratio.txt'), 'w') as f1, open(
                    os.path.join(STAT_PLOT_DIR, 'test_boundary_dist_ratio.txt'), 'w') as f2, open(
                os.path.join(STAT_PLOT_DIR, 'train_boundary_dist_ratio_cumul.txt'), 'w') as f3:
                for class_a in range(len(class_inc.cumul_wnids)):
                    f1.write('\t'.join([str(elem) for elem in train_dist_mat_now[class_a]]) + os.linesep)
                    f2.write('\t'.join([str(elem) for elem in test_dist_mat_now[class_a]]) + os.linesep)
                    f3.write('\t'.join([str(elem) for elem in train_dist_mat_cumul[class_a]]) + os.linesep)

        elif args.bias_rect == 'weight_aligning_no_bias':

            def distill(p, T):
                distilled_p = np.power(p, 1.0 / T)
                return distilled_p / np.sum(distilled_p)

            T = 3
            width = np.deg2rad(3)

            PLOT_DIR = os.path.join(args.SUB_RESULT_FOLDER, 'polar')
            if not os.path.exists(PLOT_DIR):
                os.makedirs(PLOT_DIR)

            last_fc_w = model.get_layer('fc').variables[0].numpy()

            for class_a in tqdm(range(len(class_inc.cumul_wnids))):
                class_a_w = last_fc_w[:, class_a]

                train_feats_now_class_a = train_feats_now[train_labels_now == class_a]
                train_feats_cumul_class_a = train_feats_cumul[train_labels_cumul == class_a]
                test_feats_class_a = test_feats_now[class_inc.test_labels_now == class_a]

                for class_b in range(class_a):

                    if class_a in new_classes and class_b in old_classes:

                        # check dst file exists
                        SUB_POLAR_CHART_DIR = os.path.join(PLOT_DIR, 'new_old')
                        if not os.path.exists(SUB_POLAR_CHART_DIR):
                            os.makedirs(SUB_POLAR_CHART_DIR)

                        polar_graph_file = os.path.join(SUB_POLAR_CHART_DIR, 'polar_%d-%d.pdf' % (class_a, class_b))

                        class_b_w = last_fc_w[:, class_b]

                        beta = np.arccos(
                            class_a_w.dot(class_b_w) / (np.linalg.norm(class_a_w) * np.linalg.norm(class_b_w)))

                        e1 = class_b_w / np.linalg.norm(class_b_w)
                        e2 = class_a_w - e1.dot(class_a_w) * e1
                        e2 = e2 / np.linalg.norm(e2)

                        old_decision_boundary = np.arctan(
                            args.lamda * np.linalg.norm(class_a_w) / np.linalg.norm(class_b_w) / np.sin(
                                beta) - 1 / np.tan(beta))
                        new_decision_boundary = np.arctan(
                            np.linalg.norm(class_a_w) / np.linalg.norm(class_b_w) / np.sin(beta) - 1 / np.tan(beta))

                        fig = plt.figure(figsize=(18, 6))

                        # train cumul
                        train_feats_cumul_class_b = train_feats_cumul[train_labels_cumul == class_b]
                        theta_a = np.arctan2(train_feats_cumul_class_a.dot(e2), train_feats_cumul_class_a.dot(e1))
                        theta_b = np.arctan2(train_feats_cumul_class_b.dot(e2), train_feats_cumul_class_b.dot(e1))

                        df1 = pandas.DataFrame(data=zip(np.concatenate((np.squeeze(theta_a), np.squeeze(theta_b))),
                                                        ['%d_%s' % (class_a, short_names[class_a])] * len(theta_a) + [
                                                            '%d_%s' % (class_b, short_names[class_b])] * len(theta_b),
                                                        ['training (all)'] * (len(theta_a) + len(theta_b))),
                                               index=range(len(theta_a) + len(theta_b)),
                                               columns=['theta', 'class', 'split'])

                        a, b = np.histogram(np.rad2deg(theta_a + np.pi), bins=np.arange(0, 361, 1))
                        centers = np.deg2rad(b)[:-1]
                        ax = fig.add_subplot(131, projection='polar')
                        ax.set_rlim(0, 0.05)
                        ax.title.set_text('training (all)')
                        ax.set_yticklabels([])
                        ax.scatter(theta_a + np.pi, np.repeat(0.049, len(theta_a)), color='#DA8B5F', s=1)
                        ax.bar(centers, distill(a / np.sum(a), T), width=width, bottom=0.0, color='#DA8B5F', alpha=0.5)
                        ax.set_theta_zero_location("E")
                        ax.set_theta_direction(1)

                        a, b = np.histogram(np.rad2deg(theta_b + np.pi), bins=np.arange(0, 361, 1))
                        ax.scatter(theta_b + np.pi, np.repeat(0.049, len(theta_b)), color='#597DBF', s=1)
                        centers = np.deg2rad(b)[:-1]
                        ax.bar(centers, distill(a / np.sum(a), T), width=width, bottom=0.0, color='#597DBF', alpha=0.5)

                        line_range = np.arange(0, 1, 0.01)
                        ax.plot(np.repeat(old_decision_boundary, len(line_range)), line_range, 'r-')
                        ax.plot(np.repeat(new_decision_boundary, len(line_range)), line_range, 'r--')
                        ax.plot(np.repeat(np.pi + old_decision_boundary, len(line_range)), line_range, 'r-')
                        ax.plot(np.repeat(np.pi + new_decision_boundary, len(line_range)), line_range, 'r--')

                        # train now
                        train_feats_now_class_b = train_feats_now[train_labels_now == class_b]
                        theta_a = np.arctan2(train_feats_now_class_a.dot(e2), train_feats_now_class_a.dot(e1))
                        theta_b = np.arctan2(train_feats_now_class_b.dot(e2), train_feats_now_class_b.dot(e1))

                        df2 = pandas.DataFrame(data=zip(np.concatenate((np.squeeze(theta_a), np.squeeze(theta_b))),
                                                        ['%d_%s' % (class_a, short_names[class_a])] * len(theta_a) + [
                                                            '%d_%s' % (class_b, short_names[class_b])] * len(theta_b),
                                                        ['training'] * (len(theta_a) + len(theta_b))),
                                               index=range(len(theta_a) + len(theta_b)),
                                               columns=['theta', 'class', 'split'])

                        a, b = np.histogram(np.rad2deg(theta_a + np.pi), bins=np.arange(0, 361, 1))
                        centers = np.deg2rad(b)[:-1]
                        ax = fig.add_subplot(132, projection='polar')
                        ax.set_rlim(0, 0.05)
                        ax.title.set_text('training')
                        ax.set_yticklabels([])
                        ax.scatter(theta_a + np.pi, np.repeat(0.049, len(theta_a)), color='#DA8B5F', s=1)
                        ax.bar(centers, distill(a / np.sum(a), T), width=width, bottom=0.0, color='#DA8B5F', alpha=0.5)
                        ax.set_theta_zero_location("E")
                        ax.set_theta_direction(1)

                        a, b = np.histogram(np.rad2deg(theta_b + np.pi), bins=np.arange(0, 361, 1))
                        centers = np.deg2rad(b)[:-1]
                        ax.scatter(theta_b + np.pi, np.repeat(0.049, len(theta_b)), color='#597DBF', s=1)
                        ax.bar(centers, distill(a / np.sum(a), T), width=width, bottom=0.0, color='#597DBF', alpha=0.5)

                        line_range = np.arange(0, 1, 0.01)
                        ax.plot(np.repeat(old_decision_boundary, len(line_range)), line_range, 'r-')
                        ax.plot(np.repeat(new_decision_boundary, len(line_range)), line_range, 'r--')
                        ax.plot(np.repeat(np.pi + old_decision_boundary, len(line_range)), line_range, 'r-')
                        ax.plot(np.repeat(np.pi + new_decision_boundary, len(line_range)), line_range, 'r--')

                        # test
                        test_feats_class_b = test_feats_now[class_inc.test_labels_now == class_b]
                        theta_a = np.arctan2(test_feats_class_a.dot(e2), test_feats_class_a.dot(e1))
                        theta_b = np.arctan2(test_feats_class_b.dot(e2), test_feats_class_b.dot(e1))

                        df3 = pandas.DataFrame(data=zip(np.concatenate((np.squeeze(theta_a), np.squeeze(theta_b))),
                                                        ['%d_%s' % (class_a, short_names[class_a])] * len(theta_a) + [
                                                            '%d_%s' % (class_b, short_names[class_b])] * len(theta_b),
                                                        ['test'] * (len(theta_a) + len(theta_b))),
                                               index=range(len(theta_a) + len(theta_b)),
                                               columns=['theta', 'class', 'split'])

                        a, b = np.histogram(np.rad2deg(theta_a + np.pi), bins=np.arange(0, 361, 1))
                        centers = np.deg2rad(b)[:-1]
                        ax = fig.add_subplot(133, projection='polar')
                        ax.set_rlim(0, 0.05)
                        ax.title.set_text('test')
                        ax.set_yticklabels([])
                        ax.scatter(theta_a + np.pi, np.repeat(0.049, len(theta_a)), color='#DA8B5F', s=1)
                        ax.bar(centers, distill(a / np.sum(a), T), width=width, bottom=0.0, color='#DA8B5F', alpha=0.5,
                               label='%d_%s' % (class_a, short_names[class_a]))
                        ax.set_theta_zero_location("E")
                        ax.set_theta_direction(1)

                        a, b = np.histogram(np.rad2deg(theta_b + np.pi), bins=np.arange(0, 361, 1))
                        centers = np.deg2rad(b)[:-1]
                        ax.scatter(theta_b + np.pi, np.repeat(0.049, len(theta_b)), color='#597DBF', s=1)
                        ax.bar(centers, distill(a / np.sum(a), T), width=width, bottom=0.0, color='#597DBF', alpha=0.5,
                               label='%d_%s' % (class_b, short_names[class_b]))

                        line_range = np.arange(0, 1, 0.01)
                        ax.plot(np.repeat(old_decision_boundary, len(line_range)), line_range, 'r-')
                        ax.plot(np.repeat(new_decision_boundary, len(line_range)), line_range, 'r--')
                        ax.plot(np.repeat(np.pi + old_decision_boundary, len(line_range)), line_range, 'r-')
                        ax.plot(np.repeat(np.pi + new_decision_boundary, len(line_range)), line_range, 'r--')
                        ax.legend()

                        plt.savefig(polar_graph_file)
        else:
            raise NotImplementedError
