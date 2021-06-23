# -*- coding:utf-8 -*-  

""" 
@time: 12/20/20 4:55 PM 
@author: Chen He 
@site:  
@file: bias_rectifier.py
@description:  
"""
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from utils.utils_func import MySummary, get_top5_acc


def bias_rect(args, model, old_model, class_inc, rehearsal_dataset):
    # skipping first phase
    if class_inc.group_idx == 0 and not args.bias_rect == 'il2m':
        return

    # start bias rectification
    if args.bias_rect == 'post_scaling':
        all_labels = np.array(
            list(class_inc.train_dataset.concatenate(rehearsal_dataset).map(lambda img, label: label)))
        correction_terms = []
        for class_idx in range(len(class_inc.cumul_wnids)):
            train_prior = np.sum(all_labels == class_idx) / len(all_labels)
            test_prior = 1 / len(class_inc.cumul_wnids)
            correction_terms.append(np.log(test_prior / train_prior))
        correction_terms = np.array(correction_terms, dtype=np.float32)
        prev_values = model.get_layer('ps_layer').variables[0].numpy()
        model.get_layer('ps_layer').variables[0].assign(prev_values + correction_terms)

    elif args.bias_rect == 'il2m':

        new_pkl_file = os.path.join(args.SUB_OUTPUT_FOLDER, 'il2m_memory.pkl')

        if not os.path.exists(new_pkl_file):

            if rehearsal_dataset is None:
                concat_train_dataset = class_inc.train_dataset
            else:
                concat_train_dataset = class_inc.train_dataset.concatenate(rehearsal_dataset)
            train_logits = model.predict(concat_train_dataset.map(lambda img, label: img).batch(args.batch_size))
            train_labels = np.stack(concat_train_dataset.map(lambda img, label: label))

            # load previous
            if class_inc.group_idx == 0:
                init_classes_means = {}
                current_classes_means = {}
                models_confidence = {}
            else:
                prev_pkl_file = os.path.join(args.PREV_SUBFOLDER, 'il2m_memory.pkl')
                assert os.path.exists(prev_pkl_file)
                pkl_data = pickle.load(open(prev_pkl_file, 'rb'))
                init_classes_means = pkl_data['init_classes_means']
                current_classes_means = pkl_data['current_classes_means']
                models_confidence = pkl_data['models_confidence']

            for class_idx in range(len(class_inc.old_wnids), len(class_inc.cumul_wnids)):
                init_classes_means[class_idx] = np.mean(train_logits[train_labels == class_idx][:, class_idx])
            for class_idx in range(len(class_inc.old_wnids)):
                current_classes_means[class_idx] = np.mean(train_logits[train_labels == class_idx][:, class_idx])
            models_confidence[class_inc.group_idx] = np.mean(
                np.max(train_logits[[i for i, label in enumerate(train_labels) if
                                     label in range(len(class_inc.old_wnids), len(class_inc.cumul_wnids))]], axis=1))

            # save new data
            pickle.dump({'init_classes_means': init_classes_means, 'current_classes_means': current_classes_means,
                         'models_confidence': models_confidence}, open(new_pkl_file, 'wb'))

        else:
            pkl_data = pickle.load(open(new_pkl_file, 'rb'))
            init_classes_means = pkl_data['init_classes_means']
            current_classes_means = pkl_data['current_classes_means']
            models_confidence = pkl_data['models_confidence']

        # start bias correction
        bias_mul_term = np.ones(len(class_inc.cumul_wnids))
        for old_class_idx in range(len(class_inc.old_wnids)):
            bias_mul_term[old_class_idx] = init_classes_means[old_class_idx] / current_classes_means[old_class_idx] * \
                                           models_confidence[class_inc.group_idx] / models_confidence[
                                               old_class_idx // args.nb_cl]
        model.get_layer('il2m_layer').variables[0].assign(bias_mul_term)


    elif args.bias_rect == 'weight_aligning_no_bias':

        fc_weights = model.get_layer('fc').trainable_variables[0].numpy()
        fc_weights_norms = np.linalg.norm(fc_weights, axis=0)
        old_mean, new_mean = np.mean(fc_weights_norms[:len(class_inc.old_wnids)]), np.mean(
            fc_weights_norms[len(class_inc.old_wnids):])
        lamda = old_mean / new_mean
        print('Lamda=%f' % lamda)

        corrected_fc_weights = fc_weights * np.array(
            [1.] * len(class_inc.old_wnids) + [lamda] * (len(class_inc.cumul_wnids) - len(class_inc.old_wnids)))
        model.get_layer('fc').trainable_variables[0].assign(corrected_fc_weights)

    elif args.bias_rect == 'weight_aligning':

        fc_weights = model.get_layer('fc').trainable_variables[0].numpy()
        fc_biases = model.get_layer('fc').trainable_variables[1].numpy()
        fc_weights_norms = np.linalg.norm(fc_weights, axis=0)
        old_mean, new_mean = np.mean(fc_weights_norms[:len(class_inc.old_wnids)]), np.mean(
            fc_weights_norms[len(class_inc.old_wnids):])
        lamda = old_mean / new_mean
        print('Lamda=%f' % lamda)

        corrected_fc_weights = fc_weights * np.array(
            [1.] * len(class_inc.old_wnids) + [lamda] * (len(class_inc.cumul_wnids) - len(class_inc.old_wnids)))
        model.get_layer('fc').trainable_variables[0].assign(corrected_fc_weights)

        corrected_fc_biases = fc_biases * np.array(
            [1.] * len(class_inc.old_wnids) + [lamda] * (len(class_inc.cumul_wnids) - len(class_inc.old_wnids)))
        model.get_layer('fc').trainable_variables[1].assign(corrected_fc_biases)

    elif args.bias_rect == 'bic':

        num_samples_per_old_class = tf.data.experimental.cardinality(rehearsal_dataset).numpy() // len(
            class_inc.old_wnids)
        num_skip_samples = tf.cast(tf.math.floor(args.val_exemplars_ratio * num_samples_per_old_class), tf.int64)
        concat_train_dataset = class_inc.train_dataset.concatenate(rehearsal_dataset)
        for class_idx in range(len(class_inc.cumul_wnids)):
            if class_idx == 0:
                bic_val_dataset = concat_train_dataset.filter(lambda img, label: tf.equal(label, class_idx)).take(
                    num_skip_samples)
            else:
                bic_val_dataset = bic_val_dataset.concatenate(
                    concat_train_dataset.filter(lambda img, label: tf.equal(label, class_idx)).take(
                        num_skip_samples))

        if args.bias_aug:
            bic_val_dataset = bic_val_dataset.cache().shuffle(num_skip_samples * len(class_inc.cumul_wnids)).map(
                class_inc.dataset.aug_fn,
                num_parallel_calls=args.AUTOTUNE).batch(args.batch_size).prefetch(args.AUTOTUNE)
        else:
            bic_val_dataset = bic_val_dataset.cache().shuffle(num_skip_samples * len(class_inc.cumul_wnids)).batch(
                args.batch_size).prefetch(args.AUTOTUNE)

        if args.sigmoid:
            bic_ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            bic_ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        lr = tf.Variable(args.base_lr)

        if args.optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif args.optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            raise Exception()

        # bic exclusive
        bic_train_acc_logger = tf.keras.metrics.SparseCategoricalAccuracy(name='bic_train_acc')
        bic_train_loss_logger = tf.keras.metrics.Mean(name='bic_train_loss')
        bic_ce_loss_logger = tf.keras.metrics.Mean(name='bic_ce_loss')
        bic_reg_loss_logger = tf.keras.metrics.Mean(name='bic_reg_loss')

        @tf.function
        def train_step_bic(images, labels):
            with tf.GradientTape() as tape:
                loss = 0.0

                # cross entropy
                logits = model(images)
                if args.sigmoid:
                    ce_loss = bic_ce_loss(tf.one_hot(labels, len(class_inc.cumul_wnids)), logits)
                else:
                    ce_loss = bic_ce_loss(labels, logits)
                loss += ce_loss

                # regularizer
                reg_loss = tf.nn.l2_loss(
                    [var for var in model.variables if 'bic_layer' in var.name and 'beta' in var.name]) * args.bias_w
                loss += reg_loss

            # optimize
            trainable_vars = [var for var in model.trainable_variables if 'bic_layer' in var.name]
            grads = tape.gradient(loss, trainable_vars)
            opt.apply_gradients(zip(grads, trainable_vars))
            bic_train_acc_logger(labels, logits)
            bic_train_loss_logger(loss)
            bic_ce_loss_logger(ce_loss)
            bic_reg_loss_logger(reg_loss)

        lr.assign(args.base_lr * 10.)
        global_step = tf.Variable(0)
        all_bias_epochs = args.epochs * args.bias_epochs
        for epoch in range(all_bias_epochs):

            # update base learning rate
            if all_bias_epochs >= 120:
                lr_desc_epochs = [70, 100]
            elif 100 <= all_bias_epochs < 120:
                lr_desc_epochs = [70, 90]
            else:
                lr_desc_epochs = [49, 63]

            if epoch in lr_desc_epochs:
                lr.assign(lr * 0.1)
                print('New learning rate: %f' % lr.numpy())

            # training
            starttime = time.time()
            tf.keras.backend.set_learning_phase(True)
            for batch_images, batch_labels in bic_val_dataset:
                global_step.assign_add(1)
                train_step_bic(batch_images, batch_labels)

            # testing
            if epoch < 5 or epoch in [49 * args.bias_epochs, 63 * args.bias_epochs] or (epoch + 1) % 5 == 0:
                tf.keras.backend.set_learning_phase(False)
                test_time = time.time()
                test_logits = model.predict(class_inc.test_images_now)
                test_scores = tf.nn.sigmoid(test_logits) if args.sigmoid else tf.nn.softmax(test_logits)
                test_scores = test_scores.numpy()
                test_preds = tf.argmax(test_scores, axis=1)
                test_conf_mat = tf.math.confusion_matrix(class_inc.test_labels_now, test_preds).numpy()
                test_acc_per_class = np.diag(test_conf_mat) * 100. / np.sum(test_conf_mat, axis=1)
                test_acc_avg = np.mean(test_acc_per_class)

                # top-5 accuracy
                top5_acc = get_top5_acc(test_scores, class_inc.test_labels_now)

                print(
                    'Epoch %d: Loss %.4f (ce %.4f, reg %.4f), Train Acc %.2f, Test Acc %.2f, Top-5 Acc %.2f, Time %.2f (%.2f)' % (
                        epoch + 1,
                        bic_train_loss_logger.result(),
                        bic_ce_loss_logger.result(),
                        bic_reg_loss_logger.result(),
                        bic_train_acc_logger.result() * 100.,
                        test_acc_avg,
                        top5_acc,
                        time.time() - starttime,
                        time.time() - test_time))

            bic_train_loss_logger.reset_states(),
            bic_ce_loss_logger.reset_states(),
            bic_reg_loss_logger.reset_states(),

    elif args.bias_rect == 'eeil':
        # only for later increments
        print('START BALANCED FINETUNING')
        concat_train_dataset = class_inc.train_dataset.concatenate(rehearsal_dataset)
        concat_train_dataset_shuffled = concat_train_dataset.shuffle(
            tf.data.experimental.cardinality(concat_train_dataset))
        num_old_exemplars = tf.data.experimental.cardinality(rehearsal_dataset) // len(class_inc.old_wnids)
        num_new_samples = tf.data.experimental.cardinality(class_inc.train_dataset) // len(class_inc.cur_wnids)
        num_reduced_samples = tf.minimum(num_old_exemplars, num_new_samples)

        # create a balanced training by keeping the number of new class samples the same as that of the old class
        for class_idx in range(len(class_inc.cumul_wnids)):
            if class_idx == 0:
                train_dataset_finetune = concat_train_dataset_shuffled.filter(
                    lambda img, label: tf.equal(label, class_idx)).take(num_reduced_samples)
            else:
                train_dataset_finetune = train_dataset_finetune.concatenate(
                    concat_train_dataset_shuffled.filter(lambda img, label: tf.equal(label, class_idx)).take(
                        num_reduced_samples))

        num_total_images = num_reduced_samples * len(class_inc.cumul_wnids)
        train_dataset_finetune = train_dataset_finetune.cache().shuffle(num_total_images).map(
            class_inc.dataset.aug_fn,
            num_parallel_calls=args.AUTOTUNE).repeat().batch(
            args.batch_size).prefetch(args.AUTOTUNE)

        num_batches_per_epoch = np.ceil(num_total_images.numpy() * 1. / args.batch_size).astype(int)

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

        global_step = tf.Variable(0)
        lr.assign(args.base_lr)
        my_summary = MySummary()

        @tf.function
        def train_step(images, labels, weights):
            with tf.GradientTape() as tape:
                loss = 0.0

                # cross entropy
                logits = model(images)
                if args.sigmoid:
                    ce_loss = loss_obj(tf.one_hot(labels, len(class_inc.cumul_wnids)), logits,
                                       sample_weight=weights)
                else:
                    ce_loss = loss_obj(labels, logits, sample_weight=weights)

                loss += ce_loss

                # regularizer
                reg_loss = tf.add_n(model.losses)
                if args.ada_weight_decay and not args.base_model:
                    reg_loss = reg_loss * len(class_inc.cumul_wnids) / (class_inc.group_idx + 1) / 10
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

        # train_dataset_for_finetune =
        for epoch in [i + args.epochs for i in range(args.finetune_epochs)]:
            # training
            starttime = time.time()
            current_step = 0

            if epoch == args.epochs:
                lr.assign(args.base_lr * 0.1)
            elif epoch % 10 == 0:
                lr.assign(lr * 0.1)

            tf.keras.backend.set_learning_phase(True)
            for batch_images, batch_labels in train_dataset_finetune:
                global_step.assign_add(1)
                batch_weights = np.ones(len(batch_labels))
                train_step(batch_images, batch_labels, batch_weights)

                step = int(global_step.numpy())
                my_summary.add('train_loss', step, train_loss_logger.result())
                my_summary.add('ce_loss', step, ce_loss_logger.result())
                my_summary.add('lwf_loss', step, lwf_loss_logger.result())
                my_summary.add('reg_loss', step, reg_loss_logger.result())
                my_summary.add('train_acc', step, train_acc_logger.result())

                current_step += 1
                if current_step % num_batches_per_epoch == 0:
                    break

            # testing
            if (epoch + 1) % 5 == 0:
                tf.keras.backend.set_learning_phase(False)
                test_time = time.time()
                test_logits = tf.concat(model.predict(class_inc.test_images_now), axis=1)
                test_scores = tf.nn.sigmoid(test_logits) if args.sigmoid else tf.nn.softmax(test_logits)
                test_scores = test_scores.numpy()
                test_preds = tf.argmax(test_scores, axis=1)
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

            # print('Test Acc: %s' % '|'.join(['%.f' % elem for elem in test_acc_per_class]))
            train_loss_logger.reset_states()
            ce_loss_logger.reset_states()
            lwf_loss_logger.reset_states()
            reg_loss_logger.reset_states()
            train_acc_logger.reset_states()
