# -*- coding: utf-8 -*-
import os
from datetime import datetime
import multiprocessing
from glob import glob

import numpy as np
from keras import backend
from keras.optimizers import adam
from keras.callbacks import TensorBoard, Callback, EarlyStopping, ReduceLROnPlateau
# from moxing.framework import file
from data_gen import data_flow
from save_model import save_pb_model
from nets import nasnet_model_fn, multimodel

backend.set_image_data_format('channels_last')

train_details = {}
train_sequence = None
validation_sequence = None


class LossHistory(Callback):
    def __init__(self, FLAGS, train_data_dir_list, model):
        super(LossHistory, self).__init__()
        self.FLAGS = FLAGS
        self.train_data_dir_list = train_data_dir_list
        self.model = model

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        train_details['epoch_' + str(epoch + 1)] = [logs.get('acc'), logs.get('val_acc')]

        save_path = os.path.join(self.FLAGS.train_local, 'weights_%03d_%.4f.h5' % (epoch + 1, logs.get('val_acc')))
        self.model.save_weights(save_path)
        print('save weights file', save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.train_local, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                for file_path in weights_files[self.FLAGS.keep_weights_file_num:]:
                    os.remove(file_path)  # only remove weights files on local path
        global train_sequence, validation_sequence
        train_sequence, validation_sequence = data_flow(self.train_data_dir_list, self.FLAGS.batch_size,
                                                        self.FLAGS.num_classes, self.FLAGS.input_size)
        test_model(self.FLAGS, self.model, need_save=False)


def test_model(FLAGS, model, need_save=True):
    if FLAGS.test_data_url != '':
        print('test dataset predicting...')
        from eval import load_test_data
        img_names, test_data, test_labels = load_test_data(FLAGS)
        predictions = model.predict(test_data, verbose=0)
        right_count = 0
        error_infos = []
        for index, pred in enumerate(predictions):
            predict_label = np.argmax(pred, axis=0)
            test_label = test_labels[index]
            if predict_label == test_label:
                right_count += 1
            else:
                error_infos.append('%s, %s, %s\n' % (img_names[index], test_label, predict_label))
        accuracy = right_count / len(img_names)
        print('accuracy: %0.4f' % accuracy)
        if need_save:
            result_file_name = os.path.join(FLAGS.train_url, 'accuracy.txt')
            with open(result_file_name, 'w') as f:
                f.write('# predict error files\n')
                f.write('####################################\n')
                f.write('file_name, true_label, pred_label\n')
                f.writelines(error_infos)
                f.write('####################################\n')
                f.write('accuracy: %s\n' % accuracy)


def train_model(FLAGS):
    start_time = datetime.now()
    # data flow generator
    train_data_dir_list = list(FLAGS.data_local.split(','))
    train_sequence, validation_sequence = data_flow(train_data_dir_list, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size)

    # dir_list = list(FLAGS.data_local.split(','))
    # train_generator, validation_generator = get_tran_val(dir_list[0], dir_list[1], FLAGS.input_size, FLAGS.batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=1, mode='auto', min_lr=1e-16)

    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.0005)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = multimodel(FLAGS, objective, optimizer, metrics)

    if not os.path.exists(FLAGS.train_local):
        os.makedirs(FLAGS.train_local)
    tensorBoard = TensorBoard(log_dir=FLAGS.train_local)
    history = LossHistory(FLAGS, train_data_dir_list, model)

    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, tensorBoard, reduce_lr,
                   EarlyStopping(monitor='val_acc', patience=3, restore_best_weights=True)],
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count() * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )
    # model.fit_generator(
    #     train_sequence,
    #     steps_per_epoch=(12621 // FLAGS.batch_size) + 1,
    #     epochs=FLAGS.max_epochs,
    #     verbose=1,
    #     callbacks=[history, reduce_lr, tensorBoard],
    #     validation_data=validation_sequence,
    #     validation_steps=(3156 // FLAGS.batch_size) + 1,
    # )

    print('training done!')
    if FLAGS.deploy_script_path != '':
        save_pb_model(FLAGS, model)

    end_time = datetime.now()
    cost_seconds = (end_time - start_time).seconds
    print('=' * 70)
    print('Cost time: {}:{}:{}\n'.format(cost_seconds // 3600, (cost_seconds % 3600) // 60,
                                         cost_seconds % 60))

    with open(os.path.join(FLAGS.train_local, 'train_details.txt'), 'w') as f:
        rank = sorted(train_details.items(), key=lambda x: int(x[0].split('_')[1]), reverse=False)
        f.write('epoch order\n')
        for item in rank:
            f.write('{}: acc: {:.7f}  val_acc: {:.7f}\n'.format(item[0], item[1][0], item[1][1]))

        f.write('=' * 70 + '\n')
        f.write('val_acc order\n')
        rank = sorted(train_details.items(), key=lambda x: x[1][1], reverse=True)
        for item in rank:
            f.write('{}: acc: {:.7f}  val_acc: {:.7f}\n'.format(item[0], item[1][0], item[1][1]))

    test_model(FLAGS, model)
