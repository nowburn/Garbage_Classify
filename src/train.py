# -*- coding: utf-8 -*-
import os
import multiprocessing
from glob import glob

import numpy as np
from keras import backend
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import TensorBoard, Callback
# from moxing.framework import file

from data_gen import data_flow
from models.resnet50 import ResNet50

from keras.applications.nasnet import NASNetLarge
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D

from classification_models.keras import Classifiers

backend.set_image_data_format('channels_last')

avg_acc = {}


def nasnet_model_fn(FLAGS, objective, optimizer, metrics):
    nasnetlarge, preprocess_input = Classifiers.get('nasnetlarge')

    # build model
    base_model = nasnetlarge(input_shape=(FLAGS.input_size, FLAGS.input_size, 3), weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[output])
    model.load_weights(filepath='/home/nowburn/disk/data/Garbage_Classify/model_snapshots/weights_004_0.9921.h5',
                       by_name=True)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


def nasnet2_model_fn(FLAGS, objective, optimizer, metrics):
    """
    pre-trained nasnetlarge model
    """
    # 构建不带分类器的预训练模型
    base_model = NASNetLarge(input_shape=(FLAGS.input_size, FLAGS.input_size, 3), weights='imagenet', include_top=False)

    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    #                        gamma_initializer='ones', moving_mean_initializer='zeros',
    #                        moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    #                        beta_constraint=None, gamma_constraint=None)(x)

    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)
    # x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    #                        gamma_initializer='ones', moving_mean_initializer='zeros',
    #                        moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    #                        beta_constraint=None, gamma_constraint=None)(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


def model_fn(FLAGS, objective, optimizer, metrics):
    """
    pre-trained resnet50 model
    """
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


class LossHistory(Callback):
    def __init__(self, FLAGS):
        super(LossHistory, self).__init__()
        self.FLAGS = FLAGS

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        avg_acc['epoch_' + str(epoch + 1)] = (logs.get('acc') + logs.get('val_acc')) / 2

        save_path = os.path.join(self.FLAGS.train_local, 'weights_%03d_%.4f.h5' % (epoch + 1, logs.get('val_acc')))
        self.model.save_weights(save_path)
        print('save weights file', save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.train_local, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                for file_path in weights_files[self.FLAGS.keep_weights_file_num:]:
                    os.remove(file_path)  # only remove weights files on local path


def test_model(FLAGS, model):
    if FLAGS.test_data_url != '':
        print('test dataset predicting...')
        from eval import load_test_data
        img_names, test_data, test_labels = load_test_data(FLAGS)
        predictions = model.predict(test_data, verbose=0)

        right_count = 0
        for index, pred in enumerate(predictions):
            predict_label = np.argmax(pred, axis=0)
            test_label = test_labels[index]
            if predict_label == test_label:
                right_count += 1
        accuracy = right_count / len(img_names)
        print('accuracy: %0.4f' % accuracy)
        metric_file_name = os.path.join(FLAGS.train_local, 'metric.json')
        metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
        with open(metric_file_name, "w") as f:
            f.write(metric_file_content + '\n')


def train_model(FLAGS):
    # data flow generator
    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size)

    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.001)
    objective = 'binary_crossentropy'
    # objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = nasnet_model_fn(FLAGS, objective, optimizer, metrics)

    if not os.path.exists(FLAGS.train_local):
        os.makedirs(FLAGS.train_local)
    tensorBoard = TensorBoard(log_dir=FLAGS.train_local)
    history = LossHistory(FLAGS)

    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, tensorBoard],
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count() * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )
    # print('TOP FC trained over (%s)' % (FLAGS.max_epochs // 2))
    # print('Fine tuning start ...')
    # # fine tuning
    # for layer in model.layers[:1031]:
    #     layer.trainable = False
    # for layer in model.layers[1031:]:
    #     layer.trainable = True
    #
    # model.compile(loss=objective, optimizer=adam(lr=1e-4, clipnorm=0.001), metrics=metrics)
    # model.fit_generator(
    #     train_sequence,
    #     steps_per_epoch=len(train_sequence),
    #     epochs=FLAGS.max_epochs - FLAGS.max_epochs // 2,
    #     verbose=1,
    #     callbacks=[history, tensorBoard],
    #     validation_data=validation_sequence,
    #     max_queue_size=10,
    #     workers=int(multiprocessing.cpu_count() * 0.7),
    #     use_multiprocessing=True,
    #     shuffle=True
    # )
    top_3 = sorted(avg_acc.items(), key=lambda x: x[1], reverse=True)

    print('training done!')
    if FLAGS.deploy_script_path != '':
        from save_model import save_pb_model
        save_pb_model(FLAGS, model)

    test_model(FLAGS, model)

    print('end\n')
    print(top_3)
    with open(os.path.join(FLAGS.train_local, 'acc_rank.txt'), 'w') as f:
        for item in top_3:
            f.write('{}: {}\n'.format(item[0], item[1]))
