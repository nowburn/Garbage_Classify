# -*- coding: utf-8 -*-
import os
from datetime import datetime
import multiprocessing
from glob import glob

import numpy as np
from keras import backend
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau
# from moxing.framework import file

from models.resnet50 import ResNet50

from keras.layers import Dense, Flatten, GlobalAveragePooling2D

from classification_models.keras import Classifiers
from datasets.my_data_gen import get_tran_val
from keras import regularizers

backend.set_image_data_format('channels_last')

avg_acc = {}


def senet_model_fn(FLAGS, objective, optimizer, metrics):
    senet, preprocess_input = Classifiers.get('senet154')

    # build model
    base_model = senet(input_shape=(FLAGS.input_size, FLAGS.input_size, 3), weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    for layer in model.layers[:4000]:
        layer.trainable = False
    for layer in model.layers[4000:]:
        layer.trainable = True

    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


def nasnet_model_fn(FLAGS, objective, optimizer, metrics):
    weight_decay = 5e-4
    nasnetlarge, preprocess_input = Classifiers.get('nasnetlarge')
    # build model
    base_model = nasnetlarge(input_shape=(FLAGS.input_size, FLAGS.input_size, 3), weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(FLAGS.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)
    model = Model(inputs=[base_model.input], outputs=[output])
    # model.load_weights(
    #     filepath='/home/nowburn/disk/data/Garbage_Classify/model_snapshots/model-nasnetlarge-5-all/weights_005_0.9944.h5',
    #     by_name=True)
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
    start_time = datetime.now()
    # data flow generator
    # train_data_dir_list = list(FLAGS.data_local.split(','))
    # train_sequence, validation_sequence = data_flow(train_data_dir_list, FLAGS.batch_size,
    #                                                 FLAGS.num_classes, FLAGS.input_size)
    dir_list = list(FLAGS.data_local.split(','))
    train_generator, validation_generator = get_tran_val(dir_list[0], dir_list[1], FLAGS.input_size, FLAGS.batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, mode='auto', min_lr=1e-16)

    optimizer = adam(lr=FLAGS.learning_rate, clipnorm=0.001)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = nasnet_model_fn(FLAGS, objective, optimizer, metrics)

    if not os.path.exists(FLAGS.train_local):
        os.makedirs(FLAGS.train_local)
    tensorBoard = TensorBoard(log_dir=FLAGS.train_local)
    history = LossHistory(FLAGS)

    model.fit_generator(
        train_generator,
        steps_per_epoch=(11101 // FLAGS.batch_size) + 1,
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, reduce_lr, tensorBoard],
        validation_data=validation_generator,
        validation_steps=(3701 // FLAGS.batch_size) + 1,
        workers=int(multiprocessing.cpu_count() * 0.7),
        use_multiprocessing=True,
    )

    rank = sorted(avg_acc.items(), key=lambda x: x[1], reverse=True)

    print('training done!')
    if FLAGS.deploy_script_path != '':
        from save_model import save_pb_model
        save_pb_model(FLAGS, model)

    end_time = datetime.now()
    cost_seconds = (end_time - start_time).seconds
    print('=' * 70)
    print('Cost time: {}:{}:{}\n'.format(cost_seconds // 3600, (cost_seconds % 3600) // 60,
                                         cost_seconds % 60))
    for item in rank:
        print('{}: {}\n'.format(item[0], item[1]))

    with open(os.path.join(FLAGS.train_local, 'acc_rank.txt'), 'w') as f:
        for item in rank:
            f.write('{}: {}\n'.format(item[0], item[1]))
