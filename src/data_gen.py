# -*- coding: utf-8 -*-
import multiprocessing
import os
import math
import codecs
import random

import cv2
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """

    def __init__(self, img_paths, labels, batch_size, img_size, is_train=True):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """
        center img in a square background
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = Image.open(img_path)
        resize_scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.convert('RGB')
        img = np.array(img)
        img = self.center_img(img, self.img_size[0])
        return img

    def mixup_augment(self, batch_x, batch_y, alpha=0.1):
        tmpx = np.copy(batch_x[0])
        tmpy = np.copy(batch_y[0])
        for i in range(3):
            batch_x[i] = alpha * batch_x[i] + (1 - alpha) * batch_x[i + 1]
            batch_y[i] = alpha * batch_y[i] + (1 - alpha) * batch_y[i + 1]
        batch_x[3] = alpha * batch_x[3] + (1 - alpha) * tmpx
        batch_y[3] = alpha * batch_y[3] + (1 - alpha) * tmpy
        return batch_x, batch_y

    def get_random_data(self, img_path):

        def add_noise(image, percentage):
            noise_image = image.copy()
            im_w = image.shape[1]
            im_h = image.shape[0]
            noise_num = int(percentage * im_w * im_h)
            for i in range(noise_num):
                temp_x = np.random.randint(0, image.shape[1])
                temp_y = np.random.randint(0, image.shape[0])
                noise_image[temp_y][temp_x][np.random.randint(3)] = np.random.randn(1)[0]
            return noise_image

        def rotate(image):
            (h, w) = image.shape[:2]
            center = (w / 2, h / 2)
            angle = (np.random.random() - 0.5) * 20
            M = cv2.getRotationMatrix2D(center, angle, 1)
            image = cv2.warpAffine(image, M, (w, h))
            return image

        def crop(image):
            img_w = image.shape[1]
            img_h = image.shape[0]
            h = np.random.randint(30, 50)
            w = np.random.randint(30, 50)
            image = image[h:h + img_h, w:w + img_w, :]
            return image

        NUM_ANGMENTATION_SUPPORT = 3

        # 数据格式   imgPath,label
        image = Image.open(img_path)
        resize_scale = self.img_size[0] / max(image.size[:2])
        image = image.resize((int(image.size[0] * resize_scale), int(image.size[1] * resize_scale)))
        image = np.array(image)
        aug_num = np.random.randint(low=0, high=NUM_ANGMENTATION_SUPPORT)
        aug_queue = np.random.permutation(NUM_ANGMENTATION_SUPPORT)[0:aug_num]
        try:
            for idx in aug_queue:
                if idx == 0:
                    image = np.fliplr(image)
                elif idx == 1:
                    image = crop(image)
                elif idx == 2:
                    image = rotate(image)
        except Exception as e:
            print('\nexcept:', e)

        image = self.center_img(image, self.img_size[0])
        return image

    def get_augment_data(self, img_path):
        image = Image.open(img_path)
        resize_scale = self.img_size[0] / max(image.size[:2])
        image = image.resize((int(image.size[0] * resize_scale), int(image.size[1] * resize_scale)))
        image = np.array(image)

        augs = iaa.SomeOf((1, 3),
                          [
                              iaa.CropAndPad(percent=(-0.3, 0.3), pad_mode=["constant"], pad_cval=(255)),
                              iaa.Affine(rotate=(-25, 25)),
                              iaa.Fliplr(0.5),
                              # iaa.GaussianBlur(sigma=1.0),
                              iaa.Noop()
                              # iaa.Affine(scale={'x': (0.5, 1.2), 'y': (0.5, 1.2)}),
                              # # iaa.Affine(translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}),
                          ])
        seq = iaa.Sequential([augs])
        image = seq.augment_image(image)
        image = self.center_img(image, self.img_size[0])
        return image

    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        # batch_y = np.array(batch_y).astype(np.float32)

        batch_y = np.array(batch_y).astype(np.float32) * (1 - 0.05) + 0.05 / 40
        if self.is_train:
            batch_x = np.array([self.get_augment_data(img_path) for img_path in batch_x])
            if len(batch_y) == self.batch_size:
                return self.mixup_augment(batch_x, batch_y)
        else:
            batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])

        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)


def data_flow(train_data_dir_list, batch_size, num_classes, input_size):  # need modify

    label_files = []
    for train_data_dir in train_data_dir_list:
        label_files += glob(os.path.join(train_data_dir, '*.txt'))

    random.shuffle(label_files)
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        cur_dir = file_path.split('img_')[0]
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(cur_dir, img_name))
        labels.append(label)
    labels = np_utils.to_categorical(labels, num_classes)
    train_img_paths, validation_img_paths, train_labels, validation_labels = \
        train_test_split(img_paths, labels, test_size=0.2, random_state=0, stratify=labels)
    print('total samples: %d, training samples: %d, validation samples: %d' % (
        len(img_paths), len(train_img_paths), len(validation_img_paths)))
    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size], is_train=True)
    validation_sequence = BaseSequence(validation_img_paths, validation_labels, batch_size, [input_size, input_size],
                                       is_train=False)

    # 构造多进程的数据流生成器
    # train_enqueuer = OrderedEnqueuer(train_sequence, use_multiprocessing=True, shuffle=True)
    # validation_enqueuer = OrderedEnqueuer(validation_sequence, use_multiprocessing=True, shuffle=True)
    #
    # # 启动数据生成器
    # n_cpu = multiprocessing.cpu_count()
    # train_enqueuer.start(workers=int(n_cpu * 0.7), max_queue_size=10)
    # validation_enqueuer.start(workers=1, max_queue_size=10)
    # train_data_generator = train_enqueuer.get()
    # validation_data_generator = validation_enqueuer.get()

    # return train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator
    return train_sequence, validation_sequence


def origin_test():
    # train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator = data_flow(dog_cat_data_path, batch_size)
    # for i in range(10):
    #     train_data_batch = next(train_data_generator)
    # train_enqueuer.stop()
    # validation_enqueuer.stop()
    train_data_dir = ''
    batch_size = 16
    train_sequence, validation_sequence = data_flow(train_data_dir, batch_size)
    batch_data, bacth_label = train_sequence.__getitem__(5)
    label_name = ['cat', 'dog']
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_2_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_3_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    print('end')


def plotImage(X):
    plt.figure(figsize=(10, 10))
    plt.imshow(X.reshape(331, 331, 3))
    plt.show()
    plt.close()


def data_augmentation2():
    train_img_paths = [
        '/home/nowburn/python_projects/python/Garbage_Classify/datasets/train_data/img_17.jpg',
        '/home/nowburn/disk/data/Garbage_Classify/new/nas-new_all-aug-9/wrong/1/img_17976.jpg']

    train_labels = [[0], [1]]
    train_sequence = BaseSequence(train_img_paths, train_labels, 2, [331, 331])

    for i in range(10):
        batchx, batchy = train_sequence.__getitem__(0)
        # cv2.imwrite('/home/nowburn/disk/data/Garbage_Classify/augment/%s.jpg' % i, batchx[0])
        img = Image.fromarray(batchx[0])
        img.save('/home/nowburn/disk/data/Garbage_Classify/augment/%s.jpg' % i)

    print('Done')


def data_augmentation():
    train_img_paths = [
        '/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/train/img_1.jpg',
        '/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/train/img_17.jpg',
        '/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/train/img_2913.jpg',
        '/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/train/img_17397.jpg']
    train_labels = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    train_sequence = BaseSequence(train_img_paths, train_labels, 4, [331, 331])

    # while True:
    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(4, 3, 1)
    #     plt.axis('off')
    #     plt.imshow(Image.open(train_img_paths[0]))
    #     for i in range(4):
    #         batch_x, batch_y = train_sequence.__getitem__(0)
    #         plt.subplot(4, 3, i + 4)
    #         plt.imshow(batch_x[0])
    #         plt.axis('off')
    #     plt.show()

    plt.figure(figsize=(12, 12))
    for i, path in enumerate(train_img_paths):
        plt.subplot(2, 4, i + 1)
        plt.imshow(Image.open(path))
        plt.title(train_labels[i])
        plt.axis('off')

    batch_x, batch_y = train_sequence.__getitem__(0)
    for i in range(len(batch_x)):
        plt.subplot(2, 4, i + 5)
        plt.imshow(batch_x[i])
        plt.title(batch_y[i], fontsize='xx-small')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    data_augmentation()
