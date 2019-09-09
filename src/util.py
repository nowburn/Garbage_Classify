import collections
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from glob import glob
import shutil
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from classification_models.keras import Classifiers
import matplotlib.pyplot as plt

BASE_DIR = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/train_data/processed/'

IMG_DIR = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/test_data/'

MODEL_PATH = '/home/nowburn/disk/data/Garbage_Classify/model_imagedatagenerate/nas-origin_data-auto_augment-30/model/'


# MODEL_PATH = '/home/nowburn/disk/data/Garbage_Classify/model_snapshots/nas-origin-augment-25/model/'


class Garbage_classify_service():
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.signature_key = 'predict_images'

        self.input_size = 331  # the input image size of the model

        # add the input and output key of your pb model here,
        # these keys are defined when you save a pb file
        self.input_key_1 = 'input_img'
        self.output_key_1 = 'output_score'
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.get_default_graph().as_default():
            self.sess = tf.Session(graph=tf.Graph(), config=config)
            meta_graph_def = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.model_path)
            self.signature = meta_graph_def.signature_def

            # define input and out tensor of your model here
            input_images_tensor_name = self.signature[self.signature_key].inputs[self.input_key_1].name
            output_score_tensor_name = self.signature[self.signature_key].outputs[self.output_key_1].name
            self.input_images = self.sess.graph.get_tensor_by_name(input_images_tensor_name)
            self.output_score = self.sess.graph.get_tensor_by_name(output_score_tensor_name)

        self.label_id_name_dict = \
            {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物"
            }

    def center_img(self, img, size=None, fill_value=255):
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

    def preprocess_img(self, img):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        resize_scale = self.input_size / max(img.size[:2])
        img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.convert('RGB')
        img = np.array(img)
        img = img[:, :, ::-1]
        img = self.center_img(img, self.input_size)
        return img

    def preprocess(self, data):
        preprocessed_data = {}
        for file_name, file_content in data.items():
            img = Image.open(file_content)
            img = self.preprocess_img(img)
            preprocessed_data[file_name] = img
        return preprocessed_data

    def inference2(self, data, file_list, base_dir):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        idx = 0
        for name, img in data.items():

            img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
            pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
            if pred_score is not None:
                pred_label = np.argmax(pred_score[0], axis=1)[0]
                result = {str(pred_label) + '/' + name: self.label_id_name_dict[str(pred_label)]}
                dst_dir = os.path.join(base_dir, str(pred_label))
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                shutil.move(file_list[idx], os.path.join(dst_dir, name + '.jpg'))
            else:
                result = {'result': 'predict score is None'}
            idx += 1
            print(result)

    def inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        idx = 0
        for name, img in data.items():

            img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
            pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
            if pred_score is not None:
                pred_label = np.argmax(pred_score[0], axis=1)[0]
                result = {str(pred_label) + ': ' + name: self.label_id_name_dict[str(pred_label)]}
            else:
                result = {'result': 'predict score is None'}
            idx += 1
            print(result)

    def postprocess(self, data):
        return data


class Newdata_pipeline():

    def copy_example(self):
        type = [0 for i in range(40)]
        src_dir = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/train_data/'
        dst_dir = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/example/'
        file_list = glob(os.path.join(src_dir, '*.txt'))
        for file_path in file_list:
            with open(file_path, 'r') as f:
                line = f.readline()
                line_split = line.strip().split(', ')
                idx = int(line_split[1])
                if type[idx] < 5:
                    src = os.path.join(src_dir, line_split[0])
                    dst = os.path.join(dst_dir, line_split[0])
                    shutil.copyfile(src, dst)
                    type[idx] += 1

    def rename(self):
        dst_dir = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/example/'
        file_list = glob(os.path.join(dst_dir, '*.jpg'))
        file_list.sort(key=lambda x: int((os.path.basename(x)[4:]).split('.')[0]))
        type = [i for i in range(40)]

        cnt = 0
        prefix = ''
        for file in file_list:
            if cnt % 5 == 0:
                prefix = str(type[cnt // 5]) + '_'
            os.rename(file, os.path.join(dst_dir, prefix + str(cnt + 1) + '.jpg'))
            cnt += 1

    def format_data(self):
        last_no = 19735
        dir_list = [dir for dir in os.listdir(BASE_DIR)]
        dir_list.sort()
        for dir in dir_list:
            cnt = 0
            for old_file in glob((os.path.join(BASE_DIR + dir, '*.jpg'))):
                last_no += 1
                cnt += 1
                new_file = os.path.join(BASE_DIR + dir, 'img_' + str(last_no) + '.jpg')
                os.rename(old_file, new_file)
                with open(new_file[:-4] + '.txt', 'w') as f:
                    f.write('%s, %s' % (os.path.basename(new_file), dir))

            print(dir)
            print('=' * 50)
            print('renamed %s imgs' % cnt)
            print('last no: %s' % last_no)

    def move_file(self):
        dir_list = [dir for dir in os.listdir(BASE_DIR)]
        for dir in dir_list:
            for file in glob((os.path.join(BASE_DIR + dir, '*.*'))):
                shutil.move(file, os.path.join(BASE_DIR, os.path.basename(file)))

    def del_unpair(self):
        dir_list = [dir for dir in os.listdir(BASE_DIR)]
        for dir in dir_list:
            file_list = glob(os.path.join(BASE_DIR + dir, '*.txt'))
            for file in file_list:
                img_file = file.replace('txt', 'jpg')
                if not os.path.exists(img_file):
                    os.remove(file)

    def split_dir(self, data_dir_list, output_basedir):
        label_files = []
        train_img_paths = []
        train_labels = []
        val_img_paths = []
        val_labels = []
        for train_data_dir in data_dir_list:
            label_files += glob(os.path.join(train_data_dir, '*.txt'))

        random.shuffle(label_files)
        img_paths = []
        labels = []
        for index, file_path in enumerate(label_files):
            cur_dir = file_path.split('img_')[0]
            with open(file_path, 'r') as f:
                line = f.readline()
            line_split = line.strip().split(', ')
            if len(line_split) != 2:
                print('%s contain error lable' % os.path.basename(file_path))
                continue
            img_name = line_split[0]
            label = int(line_split[1])
            img_paths.append(os.path.join(cur_dir, img_name))
            labels.append(label)
        train_img_paths, val_img_paths, train_labels, val_labels = \
            train_test_split(img_paths, labels, test_size=0.25, random_state=0, stratify=labels)

        train_data_dir = os.path.join(output_basedir, 'train')
        val_data_dir = os.path.join(output_basedir, 'valid')
        if not os.path.exists(train_data_dir):
            os.mkdir(train_data_dir)
        if not os.path.exists(val_data_dir):
            os.mkdir(val_data_dir)
        for label, path in zip(train_labels, train_img_paths):
            class_dir = os.path.join(train_data_dir, str(label))
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
            shutil.move(path, os.path.join(class_dir, os.path.basename(path)))
            os.remove(path.replace('.jpg', '.txt'))

        for label, path in zip(val_labels, val_img_paths):
            class_dir = os.path.join(val_data_dir, str(label))
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)
            shutil.move(path, os.path.join(class_dir, os.path.basename(path)))
            os.remove(path.replace('.jpg', '.txt'))
        print('=' * 70)
        print('train_data : %s' % len(train_labels))
        print('val_data : %s' % len(val_labels))

    def split_data(self, data_dir_list, output_basedir):
        label_files = []
        for train_data_dir in data_dir_list:
            label_files += glob(os.path.join(train_data_dir, '*.txt'))

        img_paths = []
        labels = []
        for index, file_path in enumerate(label_files):
            cur_dir = file_path.split('img_')[0]
            with open(file_path, 'r') as f:
                line = f.readline()
            line_split = line.strip().split(', ')
            if len(line_split) != 2:
                print('%s contain error lable' % os.path.basename(file_path))
                continue
            img_name = line_split[0]
            label = int(line_split[1])
            img_paths.append(os.path.join(cur_dir, img_name))
            labels.append(label)
        train_img_paths, val_img_paths, train_labels, val_labels = \
            train_test_split(img_paths, labels, test_size=0.2, random_state=0, stratify=labels)

        train_data_dir = os.path.join(output_basedir, 'train')
        val_data_dir = os.path.join(output_basedir, 'test')
        if not os.path.exists(train_data_dir):
            os.mkdir(train_data_dir)
        if not os.path.exists(val_data_dir):
            os.mkdir(val_data_dir)
        for label, path in zip(train_labels, train_img_paths):
            txt_path = path.replace('.jpg', '.txt')
            shutil.copyfile(path, os.path.join(train_data_dir, os.path.basename(path)))
            shutil.copyfile(txt_path, os.path.join(train_data_dir, os.path.basename(txt_path)))

        for label, path in zip(val_labels, val_img_paths):
            txt_path = path.replace('.jpg', '.txt')
            shutil.copyfile(path, os.path.join(val_data_dir, os.path.basename(path)))
            shutil.copyfile(txt_path, os.path.join(val_data_dir, os.path.basename(txt_path)))
        print('=' * 70)
        print('train_data : %s' % len(train_labels))
        print('val_data : %s' % len(val_labels))


class Preprocess():

    def nasnetlarge_process(self, path):
        nasnetlarge, preprocess_input = Classifiers.get('nasnetlarge')
        img = Image.open(path)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        print(type(img))
        print(img.shape)
        img2 = preprocess_input(img)
        print(type(img2))
        print(img2.shape)
        plt.imshow(img2[0])
        plt.show()
        # cv2.imshow("img2", img2)
        # cv2.imwrite('/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/img2.jpg', img2)
        # cv2.waitKey(0)


def check_error_imgs(txtdir, testdir):
    label_id_name_dict = \
        {
            "0": "其他垃圾/一次性快餐盒",
            "1": "其他垃圾/污损塑料",
            "2": "其他垃圾/烟蒂",
            "3": "其他垃圾/牙签",
            "4": "其他垃圾/破碎花盆及碟碗",
            "5": "其他垃圾/竹筷",
            "6": "厨余垃圾/剩饭剩菜",
            "7": "厨余垃圾/大骨头",
            "8": "厨余垃圾/水果果皮",
            "9": "厨余垃圾/水果果肉",
            "10": "厨余垃圾/茶叶渣",
            "11": "厨余垃圾/菜叶菜根",
            "12": "厨余垃圾/蛋壳",
            "13": "厨余垃圾/鱼骨",
            "14": "可回收物/充电宝",
            "15": "可回收物/包",
            "16": "可回收物/化妆品瓶",
            "17": "可回收物/塑料玩具",
            "18": "可回收物/塑料碗盆",
            "19": "可回收物/塑料衣架",
            "20": "可回收物/快递纸袋",
            "21": "可回收物/插头电线",
            "22": "可回收物/旧衣服",
            "23": "可回收物/易拉罐",
            "24": "可回收物/枕头",
            "25": "可回收物/毛绒玩具",
            "26": "可回收物/洗发水瓶",
            "27": "可回收物/玻璃杯",
            "28": "可回收物/皮鞋",
            "29": "可回收物/砧板",
            "30": "可回收物/纸板箱",
            "31": "可回收物/调料瓶",
            "32": "可回收物/酒瓶",
            "33": "可回收物/金属食品罐",
            "34": "可回收物/锅",
            "35": "可回收物/食用油桶",
            "36": "可回收物/饮料瓶",
            "37": "有害垃圾/干电池",
            "38": "有害垃圾/软膏",
            "39": "有害垃圾/过期药物"
        }
    correct_dict = {}
    with open(os.path.join(txtdir, 'accuracy.txt'), 'r') as f:
        img_list = f.readlines()
        for line in img_list:
            if line.startswith('img_'):
                name, true_label, predit_label = line.strip().split(',')
                wrong_dir = os.path.join(txtdir, 'wrong/' + str(predit_label.strip()))
                correct_dict[os.path.join(wrong_dir, name)] = true_label.strip() + ',' + predit_label.strip()

    correct_dict = sorted(correct_dict.items(), key=lambda x: int((x[0].split('img_'))[1][:-4]))
    for item in correct_dict:
        dst_file, true_label, predit_label = item[0], item[1].split(',')[0], item[1].split(',')[1]
        wrong_dir = dst_file.split('img_')[0]
        name = os.path.basename(dst_file)
        if not os.path.exists(wrong_dir):
            os.makedirs(wrong_dir)
        shutil.copyfile(os.path.join(testdir, name), os.path.join(wrong_dir, name))
        with open(wrong_dir + '/correct.txt', 'a') as f:
            f.write('{}: {}({}) {}({})\n'.format(name, label_id_name_dict[true_label.strip()], true_label,
                                                 label_id_name_dict[predit_label.strip()], predit_label))


if __name__ == '__main__':
    # pipline = Newdata_pipeline()
    # data_dir_list = ['/home/nowburn/disk/data/Garbage_Classify/source/garbage_classify_v2/train_data_v2',
    #                  ]
    #
    # test_dir = ['/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/test_data']
    # output_dir = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data'
    # pipline.split_data(data_dir_list, output_dir)

    txt_dir = '/home/nowburn/disk/data/Garbage_Classify/models/nas-label_smoothing-tta-11'
    test_dir = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/test'
    check_error_imgs(txt_dir, test_dir)
    # train_dir_list = ['/home/nowburn/disk/data/Garbage_Classify/source/data/train_data/']
    # output_base_dir = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/'
    # pipeline = Newdata_pipeline()
    # pipeline.split_dir(train_dir_list, output_base_dir)

    # server = Garbage_classify_service('TEST', MODEL_PATH)
    # data = {}
    # img_dict = collections.OrderedDict()
    # img_list = glob(os.path.join(IMG_DIR, '*.jpg'))
    # # img_list.sort(key=lambda x: int((os.path.basename(x)[9:]).split('.')[0]))
    # for img_path in img_list:
    #     file_name = os.path.basename(img_path)
    #     img_dict[file_name.split('img_')[1]] = img_path
    #
    # data = server.preprocess(img_dict)
    # server.inference(data)
