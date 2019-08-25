import collections
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from glob import glob
import shutil

BASE_DIR = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/my_data/processed/'

IMG_DIR = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/my_data/cardboard/'
TYPE_DICT = {'cardboard': 30, 'glass': 27, 'metal': 33, 'paper': 0, 'plastic': 0, 'trash': 0}

MODEL_PATH = '/home/nowburn/disk/data/Garbage_Classify/pretrained_model/kaggle_model/model/'


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

    def inference(self, data, file_list, base_dir):
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

    def postprocess(self, data):
        return data


def copy_example():
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


def rename():
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


def format_data():
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


def move_file():
    dir_list = [dir for dir in os.listdir(BASE_DIR)]
    for dir in dir_list:
        for file in glob((os.path.join(BASE_DIR + dir, '*.*'))):
            shutil.move(file, os.path.join(BASE_DIR, os.path.basename(file)))


def del_unpair():
    dir_list = [dir for dir in os.listdir(BASE_DIR)]
    for dir in dir_list:
        file_list = glob(os.path.join(BASE_DIR + dir, '*.txt'))
        for file in file_list:
            img_file = file.replace('txt', 'jpg')
            if not os.path.exists(img_file):
                os.remove(file)


if __name__ == '__main__':
    # del_unpair()
    move_file()
    # format_data()
    # server = Garbage_classify_service('TEST', MODEL_PATH)
    # data = {}
    # img_dict = collections.OrderedDict()
    # img_list = glob(os.path.join(IMG_DIR, '*.jpg'))
    # img_list.sort(key=lambda x: int((os.path.basename(x)[9:]).split('.')[0]))
    # for img_path in img_list:
    #     file_name = os.path.basename(img_path)
    #     img_dict[file_name[:-4]] = img_path
    #
    # data = server.preprocess(img_dict)
    # server.inference(data, img_list, BASE_DIR+'processed')