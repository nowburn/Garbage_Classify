import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


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


def preprocess_img(img):
    """
    image preprocessing
    you can add your special preprocess method here
    """
    resize_scale = 331 / max(img.size[:2])
    img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    img = img.convert('RGB')
    img = np.array(img)
    img = center_img(img, 331)
    return img


def multi_crop(img_path):
    cropped_list = []
    img = Image.open(img_path)
    w, h = img.size
    crop_areas = [(0, 0, w // 2, h // 2),
                  (w // 2, 0, w, h // 2),
                  (0, h // 2, w // 2, h),
                  (w // 2, h // 2, w, h),
                  (w // 4, h // 4, w * 3 // 4, h * 3 // 4)]
    for i, crop_area in enumerate(crop_areas):
        # filename = os.path.splitext(img_path)[0]
        # ext = os.path.splitext(img_path)[1]
        # new_filename = filename + '_cropped' + str(i) + ext
        cropped_image = img.crop(crop_area)
        # cropped_image.save(new_filename)
        cropped_list.append(preprocess_img(cropped_image))
        cropped_list.append(preprocess_img(cropped_image.transpose(Image.FLIP_LEFT_RIGHT)))
    return cropped_list


if __name__ == '__main__':

    img_path = '/home/nowburn/python_projects/python/Garbage_Classify/datasets/origin_data/train/img_17.jpg'

    plt.figure(figsize=(8, 8))
    plt.subplot(4, 4, 1)
    plt.axis('off')
    plt.imshow(Image.open(img_path))

    cropped_list = multi_crop(img_path)
    for i in range(len(cropped_list)):
        plt.subplot(4, 4, i + 5)
        plt.imshow(cropped_list[i])
        plt.axis('off')
    plt.show()
