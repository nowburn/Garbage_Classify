import cv2
if __name__ == '__main__':
    train_img_paths = [
        '/home/nowburn/python_projects/python/Garbage_Classify/datasets/garbage_classify/train_data/img_17.jpg']
    train_labels = [[0]]
    #train_sequence = BaseSequence(train_img_paths, train_labels, 1, [224, 224])

    #batchx, batchy = train_sequence.__getitem__(0)
    batchx = cv2.imread(train_img_paths[0])
    cv2.imshow('img', batchx)
    cv2.waitKey(0)