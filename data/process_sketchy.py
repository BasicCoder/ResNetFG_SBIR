import os
import random
import shutil


class Config(object):

    data_root = '/home/bc/Work/Database/sketchy/256x256/photo/tx_000100000000'    # the data set location
    t_ratio = 0.9                                               # the ratio of the train set

    train_set_root = '/home/bc/Work/caffe2torch/data/dataset/photo-train'               # where to save the training set
    test_set_root = '/home/bc/Work/caffe2torch/data/dataset/photo-test'                 # where to save the testing set


class Divide(object):

    def __init__(self, config):

        self.__data_root = config.data_root
        self.__train_root = config.train_set_root
        self.__test_root = config.test_set_root
        self.__ratio = config.t_ratio

    def find_classes(self):
        root = self.__data_root

        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        return classes

    def divide(self):
        root = self.__data_root
        train_root = self.__train_root
        test_root = self.__test_root

        if not os.path.exists(train_root):
            os.mkdir(train_root)
            os.mkdir(test_root)

        classes = self.find_classes()
        for class_name in classes:
            print(class_name)
            class_path = os.path.join(root, class_name)
            train_save_path = os.path.join(train_root, class_name)
            test_save_path = os.path.join(test_root, class_name)

            if not os.path.exists(train_save_path):
                os.mkdir(train_save_path)
                os.mkdir(test_save_path)

            images = os.listdir(class_path)
            random.shuffle(images)

            threshold = int(len(images) * 0.9)

            for i, image in enumerate(images):
                src_path = os.path.join(class_path, image)

                if i < threshold:
                    dst_path = os.path.join(train_save_path, image)
                else:
                    dst_path = os.path.join(test_save_path, image)

                print(i)
                shutil.copy(src_path, dst_path)

if __name__ == '__main__':

    config = Config()
    divide = Divide(config)
    divide.divide()