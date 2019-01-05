import os
import shutil

class Config(object):

    data_root = '/home/bc/Work/Database/sketchy/256x256/photo/tx_000100000000'    # the data set location
    test_photo_txt = '/home/bc/Work/Database/sketchy/info/testset.txt'            # the photo of the test set

    train_set_root = '/home/bc/Work/caffe2torch/data/dataset/photo-train-all'               # where to save the training set
    test_set_root = '/home/bc/Work/caffe2torch/data/dataset/photo-test-all'                 # where to save the testing set

class Divide(object):

    def __init__(self, config):
        self.__data_root = config.data_root
        self.__train_root = config.train_set_root
        self.__test_root = config.test_set_root
        self.__test_photo_txt = config.test_photo_txt

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
        f_image_list = open(self.__test_photo_txt)
        image_list = [line.strip() for line in f_image_list.readlines()]

        print('length train images:', len(image_list))
        sum_train = 0
        sum_test = 0

        for class_name in classes:
            print(class_name)
            class_path = os.path.join(root, class_name)
            train_save_path = os.path.join(train_root, class_name)
            test_save_path = os.path.join(test_root, class_name)

            if not os.path.exists(train_save_path):
                os.mkdir(train_save_path)
                os.mkdir(test_save_path)

            images = os.listdir(class_path)

            for i, image in enumerate(images):
                src_path = os.path.join(class_path, image)
                cat_name = '/'.join([class_name, image])
                if cat_name in image_list:
                    dst_path = os.path.join(test_save_path, image)
                    sum_test += 1
                else:
                    dst_path = os.path.join(train_save_path, image)
                    sum_train += 1

                shutil.copy(src_path, dst_path)

        print('sum_train:', sum_train, 'sum_test:', sum_test)

if __name__ == '__main__':
    config = Config()
    divide = Divide(config)
    divide.divide()
