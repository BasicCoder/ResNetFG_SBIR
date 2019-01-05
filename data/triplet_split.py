import os
import shutil

data_root = '/home/bc/Work/Database/sketchy/256x256/sketch/tx_000100000000'  # the data set location
t_ratio = 0.9  # the ratio of the train set

train_set_root = '/home/bc/Work/caffe2torch/data/dataset/sketch-triplet-train-all'  # where to save the training set
test_set_root = '/home/bc/Work/caffe2torch/data/dataset/sketch-triplet-test-all'  # where to save the testing set

photo_root = '/home/bc/Work/Database/sketchy/256x256/photo/tx_000100000000'
train_photo_root = '/home/bc/Work/caffe2torch/data/dataset/photo-train-all'
test_photo_root = '/home/bc/Work/caffe2torch/data/dataset/photo-test-all'

cnames = sorted(os.listdir(data_root))

for cname in cnames:
    fnames = sorted(os.listdir(os.path.join(train_photo_root, cname)))
    for fname in fnames:

        snames = sorted(os.listdir(os.path.join(data_root, cname)))
        f_name = fname.split('.')[0]
        c_path = os.path.join(train_set_root, cname)
        for sname in snames:

            #s_name = sname.split('.')[0]
            if sname.find(f_name) != -1:
                if not os.path.exists(c_path):
                    os.mkdir(c_path)

                dst_path = os.path.join(c_path, sname)
                src_path = os.path.join(data_root, cname, sname)
                print(src_path)
                shutil.copy(src_path, dst_path)


for cname in cnames:
    fnames = sorted(os.listdir(os.path.join(test_photo_root, cname)))
    for fname in fnames:

        snames = sorted(os.listdir(os.path.join(data_root, cname)))
        f_name = fname.split('.')[0]
        c_path = os.path.join(test_set_root, cname)
        for sname in snames:

            #s_name = sname.split('.')[0]
            if sname.find(f_name) != -1:
                if not os.path.exists(c_path):
                    os.mkdir(c_path)

                dst_path = os.path.join(c_path, sname)
                src_path = os.path.join(data_root, cname, sname)
                print(src_path)
                shutil.copy(src_path, dst_path)