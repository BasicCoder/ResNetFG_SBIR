#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --name b16 \
                                           --device 0,1 \
                                           --batch_size 16 \
                                           --net resnet50 \
                                           --weight_decay 0 \
                                           --weight_cat 1.0 \
                                           --epoch_count 0 \
                                           --niter 2000 \
                                           --lr  1e-5 \
                                           --niter_decay 2000 \
                                           --fine_tune False \
                                           --save_dir /home/bc/Work/caffe2torch/checkpoints \
                                           --model_root /home/bc/Work/caffe2torch/checkpoints/resnet50_b48_f2/photo/photo_resnet50_best.pth \
                                           --env caffe2torch_resnet50_b16
