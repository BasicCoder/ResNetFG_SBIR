#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --name b48_f \
                                           --device 0,1 \
                                           --batch_size 48 \
                                           --net resnet101 \
                                           --weight_decay 0 \
                                           --weight_cat 1.0 \
                                           --epoch_count 3000 \
                                           --niter 2000 \
                                           --niter_decay 1500 \
                                           --fine_tune True \
                                           --save_dir /home/bc/Work/caffe2torch/checkpoints \
                                           --model_root /home/bc/Work/caffe2torch/checkpoints/resnet101_b48_f/photo/photo_resnet101_best.pth \
                                           --env caffe2torch_resnet101_b48_f
