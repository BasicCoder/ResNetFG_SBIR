#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python train.py --name b48_f \
                                           --device 0,1 \
                                           --batch_size 48 \
                                           --net resnet101 \
                                           --weight_decay 0 \
                                           --weight_cat 1.0 \
                                           --niter 1000 \
                                           --niter_decay 9000 \
                                           --env caffe2torch_resnet101_b48_f
