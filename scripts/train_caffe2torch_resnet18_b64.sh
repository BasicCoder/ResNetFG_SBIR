#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python train.py --name b64_mutual \
                                           --device 0 \
                                           --batch_size 64 \
                                           --net resnet18 \
                                           --weight_decay 0 \
                                           --weight_cat 1.0 \
                                           --weight_tri 50.0 \
                                           --weight_mut 0.0 \
                                           --epoch_count 2000 \
                                           --niter 1500 \
                                           --lr  1e-5 \
                                           --niter_decay 1500 \
                                           --fine_tune False \
                                           --save_dir /home/bc/Work/caffe2torch/checkpoints \
                                           --model_root /home/bc/Work/caffe2torch/checkpoints/resnet18_b64_mutual/photo/photo_resnet18_2000.pth \
                                           --env caffe2torch_resnet101_b64_mutual
