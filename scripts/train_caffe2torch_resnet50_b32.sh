#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --name b32 \
                                           --device 0,1 \
                                           --batch_size 32 \
                                           --net resnet50 \
                                           --weight_decay 0 \
                                           --weight_cat 1.0 \
                                           --weight_tri 1.0 \
                                           --epoch_count 3670 \
                                           --niter 2000 \
                                           --lr  1e-5 \
                                           --niter_decay 2000 \
                                           --fine_tune True \
                                           --save_dir /home/bc/Work/caffe2torch/checkpoints \
                                           --model_root /home/bc/Work/caffe2torch/checkpoints/resnet50_b32/photo/photo_resnet50_3670.pth \
                                           --env caffe2torch_resnet50_b32
