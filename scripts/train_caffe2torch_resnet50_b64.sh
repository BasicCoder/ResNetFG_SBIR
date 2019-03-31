#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python train.py --name b64_mutual \
                                           --device 0,1 \
                                           --batch_size 64 \
                                           --net resnet50 \
                                           --weight_decay 0 \
                                           --weight_cat 1.0 \
                                           --weight_tri 50.0 \
                                           --weight_mut 20.0 \
                                           --epoch_count 1000 \
                                           --niter 1500 \
                                           --lr  1e-5 \
                                           --niter_decay 1500 \
                                           --fine_tune True \
                                           --save_dir /home/bc/Work/caffe2torch/checkpoints \
                                           --model_root /home/bc/Work/caffe2torch/checkpoints/resnet50_b64_mutual/photo/photo_resnet50_1000.pth \
                                           --env caffe2torch_resnet50_b64_mutual
