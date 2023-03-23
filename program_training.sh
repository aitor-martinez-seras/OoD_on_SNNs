#!/usr/bin bash

CUDA_VISIBLE_DEVICES=0 python train.py --conf datasets --dataset CIFAR10 --model ConvNet --arch-selector 10 --f-max 200 -b 128 --lr 0.002 --epochs 75 --weight-decay 0.00001 --opt adamw --n-time-steps 64 --save-every 10  --lr-decay-milestones 10 20 25 30 --lr-decay-rate 0.8 --penultimate-layer-neurons 100