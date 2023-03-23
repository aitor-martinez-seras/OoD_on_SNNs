#!/usr/bin bash

CUDA_VISIBLE_DEVICES=0 python main.py --conf bw-benchmark --arch-selector 2 --pretrained --f-max 100 --n-time-steps 50