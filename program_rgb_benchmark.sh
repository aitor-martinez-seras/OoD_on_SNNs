#!/usr/bin bash

CUDA_VISIBLE_DEVICES=0 python main.py --conf rgb-benchmark --n-hidden-layers 24 --f-max 200 --n-time-steps 64