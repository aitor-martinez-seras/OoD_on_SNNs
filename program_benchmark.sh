#!/usr/bin bash

CUDA_VISIBLE_DEVICES=1 python main.py --conf rgb-benchmark --n-hidden-layers 10 --f-max 200 --n-time-steps 64 --cluster-mode labels
CUDA_VISIBLE_DEVICES=1 python main.py --conf rgb-benchmark --n-hidden-layers 10 --f-max 200 --n-time-steps 64 --cluster-mode predictions