#!/usr/bin bash

CUDA_VISIBLE_DEVICES=0 python main.py --conf rgb-benchmark --n-hidden-layers 24 --f-max 200 --n-time-steps 64 --cluster-mode correct-predictions
CUDA_VISIBLE_DEVICES=0 python main.py --conf rgb-benchmark --n-hidden-layers 24 --f-max 200 --n-time-steps 64 --cluster-mode predictions
CUDA_VISIBLE_DEVICES=0 python main.py --conf rgb-benchmark --n-hidden-layers 24 --f-max 200 --n-time-steps 64 --cluster-mode labels