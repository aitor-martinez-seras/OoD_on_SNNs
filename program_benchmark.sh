#!/usr/bin bash

CUDA_VISIBLE_DEVICES=1 python main.py --conf bw-benchmark --n-hidden-layers 1 --pretrained --f-max 100 --n-time-steps 50 --cluster-mode predictions --use-only-correct-test-images
CUDA_VISIBLE_DEVICES=1 python main.py --conf bw-benchmark --n-hidden-layers 1 --pretrained --f-max 100 --n-time-steps 50 --cluster-mode correct-predictions --use-only-correct-test-images
CUDA_VISIBLE_DEVICES=1 python main.py --conf bw-benchmark --n-hidden-layers 1 --pretrained --f-max 100 --n-time-steps 50 --cluster-mode labels --use-only-correct-test-images
