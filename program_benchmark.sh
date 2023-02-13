#!/usr/bin bash

CUDA_VISIBLE_DEVICES=1 python main.py --conf bw-benchmark --n-hidden-layers 1 --f-max 200 --n-time-steps 64 --cluster-mode predictions --use-only-correct-test-images
CUDA_VISIBLE_DEVICES=1 python main.py --conf bw-benchmark --n-hidden-layers 1 --f-max 200 --n-time-steps 64 --cluster-mode correct-predictions --use-only-correct-test-images
CUDA_VISIBLE_DEVICES=1 python main.py --conf bw-benchmark --n-hidden-layers 1 --f-max 200 --n-time-steps 64 --cluster-mode labels --use-only-correct-test-images
