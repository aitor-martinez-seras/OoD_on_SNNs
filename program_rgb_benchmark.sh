#!/usr/bin bash

CUDA_VISIBLE_DEVICES=2 python main.py --conf rgb-benchmark --n-hidden-layers 24 --f-max 200 --n-time-steps 64 --cluster-mode correct-predictions --save-histograms-for scp --save-metric-plots
CUDA_VISIBLE_DEVICES=2 python main.py --conf rgb-benchmark --n-hidden-layers 24 --f-max 200 --n-time-steps 64 --cluster-mode predictions --save-histograms-for scp --save-metric-plots
CUDA_VISIBLE_DEVICES=2 python main.py --conf rgb-benchmark --n-hidden-layers 24 --f-max 200 --n-time-steps 64 --cluster-mode labels --save-histograms-for scp --save-metric-plots