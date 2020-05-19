#! /usr/bin/env bash
python train_ppi.py \
    --num-layers 2 \
    --k 5 \
    --kq 5 \
    --epoch 20 \
    --lr 0.0003 \
    --gpu-id 0 \
    --graph-mode descriptor \
    --beta-percentile 98 \
    --batch-size 0
