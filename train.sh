#! /usr/bin/env bash
python train.py \
    --emb-file whole_graph_node2vec_pathway_walk_num_64_len_16.embs.txt \
    --num-layers 2 \
    --hidden-units 128 \
    --k 5 \
    --kq 5 \
    --epoch 20 \
    --lr 0.0003 \
    --graph-mode descriptor \
    --beta-percentile 98 \
    --batch-size 2048
    # --gpu-id 0 \