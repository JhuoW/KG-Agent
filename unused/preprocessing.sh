#!/bin/bash

python build_gt_path_gcr.py \
    --data_path rmanluo \
    --dataset RoG-webqsp \
    --split train \
    --output_path data/shortest_gt_paths \
    --num_processes 64

# Optional: Add --undirected flag if you want to treat the graph as undirected
# python build_gt_path.py \
#     --data_path rmanluo \
#     --dataset RoG-webqsp \
#     --split train \
#     --output_path data/shortest_gt_paths \
#     --undirected \
#     --num_processes 8
