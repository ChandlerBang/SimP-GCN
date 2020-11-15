#!/bin/bash
python ./src/train_simp.py \
    --dataset film \
    --ptb_rate 0 \
    --bias_init 0 \
    --k 20 \
    --gamma 0.1 \
    --lambda_ 1 \
    --seed 15 \
    --epochs 500 \
    --lr 0.05 \
    --hidden 16 \
    --weight_decay 5e-04 \
    --ssl PairwiseAttrSim \
    --datapath data// \
    --type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 0 \
    --early_stopping 100 \
    --sampling_percent 1 \
    --dropout 0.5 \
    --normalization AugNormAdj --task_type semi \
     \

