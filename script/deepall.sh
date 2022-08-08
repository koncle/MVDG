#!/bin/bash

python main.py \
  --data-root='/data/DataSets/' \
  --dataset='PACS' \
  --save-path='deepall' \
  --gpu=0 \
  --do-train=True \
  --lr=1e-3 \
  --model='erm' \
  --backbone='resnet18' \
  --batch-size=128 \
  --num-epoch=30 \
  \
  --exp-num=-2 \
  --start-time=0 \
  --times=5 \
  --train='deepall' \
  --eval='deepall'