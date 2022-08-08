#!/bin/bash


python main.py \
  --data-root='/data/DataSets/' \
  --dataset='PACS' \
  --save-path='mvrml' \
  --gpu=0 \
  --do-train=True \
  \
  --model='erm' \
  --backbone='resnet18' \
  --batch-size=64 \
  --workers=6 \
  --lr=0.05 \
  --momentum=0 \
  --nesterov=False \
  --num-epoch=30 \
  \
  --trajectory=3 \
  --length=3 \
  \
  --exp-num -2 \
  --start-time=0 \
  --times=3 \
  --train='mvrml_p' \
  --eval='deepall' \
  --loader='meta'
