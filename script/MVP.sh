#!/bin/bash

python ../main.py \
  --data-root='/data/DataSets/' \
  --dataset='PACS' \
  --save-path='mvrml' \
  --do-train=False \
  --gpu=0 \
  \
  --MVP \
  --MVP-bs 32 \
  --batch-size=16 \
  --model='erm' \
  \
  --eval='mvp' \
  --exp-num -2 \
  --start-time=0 \
  --times=3
