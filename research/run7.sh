#!/bin/bash
# run7.sh NAME LO [args]: v4 recipe trained on LO..68, scored on rounds 69-74 (v4.1 pipeline)
n=$1; lo=$2; shift 2
OMP_NUM_THREADS=5 .venv-research/bin/python research/train.py --threads 5 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 --idf 1 --wx 0.5 --rows 4 --nz_start 0.3 --anneal 0.5 --train $lo 68 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
MINR=69 OMP_NUM_THREADS=1 .venv-research/bin/python research/eval_D.py cache/W_$n.npy 8192 16384 5 2>&1 | grep round > research/D_$n.log
touch research/done_$n
