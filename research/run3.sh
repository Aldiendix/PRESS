#!/bin/bash
# run3.sh NAME [train args]: train 40-68 (3 threads), then lab-style eval on set A via a temp table
n=$1; shift 1
OMP_NUM_THREADS=3 .venv-research/bin/python research/train.py --threads 3 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 --idf 1 --rows 4 --train 40 68 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
OMP_NUM_THREADS=1 .venv-research/bin/python research/noise2.py cache/W_$n.npy 2>&1 | grep -v -i warn | head -2 > research/nz_$n.log
touch research/done_$n
