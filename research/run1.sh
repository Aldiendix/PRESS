#!/bin/bash
# run1.sh NAME FEATDIR HASH2 [extra train args]  -> trains on rounds 40-68, builds nothing (research eval via selftrain2)
n=$1; feat=$2; h2=$3; shift 3
FEAT=$feat HASH2=$h2 OMP_NUM_THREADS=5 .venv-research/bin/python research/train.py --threads 5 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 --idf 1 --rows 4 --wx 0.5 --train 40 68 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
FEAT=$feat HASH2=$h2 OMP_NUM_THREADS=1 .venv-research/bin/python research/selftrain2.py cache/W_$n.npy 8192 16384 2>&1 | grep -v -i warn > research/st_$n.log
touch research/done_$n
