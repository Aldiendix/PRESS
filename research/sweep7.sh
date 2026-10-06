#!/bin/bash
run(){ n=$1; feat=$2; shift 2; FEAT=$feat OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 --idf 1 --rows 4 --wx 0.5 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1; FEAT=$feat IDF=1 .venv-research/bin/python research/evalw.py cache/W_$n.npy 8192 16384 71 72 73 2>&1 | grep -v -i warn > research/eval_$n.log; touch research/done_$n; }
run w1 cache/feat2 --train 30 68 &
run w2 cache/feat --train 40 68 &
run w3 cache/feat --train 50 68 &
run w4 cache/feat2 --train 40 68 &
wait
