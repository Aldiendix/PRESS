#!/bin/bash
run(){ n=$1; shift 1; OMP_NUM_THREADS=3 .venv-research/bin/python research/train.py --threads 3 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 --idf 1 --rows 4 --wx 0.5 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1; IDF=1 .venv-research/bin/python research/evalw.py cache/W_$n.npy 8192 16384 71 72 73 2>&1 | grep -v -i warn > research/eval_$n.log; touch research/done_$n; }
run v1 &
run v2 --tau 0.07 &
run v3 --tau 0.15 &
run v4 --noise_neg 0 &
run v5 --train 30 68 &
wait
