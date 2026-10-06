#!/bin/bash
run(){ n=$1; fw=$2; fc=$3; d=$4; nz=$5; shift 5; OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw $fw --fc $fc --d $d --nz $nz --steps 6000 --idf 1 --rows 4 --wx 0.5 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1; IDF=1 .venv-research/bin/python research/evalw.py cache/W_$n.npy $fw $fc 71 72 73 2>&1 | grep -v -i warn > research/eval_$n.log; touch research/done_$n; }
run u1 16384 16384 64 0.038 &
run u2 16384 32768 48 0.031 &
run u3 32768 32768 32 0.033 &
run u4 8192 16384 96 0.035 &
wait
