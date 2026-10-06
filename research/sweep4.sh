#!/bin/bash
run(){ n=$1; fw=$2; fc=$3; d=$4; t=$5; shift 5; OMP_NUM_THREADS=3 .venv-research/bin/python research/train.py --threads 3 --fw $fw --fc $fc --d $d --tern $t --steps 6000 --idf 1 --rows 8 --wx 0.5 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1; IDF=1 .venv-research/bin/python research/evalw.py cache/W_$n.npy $fw $fc 71 72 73 2>&1 | grep -v -i warn > research/eval_$n.log; touch research/done_$n; }
run s1 8192 4096 48 1.2 &
run s2 16384 8192 48 1.9 &
run s3 8192 8192 64 1.6 &
run s4 8192 4096 96 1.8 &
wait
