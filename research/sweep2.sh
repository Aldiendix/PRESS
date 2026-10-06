#!/bin/bash
run(){ n=$1; fw=$2; fc=$3; d=$4; shift 4; OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw $fw --fc $fc --d $d --steps 5000 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1; .venv-research/bin/python research/evalw.py cache/W_$n.npy $fw $fc 71 72 73 2>&1 | grep -v -i warn > research/eval_$n.log; }
run q1 8192 4096 48 --tern 1.6 &
run q2 8192 4096 32 --tern 1.1 &
run q3 8192 4096 24 --tern 0.5 &
run q4 16384 8192 24 --tern 1.6 &
wait
