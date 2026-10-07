#!/bin/bash
# run6.sh NAME LO FW FC D NZ [args]: v4 recipe trained on LO..60, scored on rounds 61-74 (v4.1 pipeline)
n=$1; lo=$2; fw=$3; fc=$4; d=$5; nz=$6; shift 6
OMP_NUM_THREADS=5 .venv-research/bin/python research/train.py --threads 5 --fw $fw --fc $fc --d $d --nz $nz --steps 6000 --idf 1 --wx 0.5 --train $lo 60 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
OMP_NUM_THREADS=1 .venv-research/bin/python research/eval_D.py cache/W_$n.npy $fw $fc 5 2>&1 | grep round > research/D_$n.log
touch research/done_$n
