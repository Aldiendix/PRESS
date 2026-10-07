#!/bin/bash
# run5.sh NAME FW FC CW [args]: v4 recipe on rounds 40-68, research eval with self-training (v3 settings) on 69-73
n=$1; fw=$2; fc=$3; cw=$4; shift 4
CW=$cw OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw $fw --fc $fc --d 96 --nz $(python3 -c "print(round(0.033*24576/($fw+$fc),4))") --steps 6000 --idf 1 --rows 4 --wx 0.5 --nz_start 0.3 --anneal 0.5 --train 40 68 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
CW=$cw OMP_NUM_THREADS=1 .venv-research/bin/python research/selftrain2.py cache/W_$n.npy $fw $fc 2>&1 | grep -v -i warn > research/st_$n.log
touch research/done_$n
