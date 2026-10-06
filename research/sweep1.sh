#!/bin/bash
# capacity sweep (unquantized): name fw fc d extra
run(){ n=$1; fw=$2; fc=$3; d=$4; shift 4; OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw $fw --fc $fc --d $d --steps 5000 --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1; .venv-research/bin/python research/evalw.py cache/W_$n.npy $fw $fc 71 72 73 2>&1 | grep -v -i warn > research/eval_$n.log; }
run b 8192 4096 96 &
run c 32768 16384 48 &
run d 16384 0 48 &
run e 8192 4096 48 --wx 0.5 &
wait
