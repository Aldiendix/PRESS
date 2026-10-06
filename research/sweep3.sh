#!/bin/bash
run(){ n=$1; idf=$2; st=$3; shift 3; OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw 8192 --fc 4096 --d 48 --tern 1.6 --steps $st --idf $idf --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1; IDF=$idf .venv-research/bin/python research/evalw.py cache/W_$n.npy 8192 4096 71 72 73 2>&1 | grep -v -i warn > research/eval_$n.log; touch research/done_$n; }
run r1 1 5000 &
run r2 0 5000 --rows 8 &
run r3 1 5000 --rows 8 --wx 0.5 &
run r4 0 15000 &
wait
