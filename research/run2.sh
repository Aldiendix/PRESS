#!/bin/bash
# run2.sh NAME LO T [train args]: train on rounds LO..T, build with self-training, score on round T+1 (unseen)
n=$1; lo=$2; T=$3; shift 3
OMP_NUM_THREADS=5 .venv-research/bin/python research/train.py --threads 5 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 --idf 1 --rows 4 --wx 0.5 --train $lo $T --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
.venv/bin/python tools/build.py --table cache/W_$n --fw 8192 --fc 16384 --d 96 --idf 1 --social "(20, 0.4, 4, 120, 0.3)" --title "(35, 0.4, 4, 80, 0.4)" --refine "(90, 10.0, 0.75)" --out dist/test_$n.py > research/build_$n.log 2>&1
.venv/bin/python tools/evaluate.py dist/test_$n.py --rounds $((T+1)) --jobs 4 2>&1 | grep MEAN | cut -c1-50 > research/next_$n.log
touch research/done_$n
