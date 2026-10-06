#!/bin/bash
B="--fw 8192 --fc 16384 --d 96 --idf 1 --rows 4 --wx 0.5 --train 40 68 --steps 6000 --nz_start 0.3 --anneal 0.5"
run(){ n=$1; shift 1; OMP_NUM_THREADS=5 .venv-research/bin/python research/train.py --threads 5 $B --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
  FORCE=1 .venv/bin/python tools/build.py --table cache/W_$n --fw 8192 --fc 16384 --d 96 --idf 1 --social "(20, 0.4, 4, 120, 0.3)" --title "(35, 0.4, 4, 80, 0.4)" --refine "(90, 10.0, 0.75)" --out dist/test_$n.py > research/build_$n.log 2>&1; touch research/done_$n; }
run b1 --nz 0.066 &
run b2 --nz 0.0165 &
run b3 --nz 0.13 &
wait
