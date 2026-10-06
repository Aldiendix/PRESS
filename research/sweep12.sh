#!/bin/bash
B="--fw 8192 --fc 16384 --d 96 --nz 0.033 --idf 1 --rows 4 --wx 0.5 --train 40 68"
run(){ n=$1; shift 1; OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 $B --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
  .venv/bin/python tools/build.py --table cache/W_$n --fw 8192 --fc 16384 --d 96 --idf 1 --social "(20, 0.4, 4, 120, 0.3)" --title "(35, 0.4, 4, 80, 0.4)" --refine "(90, 10.0, 0.75)" --out dist/test_$n.py > research/build_$n.log 2>&1; touch research/done_$n; }
run a1 --steps 6000 --nz_start 1.0 --anneal 0.6 &
run a2 --steps 6000 --nz_start 0.3 --anneal 0.5 &
run a3 --steps 10000 --nz_start 1.0 --anneal 0.7 &
run a4 --steps 6000 --nz_start 1.0 --anneal 0.85 &
wait
