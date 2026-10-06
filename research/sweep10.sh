#!/bin/bash
B="--fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 --idf 1 --rows 4 --wx 0.5 --train 40 68 --extra 9001 9050"
run(){ n=$1; shift 1; OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 $B --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
  .venv/bin/python tools/build.py --table cache/W_$n --fw 8192 --fc 16384 --d 96 --idf 1 --social "(20, 0.4, 4, 120, 0.3)" --title "(35, 0.4, 4, 80, 0.4)" --out dist/test_$n.py > research/build_$n.log 2>&1; touch research/done_$n; }
run y1 --wt 1.0 &
run y2 --wt 0.0 &
wait
