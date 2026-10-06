#!/bin/bash
B="--steps 6000 --idf 1 --rows 4 --wx 0.5 --train 40 68"
run(){ n=$1; fw=$2; fc=$3; d=$4; shift 4; OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 --fw $fw --fc $fc --d $d $B --out cache/W_$n.npy "$@" > research/train_$n.log 2>&1
  .venv/bin/python tools/build.py --table cache/W_$n --fw $fw --fc $fc --d $d --idf 1 --social "(20, 0.4, 4, 120, 0.3)" --title "(35, 0.4, 4, 80, 0.4)" --out dist/test_$n.py > research/build_$n.log 2>&1; touch research/done_$n; }
run x1 8192 16384 96 --nz 0.033 --sm 2 &
run x2 8192 16384 96 --nz 0.033 --only arxiv &
run x3 8192 16384 128 --nz 0.026 &
run x4 8192 16384 96 --nz 0.033 --bs 3072 &
wait
