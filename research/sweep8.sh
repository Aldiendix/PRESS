#!/bin/bash
C="--fw 8192 --fc 16384 --d 96 --steps 6000 --idf 1 --rows 4 --wx 0.5"
(OMP_NUM_THREADS=6 .venv-research/bin/python research/train.py --threads 6 $C --nz 0.035 --train 40 73 --out cache/W_final1.npy > research/train_final1.log 2>&1; touch research/done_final1) &
for sd in 1 2; do (OMP_NUM_THREADS=4 .venv-research/bin/python research/train.py --threads 4 $C --nz 0.033 --train 40 68 --seed $sd --out cache/W_w2s$sd.npy > research/train_w2s$sd.log 2>&1; touch research/done_w2s$sd) & done
wait
