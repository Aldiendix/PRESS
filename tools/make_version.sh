#!/bin/bash
# Build a new version with the current recipe.  Usage: tools/make_version.sh <version number> <last finished round>
# Fetches the last round, caches features, trains the table on rounds 40..LAST, builds versions/vN/solution.py
# and prints its score on the last two rounds (inside the training data: a sanity check, not an estimate).
set -e
N=$1; LAST=$2; cd "$(dirname "$0")/.."
ls data/round_$(printf %04d $LAST)/*.parquet >/dev/null 2>&1 || .venv/bin/python tools/fetch.py $LAST | tail -5
.venv-research/bin/python research/feats.py > /dev/null
OMP_NUM_THREADS=14 .venv-research/bin/python research/train.py --threads 14 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 \
  --idf 1 --rows 4 --wx 0.5 --nz_start 0.3 --anneal 0.5 --train 40 $LAST --out cache/W_v$N.npy > research/train_v$N.log 2>&1
mkdir -p versions/v$N
.venv/bin/python tools/build.py --table cache/W_v$N --fw 8192 --fc 16384 --d 96 --idf 1 --social "(20, 0.4, 4, 120, 0.2)" \
  --title "(35, 0.4, 4, 80, 0.3)" --refine "(90, 10.0, 0.75)" --rejoin 0 --out versions/v$N/solution.py
.venv/bin/python tools/evaluate.py versions/v$N/solution.py --rounds $((LAST-1)) $LAST --jobs 8 2>&1 | tail -2
