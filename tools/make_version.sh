#!/bin/bash
# Build a new version with the current recipe.
# Usage: tools/make_version.sh <version name, e.g. 5> <last finished round> [17|19]
# Fetches the last round, caches features, trains the table on rounds 40..LAST, packs it into the stacked
# single-file template (learned singleton ranker + language split + link rule, compact container) and prints the
# file's score on the last two rounds (inside the training data: a sanity check, not an estimate).
set -e
N=$1; LAST=$2; FEAT=${3:-19}; cd "$(dirname "$0")/.."
ls data/round_$(printf %04d $LAST)/*.parquet >/dev/null 2>&1 || .venv/bin/python tools/fetch.py $LAST | tail -5
.venv-research/bin/python research/feats.py > /dev/null
OMP_NUM_THREADS=14 .venv-research/bin/python research/train.py --threads 14 --fw 8192 --fc 16384 --d 96 --nz 0.033 --steps 6000 \
  --idf 1 --rows 4 --wx 0.5 --nz_start 0.3 --anneal 0.5 --train 40 $LAST --out cache/W_v$N.npy > research/train_v$N.log 2>&1
mkdir -p versions/v$N
TPL=tools/stack/mini_template.py; [ "$FEAT" = 19 ] && TPL=tools/stack/mini19_template.py
.venv/bin/python tools/stack/build.py --table cache/W_v$N --mini --template $TPL --out versions/v$N/solution.py 2>&1 | grep chars
.venv/bin/python tools/evaluate.py versions/v$N/solution.py --rounds $((LAST-1)) $LAST --jobs 8 2>&1 | tail -2
