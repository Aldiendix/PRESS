#!/bin/bash
# tune_post.sh LABEL SOCIAL TITLE REJOIN -> score on three unseen sets
lab=$1; so=$2; ti=$3; rj=$4; out=""
for cfg in "w2:69 70 71 72 73" "nU72:73" "nU73:74"; do t=${cfg%%:*}; r=${cfg#*:}
  .venv/bin/python tools/build.py --table cache/W_$t --fw 8192 --fc 16384 --d 96 --idf 1 --social "$so" --title "$ti" --refine "(90, 10.0, 0.75)" --rejoin $rj --out dist/tp_$lab_$t.py > /dev/null 2>&1
  out="$out $(.venv/bin/python tools/evaluate.py dist/tp_$lab_$t.py --rounds $r --jobs 5 2>&1 | grep MEAN | awk '{print $2"/"$4"/"$6}')"; done
echo "$lab |$out"
