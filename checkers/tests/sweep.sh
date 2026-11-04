#!/bin/bash
# Gate (c) sweep: certified F_upper vs (N, P, refine). Usage: sweep.sh <profile> <tag> "N P r" ...
prof=$1; tag=$2; shift 2
cd "$(dirname "$0")/../.."
for cfg in "$@"; do
  set -- $cfg
  out=logs/cert/sweep_${tag}_N$1_P$2_r$3.json
  [ -s "$out" ] && continue
  python checkers/rigorous_check.py "$prof" --N $1 --P $2 --refine $3 --quiet > "$out" 2> "${out%.json}.err"
done
