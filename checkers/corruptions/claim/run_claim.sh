#!/bin/bash
# Corruption tests for the claimed certificate (certificate/c42_profile_0.686360751.json).
# Usage: checkers/corruptions/claim/run_claim.sh [list-file]   (default: list.txt)
# Every line: file N P expected.  Runs 2 jobs in parallel (each ~2.3 GB at N=18432, P=14).
# Full checker outputs go to checkers/corruptions/claim/out/<name>.json.  Runtime ~6 min per job.
cd "$(dirname "$0")/../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
D=checkers/corruptions/claim
mkdir -p $D/out
run() {  # verdicts expected: see list.txt / list2.txt
  f=$1; N=$2; P=$3; exp=$4
  python checkers/rigorous_check.py $D/$f --N $N --P $P --refine 1 --quiet > $D/out/${f%.json}.json 2>/dev/null
  v=$(python -c "import json,sys; d=json.load(open('$D/out/${f%.json}.json')); print(d['verdict'], d.get('F_upper'), d.get('F_lower'), d.get('C'), d.get('reason',''))")
  set -- $v
  if [ "$1" = "$exp" ]; then s=OK; else s=MISMATCH; fi
  echo "$s  $f  N=$N P=$P expected=$exp got=$v"
}
export -f run; export D
LIST=${1:-$D/list.txt}
grep -v '^#' $LIST | grep . | xargs -P 2 -L 1 bash -c 'run "$@"' _ | tee $D/out/summary_$(basename $LIST .txt).txt
! grep -q MISMATCH $D/out/summary_$(basename $LIST .txt).txt
