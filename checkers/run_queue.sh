#!/bin/bash
# Resumable job queue for checker #1. Each line of the job file: <profile> <N> <P> <outfile>.
# A job is skipped if <outfile> already contains a verdict. Runs 2 jobs at a time, 1 BLAS thread each.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
job() {
  prof=$1; N=$2; P=$3; out=$4
  if grep -q '"verdict"' "$out" 2>/dev/null; then echo "skip $out"; return; fi
  python checkers/rigorous_check.py "$prof" --N $N --P $P --refine 1 > "$out.tmp" 2> "${out%.json}.err" && true
  mv "$out.tmp" "$out"
  python -c "import json; d=json.load(open('$out')); print('done', '$out', d['verdict'], d.get('F_upper'), d.get('F_lower'), d.get('seconds'), d.get('max_rss_GB'), d.get('reason',''))"
}
export -f job
grep -v '^#' "$1" | grep . | xargs -P ${JOBS:-2} -L 1 bash -c 'job "$@"' _
