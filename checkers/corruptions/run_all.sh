#!/bin/bash
# Corruption tests for checkers/rigorous_check.py. Each line: file, grid N, degree P, expected verdict.
# A test passes iff the checker's verdict equals the expected one (FAIL / REFUSED; PASS only for the
# resolution test *_claim_1e-9_above_F.json).  Runtime ~25 min on 4 vCPU.
cd "$(dirname "$0")/../.."
ok=0; bad=0
while read -r f N P exp; do
  [ -z "$f" ] && continue
  v=$(python checkers/rigorous_check.py checkers/corruptions/$f --N $N --P $P --refine 1 --quiet 2>/dev/null \
      | python -c "import json,sys; print(json.load(sys.stdin)['verdict'])")
  if [ "$v" = "$exp" ]; then ok=$((ok+1)); s=OK; else bad=$((bad+1)); s=MISMATCH; fi
  echo "$s  $f  expected=$exp got=$v"
done <<'EOF'
bakeas_phase0_plus_5e-2.json 768 6 FAIL
bakeas_T_0.68636.json 768 6 FAIL
bakeas_shift_a_negative.json 768 6 REFUSED
pinho_claim_below_T.json 36864 8 REFUSED
pinho_T_below_fixed_point.json 36864 8 FAIL
pinho_phase3_plus_1e-2.json 36864 8 FAIL
pinho_T76_claim_1e-9_below_F.json 18432 8 FAIL
pinho_T76_claim_1e-9_above_F.json 18432 8 PASS
EOF
echo "corruption tests: $ok as expected, $bad mismatches"
[ $bad -eq 0 ]
