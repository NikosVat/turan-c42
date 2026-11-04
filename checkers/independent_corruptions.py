#!/usr/bin/env python3
"""Run checker #2 (checkers/independent_check.py) on corrupted certificates.

  python checkers/independent_corruptions.py [--jobs 2]

Inputs:
  1. every JSON in checkers/corruptions/ and checkers/corruptions/claim/ (made in earlier sessions), as is;
  2. for the three old-format Bakeas files (no "C", knots given as the text "j/48, j = 0..24") and for the
     Pinho files without "C", an adapted copy in the certificate format with C := T (written to
     checkers/independent_out/corruptions/adapted/);
  3. new corruptions of the certificate generated here (checkers/independent_out/corruptions/new/).
Outputs (JSON + stderr log per case) and SUMMARY.txt go to checkers/independent_out/corruptions/.
Exit code 0 iff every verdict equals the expected one.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "independent_out", "corruptions")
CERT = os.path.join(ROOT, "certificate", "c42_profile_0.686360751.json")

# expected verdicts (from each file's own "note"/"why" field, NOTES.md 8.3, and the format rules)
EXPECT = {
    "bakeas_T_0.68636.json": "REFUSED",            # old format: no C, no breaks
    "bakeas_phase0_plus_5e-2.json": "REFUSED",
    "bakeas_shift_a_negative.json": "REFUSED",
    "pinho_T76_claim_1e-9_above_F.json": "PASS",   # resolution control
    "pinho_T76_claim_1e-9_below_F.json": "FAIL",
    "pinho_T_below_fixed_point.json": "REFUSED",   # no C
    "pinho_claim_below_T.json": "REFUSED",         # C < T
    "pinho_phase3_plus_1e-2.json": "REFUSED",      # no C
    "claim/T74_claim_1e-9_above_F.json": "PASS",   # resolution control
    "claim/T74_claim_1e-9_below_F.json": "FAIL",
    "claim/T_0.68636074_C_eq_T.json": "FAIL",
    "claim/T_0.68636075_below_fixed_point.json": "FAIL",
    "claim/phase12_plus_0.005.json": "FAIL",
    "claim/phase5_plus_0.002.json": "FAIL",
    # adapted copies (C := T)
    "adapted/bakeas_T_0.68636.json": "FAIL",
    "adapted/bakeas_phase0_plus_5e-2.json": "FAIL",
    "adapted/bakeas_shift_a_negative.json": "REFUSED",
    "adapted/pinho_T_below_fixed_point.json": "FAIL",
    "adapted/pinho_phase3_plus_1e-2.json": "FAIL",
    # new corruptions of the certificate
    "new/T_C_0.686360749.json": "FAIL",
    "new/phase20_plus_1e-3.json": "FAIL",
    "new/C_below_T.json": "REFUSED",
    "new/C_eq_1.json": "REFUSED",
    "new/a_negative.json": "REFUSED",
    "new/misaligned_knot.json": "REFUSED",       # run with --N 36864
    "new/phase_count_mismatch.json": "REFUSED",
    "new/T_as_float.json": "REFUSED",
    "new/knots_not_increasing.json": "REFUSED",
}
EXTRA_ARGS = {"new/misaligned_knot.json": ["--N", "36864"]}


def make_inputs():
    os.makedirs(os.path.join(OUT, "adapted"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "new"), exist_ok=True)
    src = os.path.join(HERE, "corruptions")
    for name in ("bakeas_T_0.68636.json", "bakeas_phase0_plus_5e-2.json", "bakeas_shift_a_negative.json",
                 "pinho_T_below_fixed_point.json", "pinho_phase3_plus_1e-2.json"):
        c = json.load(open(os.path.join(src, name)))
        if "breaks" not in c:
            assert c["knots"].replace(" ", "") == "j/48,j=0..24", c["knots"]
            c["breaks"] = ["0"] + ["%d/48" % j for j in range(1, 25)]
            c["breaks"] = [b if b != "24/48" else "1/2" for b in c["breaks"]]
        c["constant"] = "C_42"
        c["phases"] = [str(x) for x in c["phases"]]
        c.setdefault("C", c["T"])
        c["adapted_by"] = "checkers/independent_corruptions.py: certificate format, C := T"
        json.dump(c, open(os.path.join(OUT, "adapted", name), "w"), indent=1)
    base = json.load(open(CERT))
    def new(name, **kw):
        c = json.loads(json.dumps(base))
        for k, v in kw.items():
            c[k] = v
        c["corruption"] = name
        json.dump(c, open(os.path.join(OUT, "new", name), "w"), indent=1)
    new("T_C_0.686360749.json", T="0.686360749", C="0.686360749")
    ph = list(base["phases"])
    ph[20] = repr(float(ph[20]) + 1e-3)
    new("phase20_plus_1e-3.json", phases=ph)
    new("C_below_T.json", C="0.68636075")
    new("C_eq_1.json", C="1")
    new("a_negative.json", phases=[repr(float(x) - 0.2) for x in base["phases"]])
    br = list(base["breaks"])
    br[1] = "1/18433"
    new("misaligned_knot.json", breaks=br)
    new("phase_count_mismatch.json", phases=base["phases"][:-1])
    new("T_as_float.json", T=0.686360751)
    br = list(base["breaks"])
    br[2], br[3] = br[3], br[2]
    new("knots_not_increasing.json", breaks=br)


def path_of(key):
    if key.startswith("adapted/") or key.startswith("new/"):
        return os.path.join(OUT, key)
    return os.path.join(HERE, "corruptions", key)


def run(key):
    tag = key.replace("/", "__").replace(".json", "")
    outj = os.path.join(OUT, tag + ".json")
    errf = os.path.join(OUT, tag + ".err")
    cmd = [sys.executable, os.path.join(HERE, "independent_check.py"), path_of(key), "--out", outj]
    cmd += EXTRA_ARGS.get(key, [])
    with open(errf, "w") as fe:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=fe, text=True)
    try:
        r = json.load(open(outj))
    except Exception:
        r = {"verdict": "CRASH(exit %d)" % p.returncode}
    return key, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    make_inputs()
    keys = [k for k in EXPECT if args.only is None or args.only in k]
    # quick (REFUSED) ones first
    keys.sort(key=lambda k: EXPECT[k] != "REFUSED")
    with ThreadPoolExecutor(args.jobs) as ex:
        results = list(ex.map(run, keys))
    lines = ["%-46s %-9s %-9s %-24s %s" % ("case", "expected", "got", "F_upper", "reason / margin")]
    ok = True
    for key, r in results:
        v = r.get("verdict")
        good = v == EXPECT[key]
        ok &= good
        extra = r.get("reason", "")
        if v in ("PASS", "FAIL"):
            extra = "C = %s, C - F_upper = %.3e" % (r.get("C"), r.get("margin_C_minus_F_upper", float("nan")))
        lines.append("%-46s %-9s %-9s %-24s %s%s" % (key, EXPECT[key], v, r.get("F_upper_str", "-")[:22],
                                                    extra, "" if good else "   <-- MISMATCH"))
    lines.append("ALL AS EXPECTED" if ok else "SOME VERDICTS DIFFER FROM EXPECTATION")
    txt = "\n".join(lines)
    print(txt)
    if args.only is None:
        open(os.path.join(OUT, "SUMMARY.txt"), "w").write(txt + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
