"""Markdown table of certifier runs: python checkers/tests/table.py logs/cert/sweep_<tag>_*.json"""
import json, sys, re


def num(s):
    if s is None:
        return None
    m = re.match(r"\[?\s*([-0-9.e+]+)", s)
    return float(m.group(1)) if m else None


rows = []
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append(d | {"file": f})
rows.sort(key=lambda d: (d.get("N", 0), d.get("P", 0), d.get("refine", 0)))
print("| N | P | refine | verdict | sup R/t | EN | ED | F_upper | F_lower | width | s | GB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in rows:
    print(f"| {d.get('N')} | {d.get('P')} | {d.get('refine')} | {d.get('verdict')} | {num(d.get('residual_sup')):.2e} | "
          f"{num(d.get('EN')):.2e} | {num(d.get('ED')):.2e} | {(d.get('F_upper') or '')[1:20]} | "
          f"{(d.get('F_lower') or '')[1:20]} | {num(d.get('enclosure_width')) or float('nan'):.2e} | {d.get('seconds')} | "
          f"{d.get('max_rss_GB')} |")
