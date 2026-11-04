# certificate/ — index

The certificate is **`c42_profile_0.686360751.json`**: exact profile (T, C, rational breaks, decimal phases).
Its format is specified in the repository README ("Certificate format"); the mathematics in NOTES.md.

| file | what |
|---|---|
| `c42_profile_0.686360751.json` | the certificate (claim C = T = 0.686360751) |
| `SHA256SUMS` | SHA-256 of the certificate, of checker #1 (`checkers/rigorous_check.py`, `checkers/spectral_witness.py`) and of checker #2 (`checkers/independent_check.py`) |
| `stability_N73728_P10.json` | checker #1, doubled grid: the tightest enclosure (PASS) |
| `stability_N36864_P12.json` | checker #1, higher degree (PASS) |
| `expected_N36864_P10.json` | checker #1 at the quick setting used by the README and the corruption tests (PASS) |
| `*.err` | progress logs of the same runs |
| `runs_superseded/` | runs on an earlier copy of the certificate that differed only in the descriptive `checker` field (N = 18432 / 36864 settings, incl. loose FAILs); see NOTES.md §8.3 |
| `candidates/` | session S3 feasibility runs on a different (96-piece, S1) profile; **not** part of this certificate |

All three checker #1 outputs above are from session S7, with the checker version in `SHA256SUMS`. In each
checker #1 output JSON, `certificate_sha256` is the SHA-256 of the input file, and `checker_sha256` that of
`rigorous_check.py`.

Checker #2's outputs on the certificate are in `checkers/independent_out/cert_N*_P*.json` (`.log` = stdout, `.err`
= progress log).
