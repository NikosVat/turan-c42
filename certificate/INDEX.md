# certificate/ — index

The certificate is **`c42_profile_0.686360751.json`**: exact profile (T, C, rational breaks, decimal phases).
Its format is specified in the repository README ("Certificate format"); the mathematics in NOTES.md.

| file | what |
|---|---|
| `c42_profile_0.686360751.json` | the certificate (claim C = T = 0.686360751) |
| `SHA256SUMS` | SHA-256 of the certificate and of checker #1 (`checkers/rigorous_check.py`, `checkers/spectral_witness.py`) |
| `expected_N36864_P10.json` | checker #1 output at the primary setting (PASS, F ∈ [0.6863607487772, 0.6863607500725]) |
| `stability_N36864_P12.json` | different degree (PASS, F ∈ [0.6863607494100, 0.6863607494397]) |
| `stability_N73728_P10.json` | doubled grid (PASS, F ∈ [0.6863607494207, 0.6863607494290]) |
| `*.err` | progress logs of the same runs |
| `runs_superseded/` | runs on an earlier copy of the certificate that differed only in the descriptive `checker` field (N = 18432 / 36864 settings, incl. loose FAILs); see NOTES.md §8.3 |
| `candidates/` | session S3 feasibility runs on a different (96-piece, S1) profile; **not** part of this certificate |

In each output JSON, `certificate_sha256` is the SHA-256 of the input file, and `checker_sha256` that of
`rigorous_check.py`.
