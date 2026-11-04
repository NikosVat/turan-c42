# Pull request draft for teorth/optimizationproblems

Placeholders: `<REPO_URL>` = URL of this repository, `<COMMIT>` = the commit hash to cite (pin a commit, as the
other 42a rows do). Files to submit: `submission/42a.md` → `constants/42a.md`, and the two README changes in
`submission/README_row42.md` (a full patched copy is `submission/README.md`). Both are written on top of PR #305
(teorth/optimizationproblems commit `bac1d85`), which is itself on top of PR #302.

Before submitting, the human contributor should read and verify the references and links, as CONTRIBUTING's AI
use policy requires, and re-check the open PRs for a lower C42 claim.

---

**Title:** 42a: upper bound 0.686360751* on Bakeas's Theorem A (33-piece phase profile, unrefereed)

**Body:**

This PR records **C₄₂ ≤ 686360751/1000000000 = 0.686360751\*** for Turán's pure power-sum constant. That is
4.40e-8 below #305 (0.6863607950\*), 9.249e-6 below #302 (0.68637\*), and 9.750553e-4 below the merged
0.6873358063\*. The margins are exact differences of the decimal values.

**Idea.** The construction and the limit theorem are Bakeas's (#302, Theorem A): Harcos's linearization with
s_k = T e^{iφ(k/n)} for k ≤ ⌊n/2⌋, which gives limsup R_n ≤ max(T, F). Pinho (#305) observed that the proof holds
for any finite set of knots, and added the integrated (Gronwall) form of the residual bound. Both are used
unchanged. Two things are new:

- **A 33-piece profile.** It was found by an adjoint-gradient search on the limit functional. It uses the knots
  j²/1152 (Pinho's quadratic family with 24 pieces), six dyadic knots 1/18432, …, 1/576 near 0, and four greedy
  insertions. Its floating-point fixed point T = F is about 0.6863607503, and the certificate takes T = C = 0.686360751.
- **A new a posteriori certifier.**
  - The trial function ρ̃ is a discontinuous piecewise polynomial of degree P on N uniform cells aligned with
    all knots.
  - The residual d + d∗ρ̃ − tρ̃ is then computed *exactly* on every cell, from an identity for convolutions of
    piecewise polynomials (two Kronecker-substituted polynomial products). Its sup per cell is bounded by
    Chebyshev coefficients.
  - Lemma 3, in Pinho's integrated form, turns this into bounds on |p(1) − p̃(1)| and on the denominator.
  - p̃ is evaluated on each cell with the kernel singularity handled in closed form.
  - The result is an enclosure of F of width about 1e-11 at N = 73728 and 7 GB. For comparison, #305's
    certificate used N = 14,155,776 for a one-sided gap of about 3e-8.

**Certificate, note, format specification and checkers:** <REPO_URL>/tree/<COMMIT>

- Certificate `certificate/c42_profile_0.686360751.json`, SHA-256
  `d015a8295d809fa26c9f18888ece53d77c9fea71712420313e75911065855652`. It contains T = C = 0.686360751, 34 exact
  rational knots (least common denominator 18432) and 34 exact decimal phases. The row lists them all.
- The hypotheses of Theorem A are certified: a = Re γ = T cos ψ_0 = 0.1016540… > 0, and κ = sup|d| = T < 1
  (on (0,½], |d| ≤ 2T sin(E/2) = 0.579 with E = max|ψ_j − ψ_0| = 0.871).
- **Checker #1** (`checkers/rigorous_check.py`, python-flint 0.9.0, 256-bit Arb):
  `OPENBLAS_NUM_THREADS=1 python checkers/rigorous_check.py certificate/c42_profile_0.686360751.json --N 73728 --P 10 --refine 1`
  gives PASS with F ∈ [@@C1_73728_LO@@, @@C1_73728_UP@@] (@@C1_73728_TIME@@, @@C1_73728_MEM@@). The
  quick setting `--N 36864 --P 10` gives PASS, F ≤ @@C1_36864_P10_UP@@, in @@C1_36864_P10_TIME@@ and
  @@C1_36864_P10_MEM@@.
- **Checker #2** (`checkers/independent_check.py`, 448-bit Arb) was written in a separate session from the note
  and the certificate-format specification only, without access to the search code or checker #1.
  `python checkers/independent_check.py certificate/c42_profile_0.686360751.json --N 73728 --P 12` gives PASS,
  F ≤ @@C2_73728_UP@@ (@@C2_73728_TIME@@, @@C2_73728_MEM@@), inside checker #1's enclosure.
  - It uses a different rigorous method at every step where there is a choice: exact integer convolutions, a
    closed form for p̃(1) with no truncation, and a linear-minorant lower bound for the denominator.
  - It shares the mathematics of Theorem A and Lemma 3.
- **Corruption tests.** Perturbed phases, a lowered T = C, and claims 1e-9 below the certified enclosure all FAIL.
  Each FAIL is certifiably false, because checker #1's lower bound exceeds C. The 1e-9-above controls PASS.
  Hypothesis violations (C < T, a < 0, a misaligned grid, malformed input) are REFUSED. Checker #2 gives the
  expected verdict on all 28 cases, and checker #1 on all of its cases.
- A floating-point cross-check that uses no limit theorem: the exact finite-n construction on the same profile,
  at n up to 4·10⁶, is monotone and consistent with the certified limit. Its error at these n (~5e-6) is far
  larger than the margin, so this is a consistency check only.

**Scope.** This is an unrefereed analytic asymptotic reduction (a bound on limsup R_n). No explicit finite
threshold, optimality or formal verification is claimed. The asterisk follows CONTRIBUTING's convention for
self-archived results. The margin below #305 is small (4.4e-8) but large compared with the certified enclosure
width (about 1e-11). The float search (session S2) puts the best fixed point of the phase-only family at about
0.68636074, so further gains inside this family are confined to about 1e-8.

**AI use.** Found, proved and checked by Claude agents (Claude Code, Anthropic), directed by Nikolaos Vatopoulos,
who takes responsibility for the submission. The second checker and a referee report on the note were written by
a separate Claude agent, working blind from the note and the certificate format only. The two checkers share no
code or data, but both are AI-written, so their independence is procedural. No human mathematician has checked
the argument.

**Overlap.** This PR edits the same table, reference list and README cell as #302 and #305, both open as of
4 Nov 2026. It is written on top of #305, because the new row cites [Bak26] and [Pin26b]. Please merge them in
chronological order: the new row goes after the [Pin26b] row. #310 changes only the lower-bound table and does
not conflict.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
