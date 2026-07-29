---
name: binderforge-results
description: >
  Interpreting BinderForge outputs: the ranked CSV/Excel columns
  (BinderScore, candidate_tier, ipSAE, ipTM, dG, etc.), the HTML dashboard,
  PyMOL sessions, and the binder-refold / homodimer-refold self-consistency
  checks. Reach for this skill whenever the user says "explain these
  results", "what does ipsae_min mean", "which designs are actually good",
  "why was this design eliminated", or "run the dimer refold check".
compatibility: "a completed (or partially completed) run with outputs/04_Scoring or outputs/05_Dashboard populated"
allowed-tools: Read, Bash, Grep
---

# BinderForge Results Skill

## Where things land (manual.md §7)

```
outputs/04_Scoring/    {project}_summary.csv, {project}_ranked.csv
outputs/05_Dashboard/  {project}_dashboard.html, List_1_AF3_Candidates.xlsx, PyMOL_Ready/
```

## Reading `_ranked.csv` (`scripts/columns.py:COLUMN_ORDER` is the source of truth)

One row per design, best-first. Read in this order:
- `status` (`PASS`/`ELIMINATED`) + `elimination_reason` — check this before
  anything else; an eliminated design's other columns are informational only.
- `BinderScore` (0–1) + `candidate_tier` (`Tier1`/`Tier2`/`Tier3`) — the
  composite ranking.
- `ipsae_min` (higher=better; hard cutoff in `scoring.ipsae_min_hard_cutoff`),
  `iptm`, `model_confidence` (`0.8*ipTM + 0.2*pTM`) — primary interface-quality
  signals.
- `dG`, `sc_score`, `dSASA`, `unsat_hbonds`, `clashes` — Rosetta/biophysical
  interface quality.
- `lrmsd` — ligand/binder-frame RMSD vs. the original backbone design; a
  hard eliminator (`scoring.hard_eliminators.lrmsd`). Not covered by
  `04_Scoring.py`'s own docstring legend (checked — it's missing there), so
  don't expect to find it by reading that docstring alone.
- `rmsd_binder` — how much the predicted structure moved from the original
  backbone design.
- `refold_*` columns — binder-alone AlphaFast refold (self-consistency: does
  the binder fold the same way alone as it does bound?). **Only populated
  when `predictor: alphafast`** — `04_Scoring.py` explicitly skips this phase
  for Boltz runs ("requires alphafast predictor"), so all-NaN `refold_*`
  columns on a Boltz run is expected, not a bug.
- `dimer_iptm` / `dimer_ptm` / `dimer_model_confidence` (from
  `scripts/test_dimer_refold.py`, not yet wired into the main pipeline) —
  homodimer self-association check; a confident homodimer on a sequence
  meant to be a specific binder is a red flag for stickiness, not a good sign.
  Same AlphaFast-only caveat applies (it shells out to the AlphaFast docker
  container).

Most of the column legend lives as a docstring at the top of
`scripts/04_Scoring.py` — read it there for anything not covered above, but
know it's not exhaustive (e.g. it omits `lrmsd`).

## Secondary validation: homodimer refold

```bash
python scripts/test_dimer_refold.py \
    --config <run>/outputs/.resolved_config.yaml \
    --ranked-csv <run>/outputs/04_Scoring/<project>_ranked.csv \
    [--top N]
```
Standalone script, run manually against a completed run's ranked CSV for the
top-N PASS candidates — not part of `binderforge all` yet.

## PyMOL sessions

`outputs/05_Dashboard/PyMOL_Ready/*.pml` — `Refold_*.pml` overlays the bound
complex against the binder-alone refold; open with `pymol <file>.pml`.

## Re-scoring after tuning `cutoffs`/`scoring` in config.yaml

Hand off to **binderforge-run** — steps 4 then 5, no need to regenerate
structures. Note `--no-refold`/`--top N` only work when calling
`scripts/04_Scoring.py` directly (not through `Run_Pipeline.py 4`, which
doesn't forward extra flags) — see binderforge-run for the exact command.
