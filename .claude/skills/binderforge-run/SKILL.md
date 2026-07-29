---
name: binderforge-run
description: >
  Running the BinderForge pipeline: `binderforge doctor` / `all` / a single
  step, understanding which conda env each step needs, monitoring a
  long-running GPU job, and first-line troubleshooting when a step fails.
  Reach for this skill whenever the user says "run the pipeline", "run
  binderforge", "run step 4", "re-score only", "why did step 3 fail",
  "binderforge is stuck", or "check on my run".
compatibility: "a config folder with config.yaml TODOs filled and binderforge doctor passing"
allowed-tools: Bash, Read, Grep
---

# BinderForge Run Skill

## Step 0: Always doctor first

```bash
cd <Project>/<NN-Label>
binderforge doctor
```
Don't launch a run against a ❌ — hand off to **binderforge-setup** first.

## Step map (manual.md §1 / §4)

| Step | Name | Env | What it does |
|---|---|---|---|
| 0 | MSA | `msa_tools` | Target MSA (skipped if `<target-pdb-stem>_msa.a3m` already exists next to the target PDB) |
| 1 | Backbone | `rf3`, or `dash` launching the Complexa container | RFdiffusion or Proteina Complexa |
| 2 | ProteinMPNN | `mlfold` | Sequence design (skipped entirely for Complexa) |
| 3 | Prediction | `boltz`, or `dash` launching the AlphaFast container | Structure prediction of the binder–target complex |
| 4 | Scoring | `pyrosetta` | RMSD, ipSAE, physics, BinderScore, `binder_id` assignment |
| 5 | Dashboard | `dash` | HTML dashboard + ranked Excel + PyMOL sessions |

Note the docker-based steps (Complexa, AlphaFast) still run their Python
launcher inside the `dash` role's conda env (`Run_Pipeline.py:
ENV_DASH = envs.get('dash', 'base')`) — only the model compute itself is
containerized. And the MSA filename is paired with whatever the actual
target PDB is named (`{stem}_msa.a3m`), not literally `target_msa.a3m`
unless the target file happens to be named `target.pdb`
(`Run_Pipeline.py:143-144`).

## Running

```bash
binderforge all          # steps 0→5
binderforge 4            # a single step, e.g. re-scoring only
```
Power users can bypass the cwd convention with `--config <path>` (manual.md tip).

## Monitoring a long run

Steps 1/3/4 are the expensive ones (GPU sampling, GPU inference, CPU-bound
Rosetta respectively). For a run expected to take hours:
- Launch it as a background command rather than blocking on it.
- Progress signal per step: step 1 batches land under
  `outputs/01_BackboneGeneration/`; step 3 status is tracked by counting
  `*_model.cif` files (`scripts/03_AlphaFast.py`'s own progress monitor does
  this); step 4 is intentionally slow (manual.md §10: "Normal (Rosetta per
  design); uses CPU cores − 2") — don't mistake it for a hang.

## First-line troubleshooting (manual.md §10)

| Symptom | Likely cause |
|---|---|
| `config_id: REQUIRED` (only when validating a raw config directly with `config_schema.py`) | Normal `binderforge`/`Run_Pipeline.py` invocations auto-backfill `config_id` from the folder's `NN-` prefix (defaulting to 0) before validation ever runs — this only surfaces when someone bypasses that and validates a bare config file by hand |
| `ipsae_min` is NaN | AF3 `*_full_data.json` missing/pruned |
| Step reads the wrong folder on old data | Point `outputs:` keys at the existing folder names |
| Nothing to score | Prediction outputs missing — check step 3 actually completed |
| Scoring is very slow | Normal — not a bug |

For anything not in this table, check `outputs/04_Scoring/scoring_errors.log`
and `run_manifest.json` (git commit, config hash, versions) before guessing.

## Re-scoring only (after tuning cutoffs)

```bash
python Run_Pipeline.py 4 --config <run>/config.yaml
python Run_Pipeline.py 5 --config <run>/config.yaml
```
No need to rerun steps 0-3 just to re-apply new `scoring:` cutoffs.

`scripts/04_Scoring.py` also has `--no-refold` and `--top N` flags
(skip/resize the binder-refold phase), but `Run_Pipeline.py`'s step-4
dispatch only ever passes `--config <resolved_path>` through — it doesn't
forward extra args. To use `--no-refold`/`--top`, call
`python scripts/04_Scoring.py --config <run>/outputs/.resolved_config.yaml --no-refold`
directly instead of going through `Run_Pipeline.py 4`.

## Hand off

Once a run has produced `outputs/04_Scoring`/`outputs/05_Dashboard`, hand off
to **binderforge-results** to interpret them.
