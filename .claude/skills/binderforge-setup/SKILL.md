---
name: binderforge-setup
description: >
  First-time install and machine setup for BinderForge: creating the conda
  envs, writing/filling site.yaml (machine-specific paths), the per-tool
  follow-ups setup can't automate (RFdiffusion3, ProteinMPNN, PyRosetta
  license), and pointing at gated model weights (AF3, RFdiffusion checkpoint,
  AF2 params, MSA database). Reach for this skill whenever the user says
  "install binderforge", "set up binderforge", "binderforge setup", "create
  the conda envs", "what's missing in my site.yaml", "binderforge doctor is
  failing", "configure site.yaml", "where do I put the AF3 weights", or
  "download model weights" — or any time a fresh checkout needs to become
  runnable.
compatibility: "conda; git; pip install -e . done (or being done in this skill); GPU(s) recommended"
allowed-tools: Bash, Read, Edit, Write, AskUserQuestion
---

# BinderForge Setup Skill

Get a fresh checkout to the point where `binderforge doctor` is clean for
whichever backbone generator + predictor the user wants. Three layers, in
order: the orchestrator, the conda envs, then the machine-specific
`site.yaml` that every step reads from.

## Step 1: Install the orchestrator

```bash
pip install -e .          # installs the `binderforge` CLI (README.md, pyproject.toml)
```
If `binderforge` isn't on PATH afterward, fall back to
`python Run_Pipeline.py <step>` for every command below (manual.md §2).

## Step 2: Create conda envs + starter site.yaml

```bash
binderforge setup            # dry-run: prints the plan
binderforge setup --create   # actually creates the missing envs (envs/*.yml) + writes a starter site.yaml
```
`binderforge setup-envs` (same flags) does only the conda-env half, if the
user already has a `site.yaml` they don't want touched.

## Step 3: Per-env follow-ups (setup can't do these — envs/README.md is the source of truth)

| Env | Role | Manual step |
|---|---|---|
| `rf3` | RFdiffusion3 (`backbone_generator: rfdiffusion`) | Install torch matching the user's CUDA; `git clone`/install RFdiffusion3 so `rfd3` is on PATH; set `checkpoints.rfd` |
| `mlfold` | ProteinMPNN | Install torch; `git clone` ProteinMPNN; set `tools.mpnn_script` |
| `boltz` | Boltz (`predictor: boltz`) | Install torch; `boltz` weights auto-download on first run |
| `pyrosetta` | Scoring (step 4) | Accept the PyRosetta academic license, then run the installer the recipe pulls in (`pyrosetta-installer`) |
| `msa_tools` | MSA (step 0) | Download an mmseqs/colabfold database; set `tools.msa_db` |
| `dash` | Dashboard | Optional — defaults to conda `base` |

Docker-based steps (Complexa backbone, AlphaFast predictor) don't need a
*dedicated* env for the heavy compute — that's containerized — but the
Python launcher script itself (`01_ProteinaComplexa.py` / `03_AlphaFast.py`)
still runs inside the `dash` role's conda env (`Run_Pipeline.py`:
`ENV_DASH = envs.get('dash', 'base')`). Set the container image/paths in
`site.yaml` (Step 4); the conda env for the launcher is whatever `envs.dash`
resolves to.

Never assume a CUDA version or fabricate a download URL for gated weights —
ask the user what they have installed and where.

**Gotcha:** `envs/dash.yml` creates a conda env literally named `dash`, but
the built-in default (`scripts/sitecfg.py:default_machine`) maps the `dash`
role to conda env **`base`**, not `dash`. If the user runs
`binderforge setup --create` and never adds `envs.dash: dash` to
`site.yaml`, the new `dash` env sits unused — and `binderforge doctor` won't
catch this, because it only checks that *some* env matching the resolved
name exists, and `base` always does. Worth pointing out explicitly rather
than relying on doctor to flag it.

## Step 4: Fill site.yaml (ask, don't guess)

`site.yaml` resolves in this order: `$BINDERFORGE_SITE` → `<tool root>/site.yaml`
→ `~/.binderforge/site.yaml` (`scripts/sitecfg.py`). Everything has a
built-in default under `~/.binderforge/{tools,models}/`
(`scripts/sitecfg.py:default_machine`) — only ask about paths that live
elsewhere.

Use `AskUserQuestion` to fill whichever of these the user's install doesn't
match the default layout — only the keys for the generator/predictor
combination actually in use, `site.yaml.example` is a superset template, not
something to fill in wholesale:
- `tools.mpnn_script`, `tools.msa_db`, `tools.pymol` (optional), `tools.af_input_dir`
- `checkpoints.rfd` (only if `backbone_generator: rfdiffusion`)
- `step1_complexa.{complexa_dir,docker_image,af2_params_dir}` (only if `backbone_generator: complexa`)
- `step3_alphafast.{alphafast_dir,db_dir,weights_dir}` (only if `predictor: alphafast`)
  — `weights_dir` is the access-gated AF3 weights; the user downloads these
  themselves, this skill only records where they put them.

## Step 5: Verify

```bash
cd <Project>/<NN-Label>       # from inside a config folder
binderforge doctor
```
Read the ✅/❌ output — each ❌ line names the exact `site.yaml` key or env
still missing. Loop back to Step 3/4 for whatever's flagged; don't work
around a ❌ by editing the run's `config.yaml` instead — machine paths
belong in `site.yaml`, never in the portable run config.

## Hand off

Once `binderforge doctor` is clean, hand off to **binderforge-target** (new
project) or **binderforge-run** (an existing, already-configured project).
