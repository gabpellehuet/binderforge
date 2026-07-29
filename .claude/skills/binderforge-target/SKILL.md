---
name: binderforge-target
description: >
  Scaffolding a new target/project and its run config in BinderForge:
  copying the ProjectName template, placing the target PDB, filling the
  `# TODO` fields in config.yaml (backbone_generator, predictor,
  binder_length, target_residues, hotspot_residues), and understanding the
  folder-convention identity (binder_name / config_id / binder_id). Reach for
  this skill whenever the user says "set up a new target", "start a new
  project", "create a config for <protein>", "what hotspot residues should I
  use", "fill in the config TODOs", or "validate my config".
compatibility: "a target .pdb file; BinderForge cloned (envs/site.yaml not required for this skill)"
allowed-tools: Bash, Read, Edit, Write, AskUserQuestion
---

# BinderForge Target Skill

## Step 1: Understand the folder convention first (manual.md §3)

BinderForge's identity is derived from folder names, not config fields:
```
<ProjectName>/            one target — target.pdb lives here, shared by every config
└── <NN-Label>/           one config/run — config.yaml + outputs/ live here
```
- `binder_name` ← the Project folder name.
- `config_id` ← the `NN-` prefix of the config folder (auto-added on first
  run if missing — `RFdiffusion1` → `00-RFdiffusion1`).
- target pdb/msa ← files directly inside the Project folder (any filename
  for the PDB, auto-detected; MSA is auto-made in step 0 if absent).

Don't put target/identity fields into `config.yaml` unless deliberately
overriding — they're derived from where the folder sits
(`ProjectName/README.md` has a live example).

## Step 2: Scaffold

```bash
cp -r ProjectName MyTarget          # template scaffold
# replace MyTarget/target.pdb with the real target structure (any filename, auto-detected)
```
Rename `MyTarget/00-Label` to something meaningful, or leave it — the `00-`
prefix is what matters, not the label text after it.

## Step 3: Fill the `# TODO`s in config.yaml

Every `# TODO` line in `config.yaml` needs a decision; everything else has a
sensible default (the file documents this itself). Ask the user, don't guess
biology — use `AskUserQuestion` for anything that isn't purely mechanical:

- `backbone_generator`: `rfdiffusion` (template default) or `complexa`
- `predictor`: `alphafast` (template default) or `boltz`
- `step1_rfd.binder_length` / `step1_complexa.binder_length`: `[min, max]` residues
- `step1_rfd.target_residues` / `step1_complexa.target_input`: target chain
  range, e.g. `"A1-346"` — read this off the target PDB's actual chain IDs,
  don't invent one
- `hotspot_residues`: specific residues if the user knows the epitope/interface,
  else `[]` for whole-surface sampling

Everything under `step2_mpnn` / `step3_boltz` / `step3_alphafast` / `cutoffs`
/ `scoring` already has defaults in the template — only touch those if the
user explicitly asks to tune them.

## Step 4: Validate before running

The orchestrator resolves the folder convention into
`outputs/.resolved_config.yaml` (which merges in `site.yaml`'s machine keys)
and validates *that* automatically in pre-flight (manual.md §5) — running
this by hand is only for debugging:
```bash
python scripts/config_schema.py <config folder>/outputs/.resolved_config.yaml
```
**Always point it at the resolved config, not the bare `config.yaml`.**
Identity/target/outputs being unset in a bare params-only `config.yaml`
only produces warnings (`config_schema.py`: `target.pdb`/`outputs` are
"derived at run time") — but `envs`, `gpu_devices`, `tools.mpnn_script`,
`checkpoints.rfd` (if `backbone_generator: rfdiffusion`), and
`step3_alphafast` (if `predictor: alphafast`) are **fatal `err()`s** when
missing (`config_schema.py:107-146`). Those keys live in `site.yaml`, not
`config.yaml` — validating the bare run config directly, before that merge,
will always fail on them even when the real setup is fine.

## Hand off

Once the TODOs are filled, hand off to **binderforge-setup** if
`binderforge doctor` hasn't been run yet, or **binderforge-run** to launch
the pipeline.
