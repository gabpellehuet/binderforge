# BinderForge Skills

Five project-local Claude Code skills covering setup, target configuration,
running, results interpretation, and multi-run sweeps. Each skill picks the
cheapest tool for the job — the `binderforge` CLI where it already does the
work (env creation, pre-flight validation, provenance), and direct file edits
/ `AskUserQuestion` where the CLI would just be a thin wrapper or the
decision is inherently user-specific (biology, machine paths).

## The pipeline (README.md / manual.md)

BinderForge runs one pipeline, five steps, each in its own conda env or
Docker container, orchestrated by `Run_Pipeline.py` / the `binderforge` CLI:

| Step | Name | Env | Config switch |
|---|---|---|---|
| 0 | Target MSA | `msa_tools` | — |
| 1 | Backbone | `rf3` or Complexa docker | `backbone_generator: rfdiffusion \| complexa` |
| 2 | ProteinMPNN | `mlfold` | skipped for `complexa` |
| 3 | Prediction | `boltz` or AlphaFast docker | `predictor: boltz \| alphafast` |
| 4 | Scoring | `pyrosetta` | BinderScore, hard eliminators |
| 5 | Dashboard | `base`/`dash` | HTML dashboard + ranked Excel + PyMOL sessions |

Chain convention everywhere: **A = binder, B = target**. Machine paths
(conda source, env names, tool/model paths) live in `site.yaml`; a run's
`config.yaml` stays portable and carries none of it (`scripts/sitecfg.py`).

**If the user doesn't specify, the template defaults are
`backbone_generator: rfdiffusion` and `predictor: alphafast`** (`config.yaml`)
— only switch on an explicit ask.

## Skills

| Skill | Primary tool | Alternative paths | When to use it |
|---|---|---|---|
| [`binderforge-setup`](./binderforge-setup/) | **CLI** (`binderforge setup --create`) + **file-edit** (`site.yaml`) | `binderforge setup-envs` for just the conda envs; manual `conda env create -f envs/X.yml` for one env | Fresh checkout, verifying an install, filling `site.yaml` |
| [`binderforge-target`](./binderforge-target/) | **Folder scaffold** (`cp -r ProjectName`) + **file-edit** (`config.yaml` TODOs) | none — the folder convention *is* the scaffolder, there's no CLI generator | Registering a new target + config |
| [`binderforge-run`](./binderforge-run/) | **CLI** (`binderforge doctor` / `all` / `<step>`) | `python Run_Pipeline.py <step> --config <path>` for power users bypassing the cwd convention | Running steps, monitoring a long job, first-line troubleshooting |
| [`binderforge-results`](./binderforge-results/) | **File read** (ranked CSV/Excel/dashboard) | `scripts/test_dimer_refold.py` for the homodimer secondary-validation check | Interpreting outputs, re-scoring after tuning cutoffs |
| [`binderforge-sweep`](./binderforge-sweep/) | **Batched CLI loop** + a small ad hoc aggregation script | none built in yet — this skill documents the pattern | Comparing generators/predictors, deciding when to stop a big campaign |

## What's different from a from-scratch skill set

BinderForge already ships most of the "shared infrastructure" other projects
need to build by hand: `binderforge doctor` is the pre-flight check,
`binderforge setup`/`setup-envs` is the environment builder,
`scripts/config_schema.py` is the config validator, and `run_manifest.json`
(`scripts/provenance.py`) is the replayable-run artifact. So there's no
`_shared/scripts/preflight.sh` here — the skills below just point at the CLI
verb that already exists, and are anchored to real files (`manual.md`,
`config.yaml`, `site.yaml.example`, `scripts/columns.py`,
`scripts/04_Scoring.py`) rather than restating them.

## Adding a new skill

Anchor authoring to specific source files, keep `SKILL.md` short and
actionable, and prefer pointing at `manual.md`/`envs/README.md` over
duplicating their tables — they're the source of truth and will drift out of
sync with a skill's copy of them otherwise.
