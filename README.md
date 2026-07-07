# BinderForge

Given a target protein, design candidate **binders** and rank them — a 5-step pipeline
orchestrated by one command:

**0** target MSA → **1** backbone (RFdiffusion *or* Proteina Complexa) → **2** sequence
(ProteinMPNN) → **3** structure prediction (AlphaFast *or* Boltz) → **4** scoring
(BinderScore) → **5** dashboard.

Each heavy step runs in its own conda env or Docker container; you pick the backbone
generator and predictor in a config file. Chain convention everywhere: **A = binder,
B = target**.

## Install

```bash
git clone <repo> && cd BinderForge
pip install -e .                     # installs the `binderforge` command (orchestrator only)

binderforge setup-envs --create     # build the conda envs from envs/*.yml
#   then do the per-env pip / git / license follow-ups documented in envs/README.md
#   and download the model weights / databases for the path you'll use

cp site.yaml.example ~/.binderforge/site.yaml   # then edit every path + env name
```

`site.yaml` holds all machine-specific settings (conda source, env names, tool/model/DB
paths) — set once per machine. A run's `config.yaml` stays portable.

## Run

```bash
cp -r ProjectName MyTarget           # the template scaffold
# replace MyTarget/target.pdb with your target; edit the `# TODO`s in MyTarget/00-Label/config.yaml

cd MyTarget/00-Label
binderforge doctor                  # check envs / tools / model paths for this config
binderforge all                     # run steps 0→5  (or one step: binderforge 4)
```

Results land in `MyTarget/00-Label/outputs/` (ranked Excel, HTML dashboard, PyMOL sessions,
provenance manifest).

## More

See [`manual.md`](manual.md) for the full walkthrough (folder convention, `binder_id`,
scoring cutoffs, troubleshooting) and [`envs/README.md`](envs/README.md) for per-tool env setup.

## Requirements

GPU(s) + conda; the external tools you enable (RFdiffusion3, ProteinMPNN, Boltz/AlphaFast,
PyRosetta, colabfold/mmseqs). Model weights (AF3, RFdiffusion, PyRosetta license, MSA DBs)
are downloaded separately — `binderforge doctor` tells you what's missing.
