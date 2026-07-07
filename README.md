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
pip install -e .              # installs the `binderforge` command (orchestrator only)
binderforge setup --create    # creates the conda envs + writes a starter site.yaml
```

`setup` gets you most of the way. Three things it can't do for you:
1. **torch** matching your CUDA, and `git clone` RFdiffusion + ProteinMPNN, and the PyRosetta
   license — the per-env follow-ups in [`envs/README.md`](envs/README.md).
2. **Download the gated model weights / DBs** (AF3, RFdiffusion, AF2 params, MSA DB) for the
   path you'll use — these are licensed/large, so no tool can fetch them for you.
3. Point `site.yaml` at them — **but** if you install tools/weights under the default layout
   `~/.binderforge/{tools,models}/`, the paths are already filled in; you only edit the ones
   that live elsewhere. `binderforge doctor` (run from a config folder) lists exactly what's
   still missing for your chosen generator + predictor.

Machine settings live in `site.yaml` (conda source, env names, tool/model paths); a run's
`config.yaml` stays portable and carries none of it.

## Run

```bash
cp -r ProjectName MyTarget           # the template scaffold
# put your target .pdb in MyTarget/ (any filename, auto-detected); edit the `# TODO`s in MyTarget/00-Label/config.yaml

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
