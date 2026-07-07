# 🧬 Binder Design Pipeline — User Manual

Basic instructions for running the protein-binder design pipeline: backbone
generation → sequence design → structure prediction → scoring → dashboard.

---

## 1. What it does

Given a **target** protein, the pipeline designs candidate **binders** and ranks
them. It runs as five steps, orchestrated by `Run_Pipeline.py`:

| Step | Name | What it does | Env |
|------|------|--------------|-----|
| 1 | Backbone | Generate binder backbones (RFdiffusion *or* Proteina Complexa) | `rf3` |
| 2 | ProteinMPNN | Design sequences for each backbone (skipped for Complexa) | `mlfold` |
| 3 | Prediction | Predict binder–target complex (Boltz *or* AlphaFast) | `boltz` / `base` |
| 4 | Scoring | RMSD, ipSAE, physics, **BinderScore**, assigns `binder_id` | `pyrosetta` |
| 5 | Dashboard | Interactive HTML + ranked Excel + PyMOL sessions | `base` |

You choose the **backbone generator** and **predictor** in the config; the
orchestrator runs the matching scripts automatically.

---

## 2. Install & prerequisites

```bash
git clone <repo> && cd BinderForge
pip install -e .              # installs the `binderforge` command (orchestrator only)
binderforge setup --create    # creates the conda envs + writes a starter site.yaml
```
(If you skip `pip install -e .`, replace `binderforge` with `python /path/to/Run_Pipeline.py`.)

`setup` can't do three things for you — all documented in [`envs/README.md`](envs/README.md):
1. **Per-env follow-ups**: install torch matching your CUDA, `git clone` RFdiffusion +
   ProteinMPNN, accept the PyRosetta license.
2. **Download the gated weights / DBs** you'll use (AF3, RFdiffusion checkpoint, AF2 params,
   MSA DB) — licensed/large, so you fetch them.
3. **Point `site.yaml` at them** — but paths default under `~/.binderforge/{tools,models}/`
   and env names default to the recipe names, so `site.yaml` is **mostly optional**: you edit
   only the entries whose install lives elsewhere. `init-site` can also seed it from a filled
   config. Value precedence: built-in defaults < `site.yaml` < a run's `config.yaml`.

**Target inputs**: a target PDB and its MSA (`.a3m`, auto-made in step 0 if absent).

Then verify what the steps you enabled actually need:
```bash
cd <Project>/<NN-Label>
binderforge doctor            # ✅/❌ per env, tool, and model path — with the site.yaml key to fix each
```

### Generate the target MSA (once, before running)
```bash
conda activate msa_tools
colabfold_search your_target.fasta "$MSA_DB" msa_output   # MSA_DB = site.yaml tools.msa_db
# put the resulting .a3m path into config target.msa
```

---

## 3. Folder layout & identity (important)

Three levels — the **folder structure IS the convention**:

```
<TOOL ROOT>/                 scripts/ + Run_Pipeline.py  (shared, one copy)
└── <ProjectName>/           one target/binder
    ├── target.pdb           shared by all configs
    ├── target_msa.a3m       created on 1st run (step 0) if missing
    └── <NN-Label>/          one config/run
        ├── config.yaml      PARAMETERS ONLY
        └── outputs/         this run's outputs + manifest + summary
```

Every design gets a traceable **`binder_id = {binder_name}-{config_id:02}-{n}`**
(e.g. `GPIP-00-42`), where **`n`** is sequential (gpu0 first, then gpu1). These are
**derived from the folders**, so you don't set them:

- **`binder_name`** ← the **Project folder** name.
- **`config_id`** ← the **`NN-`** prefix of the config folder (auto-added on first
  run if you named the folder without one — `RFdiffusion1` → `00-RFdiffusion1`).
- **target pdb/msa** ← the Project folder (`target.pdb`, `target_msa.a3m`).

`config.yaml` only holds design parameters; anything above can still be overridden
there. `binder_id` is the last column of every results file.

---

## 4. Quick start

```bash
# 1. Make a project with its target, and a config folder inside it:
mkdir -p /data/Process/MyProject/RFdiffusion1
cp target.pdb /data/Process/MyProject/target.pdb        # (MSA auto-made in step 0)
cp <tool>/config.yaml /data/Process/MyProject/RFdiffusion1/config.yaml   # edit params

# 2. cd into the config folder and run the shared orchestrator:
cd /data/Process/MyProject/RFdiffusion1
python /path/to/tool/Run_Pipeline.py all
```
The folder auto-renames to `00-RFdiffusion1`, outputs land in `./outputs/`, and
results carry `binder_id = MyProject-00-<n>`.

Steps: `0`=MSA, `1`=Backbone, `2`=MPNN, `3`=Prediction, `4`=Score, `5`=Dash,
`all`=0→5. Run a single step the same way, e.g. `python …/Run_Pipeline.py 4`.

> Tip: after the first-run rename, your shell prompt may still show the old folder
> name — `cd .` (or into the new `00-…` path) to refresh it.
> Power users can bypass the cwd convention with `--config <path>`.

---

## 5. Validate a config before running
```bash
python <tool>/scripts/config_schema.py <config folder>/outputs/.resolved_config.yaml
```
The orchestrator resolves the folder convention into `outputs/.resolved_config.yaml`
and validates **that** automatically in pre-flight, refusing to start on a bad config.
(Validating a bare params-only `config.yaml` will just warn that identity/target/
outputs are "derived at run time".)

---

## 6. Key config fields

```yaml
# Identity
binder_name:  "GPIP"          # binder_id prefix (tied to the target)
config_id:    0               # REQUIRED run number 0–99
config_label: "RFdiffusion1"  # cosmetic, folder-name only
work_dir:     "/data/.../00-RFdiffusion1"

# Switches
backbone_generator: "rfdiffusion"   # or "complexa"
predictor:          "alphafast"     # or "boltz"

# Target
target:
  pdb: "/data/.../target.pdb"
  msa: "/data/.../target.a3m"

# Step 1a — RFdiffusion (used when backbone_generator = rfdiffusion)
step2_rfd:
  binder_length: [50, 150]
  target_residues: "A1-346"
  hotspot_residues: ["A271", "A244"]   # [] = full surface
  batch_size: 2
  num_batches: 100                     # total = batch_size × num_batches × n_GPUs

# Step 2 — ProteinMPNN
step3_mpnn:
  seqs_per_target: 1000
  seqs_to_validate: 1        # top N sequences forwarded to prediction
  sampling_temp: 0.1
  batch_size: 24             # auto-snapped to a divisor of seqs_per_target

gpu_devices: [0, 1]
```
Predictor/scoring/dashboard blocks (`step4_boltz`, `step4_alphafast`, `cutoffs`,
`scoring`) have sensible defaults in the template; tune only if needed.

---

## 7. Outputs

Under `work_dir/outputs/`:

```
01_BackboneGeneration/   backbones (per gpu)
02_ProteinMPNN/          designed sequences + prediction inputs
03_Boltz/ | 03_AlphaFast/ predicted complexes
04_Scoring/              {project_name}_summary.csv, {project_name}_ranked.csv
05_Dashboard/            {project}_dashboard.html, List_1_AF3_Candidates.xlsx,
                         PyMOL_Ready/
run_manifest.json        provenance: date, git commit, versions, config hash
```

### Reading the results
- **`_ranked.csv` / Excel** — one row per design, sorted best-first. Key columns:
  - `binder_id` — the traceable identity (last column).
  - `BinderScore` (0–1) + `candidate_tier` (Tier1/2/3).
  - `status` = PASS / ELIMINATED (+ `elimination_reason`).
  - `ipsae_min`, `iptm`, `model_confidence`, `pDockQ`, `dG`, `sc_score`,
    `dSASA`, `lrmsd`, `rmsd_binder`.
- **Dashboard HTML** — screening plots + score-correlation heatmap.
- **PyMOL_Ready/** — `.pml` sessions to open a design's backbone vs prediction.

---

## 8. Common recipes

**Re-score only (after tuning cutoffs):**
```bash
python Run_Pipeline.py 4 --config <run>/config.yaml
python Run_Pipeline.py 5 --config <run>/config.yaml
```

**Switch predictor:** set `predictor: "boltz"` or `"alphafast"`, re-run step 3+.

**Switch backbone method:** set `backbone_generator: "complexa"` (step 2/MPNN is
skipped — Complexa designs its own sequences).

---

## 9. Traceability

Each run writes `run_manifest.json` into `work_dir` at start: timestamp, git
commit of the pipeline code, package versions, host, and a hash of the exact
config. To activate git capture, the pipeline folder must be a git repo:
```bash
cd Pipeline_model && git init && git add -A && git commit -m "init"
```

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Pre-flight fails with `config_id: REQUIRED` | Set `config_id` (0–99) in config. |
| `ipsae_min` is NaN | AF3 `*_full_data.json` missing/pruned — needed for ipSAE. |
| Step reads wrong folder on old data | Point `outputs:` keys at the existing folder names. |
| Nothing to score | Prediction outputs missing — check step 3 completed. |
| Scoring very slow | Normal (Rosetta per design); uses CPU cores − 2. |

---

*Scripts live in `scripts/`. The orchestrator is `Run_Pipeline.py`; scaffolding is
`new_run.py`; validation is `scripts/config_schema.py`. The step-I/O contract and the
canonical design-name scheme are documented in `scripts/contract.py`.*
