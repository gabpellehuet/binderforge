# Conda env recipes

One `environment.yml` per pipeline role. They're **scaffolds**: conda captures the base
(python/CUDA/light deps), but the actual research tools are pip/git installs that conda
history can't record — do those follow-up steps below. GPU stacks are CUDA-specific, so
**install torch to match your driver** (this machine used cu128 nightlies for an RTX 50xx),
exactly as most model repos instruct.

Create the envs (skips any that already exist):
```bash
python /path/to/Run_Pipeline.py setup-envs        # prints a plan + the create commands
python /path/to/Run_Pipeline.py setup-envs --create   # actually create the missing ones
```
Then point `site.yaml` at your installs and verify with `Run_Pipeline.py doctor`.

## Per-env follow-ups (after `conda env create`)

| Env | Role | Manual step after env create |
|-----|------|------------------------------|
| `rf3` | RFdiffusion3 (step 1) | Install torch (your CUDA), then clone/install RFdiffusion3 so the `rfd3` CLI is on PATH; set `checkpoints.rfd` in site.yaml. |
| `mlfold` | ProteinMPNN (step 2) | Install torch (your CUDA); `git clone` ProteinMPNN; set `tools.mpnn_script` to its `protein_mpnn_run.py`. |
| `boltz` | Boltz (step 3) | Install torch (your CUDA). `boltz` comes from the recipe; weights auto-download on first run. |
| `pyrosetta` | Scoring (step 4) | Accept the free academic PyRosetta license; the recipe installs `pyrosetta-installer`, then run its installer to fetch the wheel. |
| `msa_tools` | MSA (step 0) | Download the colabfold/mmseqs database; set `tools.msa_db`. |
| `dash` | Dashboard (step 5) | Optional — `envs.dash` defaults to conda `base`. If you create this env, also set `envs.dash: dash` in `site.yaml`, or it sits unused: `doctor` only checks that *some* env matching the resolved name exists, and `base` always does, so it won't warn you about the mismatch. |

Docker-based steps (Complexa backbone, AlphaFast prediction) don't need a
**dedicated** conda env for the model compute itself — that part runs in a
container. But the Python launcher script for those steps still runs inside
the `dash` role's conda env (`envs.dash`, default `base`), same as step 5 —
so "no conda env" only means "no new env to create," not "no env activated."
Set the container image/paths in `site.yaml`.
