"""
==============================================================================
   🧩 CONFIG SCHEMA VALIDATION
==============================================================================
   Dependency-free validation of config.yaml (no pydantic/jsonschema needed,
   so it runs in every conda env). Catches bad configs BEFORE GPU-hours burn.

   Usage:
     Standalone:  python scripts/config_schema.py [config.yaml]
     Imported:    from config_schema import validate_config
                  ok, errors, warnings = validate_config(cfg)

   errors  -> fatal: the run should not start.
   warnings -> non-fatal: surfaced but the run may proceed.
==============================================================================
"""
import os
import sys
import yaml

GENERATORS = {"rfdiffusion", "complexa"}
PREDICTORS = {"boltz", "alphafast"}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _get(cfg, dotted, default=None):
    """Fetch a nested key by dotted path, e.g. 'target.pdb'."""
    cur = cfg
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def validate_config(cfg, check_paths=True):
    """Validate a loaded config dict. Returns (ok, errors, warnings)."""
    errors, warnings = [], []

    def err(m):  errors.append(m)
    def warn(m): warnings.append(m)

    if not isinstance(cfg, dict):
        return False, ["Config root is not a mapping/dict."], []

    # ── Identity ──────────────────────────────────────────────────────────────
    binder = _get(cfg, 'binder_name')
    project = _get(cfg, 'project_name')
    has_binder  = isinstance(binder, str) and binder.strip()
    has_project = isinstance(project, str) and project.strip()
    if not has_binder and not has_project:
        err("binder_name: required (identity prefix for binder_id); set binder_name, or at least project_name.")
    elif not has_binder and has_project:
        warn("binder_name: not set — binder_id will fall back to project_name.")

    config_id = _get(cfg, 'config_id', _get(cfg, 'run_id'))
    if config_id is None:
        err("config_id: REQUIRED config/run number (integer 0–99) — it goes into every binder_id.")
    elif not _is_int(config_id) or not (0 <= config_id <= 99):
        err(f"config_id: must be an integer 0–99 (got {config_id!r}).")

    label = _get(cfg, 'config_label')
    if label is not None and not isinstance(label, str):
        err(f"config_label: if set, must be a string (got {type(label).__name__}).")

    work_dir = _get(cfg, 'work_dir')
    if work_dir is None or (isinstance(work_dir, str) and not work_dir.strip()):
        warn("work_dir: not set — derived from the config folder at run time.")
    elif not isinstance(work_dir, str):
        err("work_dir: must be a path string.")
    elif check_paths and not os.path.isdir(work_dir):
        err(f"work_dir: directory does not exist: {work_dir}")

    # ── Generator / predictor switches ────────────────────────────────────────
    gen = _get(cfg, 'backbone_generator', 'rfdiffusion')
    if gen not in GENERATORS:
        err(f"backbone_generator: must be one of {sorted(GENERATORS)} (got {gen!r}).")

    pred = _get(cfg, 'predictor', 'boltz')
    if pred not in PREDICTORS:
        err(f"predictor: must be one of {sorted(PREDICTORS)} (got {pred!r}).")

    # ── Target ────────────────────────────────────────────────────────────────
    tgt_pdb = _get(cfg, 'target.pdb')
    if tgt_pdb is None or (isinstance(tgt_pdb, str) and not tgt_pdb.strip()):
        warn("target.pdb: not set — derived from the Project folder (target.pdb) at run time.")
    elif not isinstance(tgt_pdb, str):
        err("target.pdb: must be a path string.")
    elif check_paths and not os.path.isfile(tgt_pdb):
        err(f"target.pdb: file not found: {tgt_pdb}")

    tgt_msa = _get(cfg, 'target.msa')
    if not isinstance(tgt_msa, str) or not tgt_msa.strip():
        warn("target.msa: not set — Boltz accuracy benefits from a target MSA.")
    elif check_paths and not os.path.isfile(tgt_msa):
        warn(f"target.msa: file not found: {tgt_msa}")

    # ── ProteinMPNN block ─────────────────────────────────────────────────────
    spt = _get(cfg, 'step2_mpnn.seqs_per_target')
    if not _is_int(spt) or spt <= 0:
        err(f"step2_mpnn.seqs_per_target: must be a positive integer (got {spt!r}).")
    bs = _get(cfg, 'step2_mpnn.batch_size')
    if bs is not None and (not _is_int(bs) or bs <= 0):
        err(f"step2_mpnn.batch_size: must be a positive integer (got {bs!r}).")
    temp = _get(cfg, 'step2_mpnn.sampling_temp')
    if not _is_num(temp) or temp <= 0:
        err(f"step2_mpnn.sampling_temp: must be a positive number (got {temp!r}).")
    s2v = _get(cfg, 'step2_mpnn.seqs_to_validate')
    if s2v is not None and (not _is_int(s2v) or s2v <= 0):
        err(f"step2_mpnn.seqs_to_validate: must be a positive integer (got {s2v!r}).")

    # ── GPU devices ───────────────────────────────────────────────────────────
    gpus = _get(cfg, 'gpu_devices')
    if not isinstance(gpus, list) or not gpus or not all(_is_int(g) for g in gpus):
        err(f"gpu_devices: must be a non-empty list of integers (got {gpus!r}).")

    # ── Tools ─────────────────────────────────────────────────────────────────
    mpnn = _get(cfg, 'tools.mpnn_script')
    if not isinstance(mpnn, str) or not mpnn.strip():
        err("tools.mpnn_script: required, must be a path string.")
    elif check_paths and not os.path.isfile(os.path.expanduser(mpnn)):
        warn(f"tools.mpnn_script: file not found: {mpnn}")

    # ── Generator-specific requirements ───────────────────────────────────────
    if gen == 'rfdiffusion':
        ckpt = _get(cfg, 'checkpoints.rfd')
        if not isinstance(ckpt, str) or not ckpt.strip():
            err("checkpoints.rfd: required when backbone_generator=rfdiffusion.")
        elif check_paths and not os.path.isfile(ckpt):
            err(f"checkpoints.rfd: file not found: {ckpt}")
        if not isinstance(_get(cfg, 'step1_rfd'), dict):
            err("step1_rfd: required block when backbone_generator=rfdiffusion.")
    elif gen == 'complexa':
        if not isinstance(_get(cfg, 'step1_complexa'), dict):
            err("step1_complexa: required block when backbone_generator=complexa.")

    # ── Predictor-specific requirements ───────────────────────────────────────
    if pred == 'alphafast' and not isinstance(_get(cfg, 'step3_alphafast'), dict):
        err("step3_alphafast: required block when predictor=alphafast.")
    if pred == 'boltz' and not isinstance(_get(cfg, 'step3_boltz'), dict):
        warn("step3_boltz: block missing — defaults will be used.")

    # ── Output / env blocks ───────────────────────────────────────────────────
    outputs = _get(cfg, 'outputs')
    if outputs is None:
        warn("outputs: not set — default folder layout used (derived at run time).")
    elif not isinstance(outputs, dict):
        err("outputs: if set, must be a mapping of step names to output folders.")
    if not isinstance(_get(cfg, 'envs'), dict):
        err("envs: required block mapping steps to conda env names.")

    # ── Scoring weights sanity (non-fatal) ────────────────────────────────────
    weights = _get(cfg, 'scoring.score_weights')
    if isinstance(weights, dict) and weights:
        total = sum(v for v in weights.values() if _is_num(v))
        if abs(total - 1.0) > 0.01:
            warn(f"scoring.score_weights: sum is {total:.3f}, expected ~1.0.")

    return (len(errors) == 0), errors, warnings


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(path):
        print(f"❌ Config not found: {path}")
        sys.exit(2)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    ok, errors, warnings = validate_config(cfg)

    print("------------------------------------------------")
    print(f"🧩 Validating {path}")
    print("------------------------------------------------")
    for w in warnings:
        print(f"   ⚠️  {w}")
    for e in errors:
        print(f"   ❌ {e}")
    if ok:
        print(f"✅ Config valid ({len(warnings)} warning(s)).")
        sys.exit(0)
    else:
        print(f"❌ Config INVALID: {len(errors)} error(s), {len(warnings)} warning(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
