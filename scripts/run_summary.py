"""
==============================================================================
   📝 RUN SUMMARY
==============================================================================
   After a run, write a human-readable digest of the config/run PARAMETERS and
   what was produced, to  outputs/run_summary.md.

   This is distinct from run_manifest.json (machine provenance: date/git/versions);
   the summary is the "what did I actually run and get" page a human reads.
==============================================================================
"""
import os
import glob
import datetime


def _count(patterns):
    n = 0
    for p in patterns:
        n += len(glob.glob(p, recursive=True))
    return n


def _rows(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path) as f:
            return max(0, sum(1 for _ in f) - 1)   # minus header
    except Exception:
        return None


def write_run_summary(cfg, outputs_dir, step="all", project_dir=None):
    """Write outputs/run_summary.md. Returns its path (or None on failure)."""
    work_dir = cfg.get('work_dir', os.path.dirname(outputs_dir))
    outs = cfg.get('outputs', {})

    def od(key, default):
        return os.path.join(work_dir, outs.get(key, default))

    gen = str(cfg.get('backbone_generator', 'rfdiffusion')).lower()
    predictor = str(cfg.get('predictor', 'boltz')).lower()
    binder = cfg.get('binder_name') or cfg.get('project_name') or 'BINDER'
    cid = int(cfg.get('config_id', 0) or 0)

    # ── output counts (best-effort) ────────────────────────────────────────────
    bb_dir = od('step1_backbone', 'outputs/01_BackboneGeneration')
    n_backbones = _count([os.path.join(bb_dir, "gpu*", "*.pdb"),
                          os.path.join(bb_dir, "gpu*", "*.cif")])
    mpnn_dir = od('step2_mpnn', 'outputs/02_ProteinMPNN')
    n_seqs = _count([os.path.join(mpnn_dir, "gpu*", "seqs", "*.fa")])
    pred_dir = od('step3_alphafast' if predictor == 'alphafast' else 'step3_boltz',
                  'outputs/03_AlphaFast' if predictor == 'alphafast' else 'outputs/03_Boltz')
    n_pred = _count([os.path.join(pred_dir, "**", "*.cif")])
    score_dir = od('step4_scoring', 'outputs/04_Scoring')
    n_scored = _rows(os.path.join(score_dir, f"{cfg.get('project_name', 'project')}_ranked.csv"))

    # ── backbone params block ──────────────────────────────────────────────────
    if gen == 'rfdiffusion':
        r = cfg.get('step1_rfd', {})
        bb_params = [
            f"- binder_length: {r.get('binder_length')}",
            f"- target_residues: {r.get('target_residues')}",
            f"- hotspot_residues: {r.get('hotspot_residues') or '[] (full surface)'}",
            f"- batch_size × num_batches: {r.get('batch_size')} × {r.get('num_batches')}",
            f"- step_scale / gamma_0: {r.get('step_scale')} / {r.get('gamma_0')}",
        ]
    else:
        c = cfg.get('step1_complexa', {})
        bb_params = [
            f"- binder_length: {c.get('binder_length')}",
            f"- nsamples / batch_size: {c.get('nsamples')} / {c.get('batch_size')}",
            f"- algorithm / reward_model: {c.get('algorithm')} / {c.get('reward_model')}",
        ]

    m = cfg.get('step2_mpnn', {})
    cutoffs = cfg.get('cutoffs', {})

    lines = []
    lines.append(f"# Run summary — {binder}-{cid:02d}")
    lines.append("")
    lines.append(f"_Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} — step(s) run: `{step}`_")
    lines.append("")
    lines.append("## Identity")
    lines.append(f"- binder_name: **{binder}**")
    lines.append(f"- config_id: **{cid:02d}**  → binder_id prefix `{binder}-{cid:02d}`")
    lines.append(f"- project_name (output label): {cfg.get('project_name')}")
    if project_dir:
        lines.append(f"- project folder: `{project_dir}`")
    lines.append(f"- target: `{cfg.get('target', {}).get('pdb')}`")
    lines.append("")
    lines.append(f"## Backbone — {gen}")
    lines += bb_params
    lines.append("")
    if gen != 'complexa':
        lines.append("## ProteinMPNN")
        lines.append(f"- seqs_per_target: {m.get('seqs_per_target')}")
        lines.append(f"- seqs_to_validate: {m.get('seqs_to_validate')}")
        lines.append(f"- sampling_temp: {m.get('sampling_temp')}")
        lines.append("")
    lines.append(f"## Prediction — {predictor}")
    lines.append(f"- predictor: {predictor}")
    lines.append("")
    lines.append("## Scoring cutoffs")
    for k, v in (cutoffs.items() if isinstance(cutoffs, dict) else []):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Produced")
    lines.append(f"- backbones: {n_backbones}")
    lines.append(f"- sequences: {n_seqs}")
    lines.append(f"- predictions (cif): {n_pred}")
    lines.append(f"- scored designs: {n_scored if n_scored is not None else '—'}")
    lines.append("")

    os.makedirs(outputs_dir, exist_ok=True)
    path = os.path.join(outputs_dir, "run_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
