"""
==============================================================================
   🔌 STEP I/O CONTRACT  (model-agnostic interfaces between pipeline steps)
==============================================================================
   Single source of truth for how steps hand data to each other, so a NEW model
   for any step can plug in by honoring the same contract.

   CONVENTIONS
   -----------
   - Chains: chain A = BINDER, chain B = TARGET (everywhere).
   - Output folders (relative to the config folder = work_dir): see OUTPUT_LAYOUT.
   - Design name: a stable per-design string that flows unchanged from backbone →
     sequence → prediction → scoring (it is the `design` column, and identity.py
     parses gpu/batch/sample out of it for the binder_id).

   PER-STEP CONTRACT
   -----------------
   1 Backbone   in : target.pdb
                out: <01_BackboneGeneration>/gpu{N}/<design>.pdb   (chain A binder, B target)
   2 Sequence   in : the backbone PDBs
                out: <02_ProteinMPNN>/gpu{N}/seqs/<design>.fa  and  yaml/<design>_seq_*.yaml
   3 Prediction in : the step-2 design inputs                         [MODEL-SPECIFIC]
                out: predicted complex .cif + a confidence file       [MODEL-SPECIFIC]
                NOTE: not normalized yet — deferred to the LIVIA integration, which will
                map any model's prediction output into a single confidence schema.
   4 Scoring    in : predictions + backbones (via find_backbone)
                out: <04_Scoring>/<project>_summary.csv, _ranked.csv  (+ binder_id)
   5 Dashboard  in : _ranked.csv + predictions + backbones
                out: <05_Dashboard>/ HTML, Excel, PyMOL_Ready/
==============================================================================
"""
import os
import re
import glob

# ── Chain convention ──────────────────────────────────────────────────────────
BINDER_CHAIN = "A"
TARGET_CHAIN = "B"

# ── Design name (canonical) ───────────────────────────────────────────────────
# The design name is the stable per-design string that flows UNCHANGED from
# backbone → sequence → prediction → scoring (it becomes the `design` column, and
# identity.parse_design_coords parses gpu/batch/sample out of it for binder_id).
#
# CANONICAL FORMAT — every backbone generator MUST emit exactly this:
#       gpu{G}_{project}_B{batch}_S{sample}          e.g.  gpu0_PIV3_B473_S0
#   - gpu{G}   : GPU index the design was generated on (0-based). It also names the
#                backbone subfolder, so the design name is GLOBALLY UNIQUE and can
#                never collide with a same-numbered design from another GPU.
#   - {project}: project_name.
#   - B{batch} : generation batch index (zero-padding is optional; parser tolerant).
#   - S{sample}: sample index within the batch. Generators with no sample dimension
#                (e.g. Complexa) use S0.
#
# The backbone PDB on disk carries this SAME string as its basename:
#       <01_BackboneGeneration>/gpu{G}/gpu{G}_{project}_B{batch}_S{sample}.pdb
# so `design name == backbone-file basename` — the simplest possible resolver contract.
#
# LEGACY (pre-unification runs; still RESOLVED for back-compat, never EMITTED):
#   RFdiffusion : "{G}-{project}_B{batch}_S{sample}"   bare "G-" gpu prefix
#   Complexa    : "gpu{G}_{project}_B{batch}"          no _S; file lacked gpu prefix
DESIGN_NAME_RE = re.compile(r'^gpu(\d+)_(?P<proj>.+)_B(\d+)_S(\d+)$')


def design_name(gpu, project, batch, sample=0):
    """Build the canonical design name (== backbone-file basename) for a design.

    `batch` may be an int or a preformatted string (e.g. f"{i:04d}") so callers
    keep control of any zero-padding; the parser tolerates either width.
    """
    return f"gpu{int(gpu)}_{project}_B{batch}_S{sample}"

# ── Output folder layout (relative to work_dir = the config folder) ───────────
OUTPUT_LAYOUT = {
    'step1_hotspots':  'outputs/01_Hotspots',
    'step1_backbone':  'outputs/01_BackboneGeneration',
    'step2_mpnn':      'outputs/02_ProteinMPNN',
    'step3_boltz':     'outputs/03_Boltz',
    'step3_alphafast': 'outputs/03_AlphaFast',
    'step4_scoring':   'outputs/04_Scoring',
    'step5_dash':      'outputs/05_Dashboard',
}


def backbone_root(work_dir, cfg):
    """Absolute path to the backbone output folder for this run."""
    outs = cfg.get('outputs', {})
    rel = outs.get('step1_backbone', OUTPUT_LAYOUT['step1_backbone'])
    return os.path.join(work_dir, rel)


def find_backbone(backbone_root_dir, design_name):
    """Locate a design's backbone PDB under the per-gpu subfolders.

    For CANONICAL names the design name equals the backbone basename and carries a
    'gpu{N}_' prefix, so we resolve `gpu{N}/{design_name}.pdb` directly and the
    match is unambiguous (the prefix makes basenames globally unique).

    LEGACY fallbacks are kept for pre-unification runs: old Complexa files had no
    'gpu{N}_' prefix on disk (we retry the stripped basename inside the prefixed
    folder), and old RFdiffusion names used a bare 'N-' prefix with the gpu baked
    into the filename (found by the gpu0..gpu3 sweep). A final recursive glob is the
    last resort. Returns None if nothing matches. Centralised here so scoring and
    dashboard never diverge.
    """
    stripped = re.sub(r'^gpu\d+_', '', design_name)
    m = re.match(r'^(gpu\d+)_', design_name)
    if m:
        gpu = m.group(1)
        for nm in (design_name, stripped):
            path = os.path.join(backbone_root_dir, gpu, f"{nm}.pdb")
            if os.path.exists(path):
                return path
    for gpu in ('gpu0', 'gpu1', 'gpu2', 'gpu3'):
        for nm in (design_name, stripped):
            path = os.path.join(backbone_root_dir, gpu, f"{nm}.pdb")
            if os.path.exists(path):
                return path
    for nm in (design_name, stripped):
        hits = glob.glob(os.path.join(backbone_root_dir, "**", f"{nm}.pdb"), recursive=True)
        if hits:
            return hits[0]
    return None
