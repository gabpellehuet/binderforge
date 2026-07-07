"""
==============================================================================
   🪪 BINDER IDENTITY
==============================================================================
   Deterministic, traceable binder_id for every design.

     binder_id = {binder_name}-{config_id:02}-{n}

   - binder_name : cfg['binder_name'] (falls back to cfg['project_name'])
   - config_id   : cfg['config_id']   (the configuration/run number, 0–99)
   - n           : sequential counter per run, gpu0 first then continuing into
                   gpu1, ordered by (gpu, batch, sample) ascending, starting at 1.

   `n` is assigned by sorting designs on their generation coordinates, so it is
   STABLE regardless of how rows are later ranked — the same design always gets
   the same binder_id within a run. The coordinates are parsed from the design
   name. Both generators now emit the CANONICAL format (see contract.DESIGN_NAME_RE):
     canonical : "gpu1_PROJ_B473_S0_model"    (gpu = "gpuN_", explicit _S sample)
   and the parser keeps a LEGACY fallback for pre-unification runs:
     old RFd   : "1-PROJ_B473_S0_model"       (gpu = bare leading "N-")
     old Cplxa : "gpu1_PROJ_B1259_model"      (no _S → sample 0)

   Shared by 04_Scoring.py and 05_Dashboard.py (both live in scripts/).
==============================================================================
"""
import re


def parse_design_coords(design):
    """Return (gpu, batch, sample) parsed from a design name; missing → 0.

    Prefers the canonical 'gpu{N}_' prefix; falls back to the legacy bare 'N-'
    prefix. Sample defaults to 0 for generators (or old data) without an _S field.
    """
    d = str(design)
    m = re.search(r'gpu(\d+)', d) or re.match(r'^(\d+)-', d)
    gpu = int(m.group(1)) if m else 0
    bm = re.search(r'_B(\d+)', d)
    batch = int(bm.group(1)) if bm else 0
    sm = re.search(r'_S(\d+)', d)
    sample = int(sm.group(1)) if sm else 0
    return gpu, batch, sample


def binder_name(cfg):
    """Identity prefix: binder_name, falling back to project_name, then 'BINDER'."""
    return str(cfg.get('binder_name') or cfg.get('project_name') or 'BINDER').strip()


def assign_binder_ids(df, cfg, design_col='design'):
    """Return a copy of df with a trailing 'binder_id' column.

    Numbering is per run, gpu0→gpu1, ordered by (gpu, batch, sample), from 1.
    """
    name = binder_name(cfg)
    config_id = int(cfg.get('config_id', cfg.get('run_id', 0)) or 0)
    coords = {i: parse_design_coords(df.at[i, design_col]) for i in df.index}
    order = sorted(df.index, key=lambda i: coords[i])
    mapping = {i: f"{name}-{config_id:02d}-{n}" for n, i in enumerate(order, start=1)}
    df = df.copy()
    df['binder_id'] = [mapping[i] for i in df.index]
    return df
