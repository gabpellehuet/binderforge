"""
==============================================================================
   📋 RESULTS COLUMN ORDER (shared)
==============================================================================
   Single source of truth for the column order of the scoring / dashboard
   results tables. Imported by 04_Scoring.py and 05_Dashboard.py so the CSV and
   the Excel never drift apart.
==============================================================================
"""

COLUMN_ORDER = [
    # Identification
    "design",
    # Composite score & tier
    "BinderScore", "candidate_tier",
    # Status & QC flags
    "status", "warnings", "elimination_reason",
    # Primary interface quality
    "ipsae_min", "iptm", "ptm", "model_confidence",
    # Physics metrics
    "dG", "sc_score", "dSASA", "unsat_hbonds", "clashes", "pDockQ",
    # RMSD
    "lrmsd", "rmsd_binder", "rmsd_complex", "rmsd_target",
    # Predictor confidence
    "af3_ranking_score", "boltz_confidence_score", "ipae",
    # AF3-specific quality flags
    "af3_frac_disordered", "af3_has_clash",
    # Binder refold (self-consistency)
    "refold_rmsd_binder", "refold_plddt", "refold_ptm",
    "refold_ranking_score", "refold_frac_disordered", "refold_has_clash",
    "complex_binder_plddt",
    # Secondary counts
    "hb_count", "sb_count", "contacts", "intf_residues",
    "polar_res", "hydro_res", "charged_res",
    "ratio_mp_pDockQ",
    "complex_iplddt", "complex_plddt", "complex_pde",
    # Sequence composition
    "has_aromatic", "best_seq_with_aromatic",
    # Always last
    "sequence",
]


def reorder_df(df, front_extra=None):
    """Return df with columns in COLUMN_ORDER; front_extra pinned first, unknown cols appended."""
    prefix = [c for c in (front_extra or []) if c in df.columns]
    ordered = prefix + [c for c in COLUMN_ORDER if c in df.columns and c not in prefix]
    tail = [c for c in df.columns if c not in ordered]
    return df[ordered + tail]
