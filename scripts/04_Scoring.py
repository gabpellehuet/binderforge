"""
==============================================================================
   🧬 (RFdiffusion-Proteinmpnn) vs Boltz/AF3 SCORING v4.0
==============================================================================
 📊 CSV OUTPUT LEGEND: UNDERSTANDING YOUR DATA
 ---------------------------------------------------------------------------
   
   1. IDENTIFIERS & METADATA
      - design:             Name of the design (e.g., design_001).
      - rmsd_complex:       Deviation of the entire complex vs hallucination.
      - rmsd_target:        Deviation of the target chain.
      - rmsd_binder:        Deviation of the binder chain.

   2. FOLDING & CONFIDENCE (Boltz or AlphaFold 3)
      - *_score:            Primary ranking score (Confidence for Boltz, Ranking for AF3).
      - iptm:               Interface Predicted TM-score (0-1).
      - ptm:                Predicted TM-score (0-1).
      - model_confidence:            (0-1) Model Confidence Score (0.8*ipTM + 0.2*pTM).
      - pDockQ:             (0-1) Physics-based DockQ calculated from contacts & pLDDT.
      - ratio_mp_pDockQ:    Ratio to detect AI hallucinations (~1.0 is GOOD).

   3. INTERFACE PHYSICS (Rosetta & Biopython)
      - dG:                 (kcal/mol) Binding Energy (< -10.0 is STRONG).
      - sc_score:           (0-1) Shape Complementarity (> 0.60 is GOOD).
      - dSASA:              (Å²) Interface Buried Surface Area.
      - unsat_hbonds:       Number of buried polar atoms without H-bonds (0 is PERFECT).
      - clashes:            Heavy-atom steric clashes (< 2.2Å). (>5 is a FAILED physics model).

   4. CHEMICAL COUNTS & COMPOSITION
      - hb_count:           Number of Hydrogen Bonds.
      - sb_count:           Number of Salt Bridges.
      - contacts:           Total number of heavy-atom contacts (< 5Å).
      - intf_residues:      Total number of residues at the interface (Chain A + B).
      - polar_res:          Count of polar residues.
      - hydro_res:          Count of hydrophobic residues.
      - charged_res:        Count of charged residues.
      - has_aromatic:       Does the binder sequence contain F/W/Y? (flagged in `warnings` if False).
      - best_seq_with_aromatic: Only set when has_aromatic is False — the best-scoring
                            ProteinMPNN candidate (of the seqs_per_target pool, not just
                            the one forwarded to prediction) that does contain F/W/Y.
                            Empty if none qualify or no MPNN pool exists (Complexa).
      - sequence:           Amino acid sequence of the Binder (Chain A).

Written by Naïs Sermet, Jean-Marie Bourhis, Gabriel Pellé-Huet, Claude and Gemini.
==============================================================================
"""

import os
import re
import sys
import yaml
import glob
import json
import math
import time
import logging
import argparse
import subprocess
import multiprocessing
import pandas as pd
import numpy as np
import warnings
import contextlib

from identity import assign_binder_ids
from contract import find_backbone

# =================================================================
# ⚙️ USER SETTINGS
# =================================================================
CONFIG_FILENAME = "config.yaml"

# =================================================================
# SCORING DEFAULTS  (overridden at runtime by config.yaml scoring: section)
# =================================================================

HARD_ELIMINATORS = {
    "lrmsd":               ("gt", 3.73,  "lRMSD-Fail"),
    "sc_score":            ("lt", 0.62,  "sc-Fail"),
    "iptm":                ("lt", 0.60,  "ipTM-Fail"),
    "dSASA":               ("lt", 800.0, "dSASA-Fail"),
    "clashes":             ("gt", 5,     "Clashes-Fail"),
    "af3_has_clash":       ("eq", True,  "AF3-Clash"),
    "af3_frac_disordered": ("gt", 0.30,  "Disorder-Fail"),
}
IPSAE_MIN_HARD_CUTOFF = 0.30
IPAE_HARD_CUTOFF      = 20.0

SCORE_WEIGHTS = {
    "ipsae_min":  0.65,
    "dG_dSASA":   0.25,
    "sc_score":   0.03,
    "unsat_pen":  0.05,
    "model_confidence":    0.02,
}

TIER1_CUTOFF = 0.75
TIER2_CUTOFF = 0.60

PASS_ALL = False   # test mode: mark every design PASS (skip hard eliminators)

SOFT_WARNINGS = {
    "unsat_hbonds":  ("gt", 3,     "High unsat_hbonds (>3)"),
    "rmsd_binder":   ("gt", 1.5,   "Moderate rmsd_binder (>1.5Å)"),
    "dG":            ("gt", 0.0,   "Positive dG — verify Rosetta setup"),
    "has_aromatic":  ("eq", False, "No aromatic residues (F/W/Y) in binder sequence"),
}

# Reason labels and warning messages — display-only, not user-configurable
_HARD_ELIMINATOR_REASONS = {
    "lrmsd":               "lRMSD-Fail",
    "sc_score":            "sc-Fail",
    "iptm":                "ipTM-Fail",
    "dSASA":               "dSASA-Fail",
    "clashes":             "Clashes-Fail",
    "af3_has_clash":       "AF3-Clash",
    "af3_frac_disordered": "Disorder-Fail",
    "ipsae_min":           "ipSAE-Fail",
    "ipsae":               "ipAE-Fail",
}
_SOFT_WARNING_MESSAGES = {
    "unsat_hbonds": "High unsat_hbonds",
    "rmsd_binder":  "Moderate rmsd_binder",
    "dG":           "Positive dG — verify Rosetta setup",
    "has_aromatic": "No aromatic residues (F/W/Y) in binder sequence",
}


def _apply_scoring_config(cfg: dict) -> None:
    """
    Override module-level scoring constants from the config.yaml scoring: section.
    Python defaults above are used for any key not present in config.
    Call once after loading config, before running the scoring pipeline.
    """
    global IPSAE_MIN_HARD_CUTOFF, IPAE_HARD_CUTOFF, TIER1_CUTOFF, TIER2_CUTOFF, PASS_ALL
    sc = cfg.get('scoring', {})
    PASS_ALL = bool(sc.get('pass_all', False))

    # Hard eliminators: merge — config entries override defaults, extras are added
    for col, pair in sc.get('hard_eliminators', {}).items():
        op, threshold = str(pair[0]), pair[1]
        reason = _HARD_ELIMINATOR_REASONS.get(col, f"{col}-Fail")
        HARD_ELIMINATORS[col] = (op, threshold, reason)

    IPSAE_MIN_HARD_CUTOFF = float(sc.get('ipsae_min_hard_cutoff', IPSAE_MIN_HARD_CUTOFF))
    IPAE_HARD_CUTOFF      = float(sc.get('ipae_hard_cutoff',      IPAE_HARD_CUTOFF))

    # Score weights: merge — config values override individual defaults
    SCORE_WEIGHTS.update(sc.get('score_weights', {}))

    TIER1_CUTOFF = float(sc.get('tier1_cutoff', TIER1_CUTOFF))
    TIER2_CUTOFF = float(sc.get('tier2_cutoff', TIER2_CUTOFF))

    # Soft warnings: merge — config entries override defaults, extras are added
    for col, pair in sc.get('soft_warnings', {}).items():
        op, threshold = str(pair[0]), pair[1]
        msg = _SOFT_WARNING_MESSAGES.get(col, str(col))
        SOFT_WARNINGS[col] = (op, threshold, msg)

from columns import COLUMN_ORDER, reorder_df as _reorder_df  # shared source of truth

# --- pi_score: ARCHIVED ---------------------------------------------------------
# pi_score was a placeholder (== intf_residues * 0.5) carrying no signal independent
# of intf_residues. A real pi_score needs per-interface-residue evolutionary
# conservation, which requires conservation software not yet integrated
# (e.g. per-residue Shannon entropy from the target MSA, or a ConSurf-style pipeline).
# Re-enable the line below and the computation in process_design_worker once that exists.
# DEFAULT_CONSERVATION = 0.5

# =================================================================
# 🧪 SCIENTIFIC SETUP
# =================================================================
# Suppress Biopython warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

try:
    from Bio.PDB import PDBParser, MMCIFParser, PDBIO, NeighborSearch, is_aa, Superimposer
    from Bio.Align import PairwiseAligner
    import pyrosetta
    from pyrosetta import init, pose_from_file
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
    from pyrosetta.rosetta.core.kinematics import MoveMap
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing Library ({e}).")
    sys.exit(1)

AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Canonical aromatic side chains (Phe/Trp/Tyr) — checked against the whole
# binder sequence, not just the interface (see has_aromatic below).
AROMATIC_RESIDUES = set('FWY')

# =================================================================
# SCORING FUNCTIONS  (hard eliminators + BinderScore)
# =================================================================

def apply_hard_eliminators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["status"] = "PASS"
    if PASS_ALL:
        print("   ⚠️  scoring.pass_all=true — all designs marked PASS (hard eliminators skipped).")
        df["elimination_reason"] = ""
        return df
    per_row_reasons = {i: [] for i in df.index}

    eliminators = dict(HARD_ELIMINATORS)
    if "ipsae_min" in df.columns and df["ipsae_min"].notna().any():
        eliminators["ipsae_min"] = ("lt", IPSAE_MIN_HARD_CUTOFF, "ipSAE-Fail")
    elif "ipsae" in df.columns and df["ipsae"].notna().any():
        eliminators["ipsae"] = ("gt", IPAE_HARD_CUTOFF, "ipAE-Fail")

    for col, (op, threshold, reason) in eliminators.items():
        if col not in df.columns:
            continue
        if op == "gt":
            mask = df[col] > threshold
        elif op == "lt":
            mask = df[col] < threshold
        elif op == "eq":
            mask = df[col] == threshold
        else:
            continue
        for idx in df.index[mask]:
            per_row_reasons[idx].append(reason)

    elimination_reason = pd.Series("", index=df.index)
    for idx, reasons in per_row_reasons.items():
        if reasons:
            df.at[idx, "status"] = "ELIMINATED"
            elimination_reason.at[idx] = "; ".join(reasons)
    df["elimination_reason"] = elimination_reason
    return df


def _rank_normalize_asc(series: pd.Series) -> pd.Series:
    pct = series.rank(pct=True, na_option="keep", ascending=True)
    return (1.0 - pct).clip(0.0, 1.0)


def compute_binder_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BinderScore"]    = np.nan
    df["candidate_tier"] = ""
    df["warnings"]       = ""

    pass_mask = df["status"] == "PASS"
    if not pass_mask.any():
        return df

    p = df.loc[pass_mask].copy()

    if "ipsae_min" in p.columns and p["ipsae_min"].notna().any():
        p["_ipsae"] = p["ipsae_min"].clip(0.0, 1.0).fillna(p["iptm"].clip(0.0, 1.0))
    else:
        p["_ipsae"] = p["iptm"].clip(0.0, 1.0)

    p["_dg_dsasa"] = (-(p["dG"] / p["dSASA"].clip(lower=1.0)) * 100).clip(0.0, 3.0) / 3.0
    p["_sc"]       = p["sc_score"].clip(0.0, 1.0)
    p["_unsat"]    = 1.0 / (1.0 + p["unsat_hbonds"] * 0.20)
    p["_mpdq"]     = p["model_confidence"].clip(0.0, 1.0)

    p["BinderScore"] = (
        SCORE_WEIGHTS["ipsae_min"] * p["_ipsae"]    +
        SCORE_WEIGHTS["dG_dSASA"]  * p["_dg_dsasa"] +
        SCORE_WEIGHTS["sc_score"]  * p["_sc"]       +
        SCORE_WEIGHTS["unsat_pen"] * p["_unsat"]    +
        SCORE_WEIGHTS["model_confidence"]   * p["_mpdq"]
    ).round(4)

    p["candidate_tier"] = "Tier3"
    p.loc[p["BinderScore"] >= TIER2_CUTOFF, "candidate_tier"] = "Tier2"
    p.loc[p["BinderScore"] >= TIER1_CUTOFF, "candidate_tier"] = "Tier1"

    for col, (op, threshold, msg) in SOFT_WARNINGS.items():
        if col not in p.columns:
            continue
        if op == "gt":
            flag = p[col] > threshold
        elif op == "lt":
            flag = p[col] < threshold
        elif op == "eq":
            flag = p[col] == threshold
        else:
            continue
        p.loc[flag, "warnings"] = p.loc[flag, "warnings"].str.cat(
            pd.Series(msg + "; ", index=p.index[flag]), na_rep=""
        )

    if "ipsae_min" in p.columns:
        p.loc[p["ipsae_min"] < 0.61, "warnings"] += "Weak ipSAE_min (<0.61); "
    elif "ipsae" in p.columns:
        p.loc[p["ipsae"] > 15.0, "warnings"] += "Weak ipSAE (>15Å); "

    p["warnings"] = p["warnings"].str.rstrip("; ")
    p.drop(columns=["_ipsae", "_dg_dsasa", "_sc", "_unsat", "_mpdq"], inplace=True)
    df.update(p[["BinderScore", "candidate_tier", "warnings"]])
    return df


# =================================================================
# BINDER REFOLD HELPERS
# =================================================================

def _refold_base_name(design):
    """Strip trailing _model suffix used in CSV design names."""
    return design[:-6] if design.endswith('_model') else design


def _first_chain(structure):
    for model in structure:
        for chain in model:
            return chain
        break
    return None


def _chain_mean_plddt(chain):
    vals = [r['CA'].get_bfactor() for r in chain if is_aa(r) and 'CA' in r]
    return round(float(np.mean(vals)), 2) if vals else float('nan')


def _build_monomer_inputs(top_df, in_dir, af_in="/data/af_input"):
    # af_in = path AF3 sees inside its container (in_dir is mounted there).
    os.makedirs(in_dir, exist_ok=True)
    empty_a3m = os.path.join(in_dir, "empty_paired.a3m")
    with open(empty_a3m, 'w') as f:
        f.write("")
    n = 0
    for _, row in top_df.iterrows():
        name = _refold_base_name(row['design'])
        seq  = str(row.get('sequence', '')).strip()
        if not seq or seq == 'nan':
            print(f"   ⚠️  {name}: no sequence in ranked CSV — skipped.")
            continue
        binder_a3m = f"{name}_binder.a3m"
        with open(os.path.join(in_dir, binder_a3m), 'w') as f:
            f.write(f">{name}_binder\n{seq}\n")
        af3_dict = {
            "name": name,
            "sequences": [{"protein": {
                "id": ["A"], "sequence": seq,
                "unpairedMsaPath": f"{af_in}/{binder_a3m}",
                "pairedMsaPath":   f"{af_in}/empty_paired.a3m",
                "templates": []
            }}],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 3,
        }
        with open(os.path.join(in_dir, f"{name}.json"), 'w') as f:
            json.dump(af3_dict, f, indent=2)
        n += 1
    return n


def _run_alphafast_refold(cfg, in_dir, results_dir):
    af  = cfg['step3_alphafast']
    cmd = [
        os.path.join(af['alphafast_dir'], "scripts/run_alphafast.sh"),
        "--input_dir",   in_dir,
        "--output_dir",  results_dir,
        "--db_dir",      af['db_dir'],
        "--weights_dir", af['weights_dir'],
        "--num_gpus",    str(af['num_gpus']),
        "--backend",     "docker",
        "--container",   "romerolabduke/alphafast:latest",
    ]
    print("\n🚀 LAUNCHING ALPHAFAST (monomer refold batch)...")
    subprocess.run(cmd, cwd=af['alphafast_dir'], check=True)
    print("✨ Refold batch complete.\n")


def _find_refold_cif(results_dir, name):
    p = os.path.join(results_dir, name, f"{name}_model.cif")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(results_dir, "**", f"*{name}*_model.cif"), recursive=True)
    return sorted(hits)[0] if hits else None


def _find_refold_summary(results_dir, name):
    p = os.path.join(results_dir, name, f"{name}_summary_confidences.json")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(results_dir, "**", f"*{name}*summary_confidences.json"), recursive=True)
    return sorted(hits)[0] if hits else None


def run_binder_refold(df_ranked: pd.DataFrame, cfg: dict, work_dir: str,
                      top_n: int, dash_dir: str) -> pd.DataFrame:
    """
    Run AlphaFast on the top-N PASS binders as monomers.
    Outputs land inside dash_dir (same folder as the dashboard):
      PyMOL_Ready/Refold_*.pml  — alongside complex .pml files
      refold_inputs/            — monomer JSONs for AlphaFast
      refold_results/           — AlphaFast monomer predictions
    Adds refold_rmsd_binder (and companion refold_* columns) to df_ranked.
    """
    pymol_dir   = os.path.join(dash_dir, "PyMOL_Ready")
    in_dir      = os.path.join(dash_dir, "refold_inputs")
    results_dir = os.path.join(dash_dir, "refold_results")
    for d in (pymol_dir, in_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    top = (df_ranked[df_ranked['status'] == 'PASS']
           .sort_values('BinderScore', ascending=False)
           .head(top_n))
    if top.empty:
        print("   ⚠️  No PASS candidates for refold.")
        return df_ranked

    print(f"\n   🔬 Binder Refold: top {len(top)} PASS candidates "
          f"(BinderScore {top['BinderScore'].min():.3f}–{top['BinderScore'].max():.3f})")

    n_in = _build_monomer_inputs(top, in_dir, cfg.get('tools', {}).get('af_input_dir', '/data/af_input'))
    print(f"   ✅ {n_in} monomer JSON inputs written to {in_dir}")
    _run_alphafast_refold(cfg, in_dir, results_dir)

    # Score each refold
    complex_dir = os.path.join(work_dir, cfg['outputs']['step3_alphafast'], "results")
    cif_parser  = MMCIFParser(QUIET=True)

    refold_cols = {
        "refold_rmsd_binder":    999.9,
        "refold_plddt":          float('nan'),
        "refold_ptm":            float('nan'),
        "refold_ranking_score":  float('nan'),
        "refold_frac_disordered": float('nan'),
        "refold_has_clash":      False,
        "complex_binder_plddt":  float('nan'),
    }
    for col, default in refold_cols.items():
        if col not in df_ranked.columns:
            df_ranked[col] = default

    for _, row in top.iterrows():
        name        = _refold_base_name(row['design'])
        complex_cif = _find_refold_cif(complex_dir, name)
        refold_cif  = _find_refold_cif(results_dir, name)

        refold_rmsd = 999.9
        refold_plddt = complex_binder_plddt = float('nan')
        if complex_cif and refold_cif:
            cpx   = cif_parser.get_structure("cpx", complex_cif)
            rfd   = cif_parser.get_structure("rfd", refold_cif)
            cpx_a = get_chain_by_id(cpx, 'A')
            rfd_a = _first_chain(rfd)
            if cpx_a is not None and rfd_a is not None:
                ra, ma     = positional_ca_match(cpx_a, rfd_a)
                refold_rmsd = iterative_superimpose(ra, ma)
                refold_plddt           = _chain_mean_plddt(rfd_a)
                complex_binder_plddt   = _chain_mean_plddt(cpx_a)
        else:
            print(f"   ⚠️  {name}: cif missing "
                  f"(complex={'ok' if complex_cif else 'MISSING'}, "
                  f"refold={'ok' if refold_cif else 'MISSING'})")

        r_ptm = r_rank = r_disorder = float('nan')
        r_clash = False
        summ = _find_refold_summary(results_dir, name) if refold_cif else None
        if summ:
            with open(summ) as f:
                s = json.load(f)
            r_ptm      = s.get('ptm',                  float('nan'))
            r_rank     = s.get('ranking_score',         float('nan'))
            r_disorder = s.get('fraction_disordered',   float('nan'))
            r_clash    = bool(s.get('has_clash', False))

        # Write PyMOL session (complex + refold)
        if complex_cif and refold_cif:
            tier       = row.get('candidate_tier', '')
            tier_label = f"_{tier}" if tier else ""
            pml_path   = os.path.join(pymol_dir, f"Refold_{name}{tier_label}.pml")
            pml = (
                f"# {name}  |  {tier if tier else 'PASS'}  |  refold RMSD = {refold_rmsd} Å\n"
                f"# complex_model : chain A = binder (bound), chain B = target\n"
                f"# refold_model  : chain A = binder predicted alone\n"
                f"load {os.path.abspath(complex_cif)}, complex_model\n"
                f"load {os.path.abspath(refold_cif)},  refold_model\n\n"
                f"hide everything, all\nshow cartoon, all\n\n"
                f"color gray70,  complex_model and chain B\n"
                f"color cyan,    complex_model and chain A\n"
                f"color magenta, refold_model  and chain A\n\n"
                f"align refold_model and chain A, complex_model and chain A\n"
                f"zoom complex_model and chain A\n"
                f"bg_color black\n"
                f"set cartoon_transparency, 0.1, complex_model and chain B\n"
            )
            with open(pml_path, "w") as f:
                f.write(pml)

        # Update df_ranked row
        mask = df_ranked['design'] == row['design']
        df_ranked.loc[mask, 'refold_rmsd_binder']    = refold_rmsd
        df_ranked.loc[mask, 'refold_plddt']           = refold_plddt
        df_ranked.loc[mask, 'refold_ptm']             = r_ptm
        df_ranked.loc[mask, 'refold_ranking_score']   = r_rank
        df_ranked.loc[mask, 'refold_frac_disordered'] = r_disorder
        df_ranked.loc[mask, 'refold_has_clash']       = r_clash
        df_ranked.loc[mask, 'complex_binder_plddt']   = complex_binder_plddt

    good = int((df_ranked['refold_rmsd_binder'] <= 2.0).sum())
    done = int((df_ranked['refold_rmsd_binder'] < 900).sum())
    print(f"   ✅ Refold complete: {done} scored | {good} hold fold (≤ 2.0 Å)")
    return df_ranked


# =================================================================
# AROMATIC FALLBACK  (best alternate ProteinMPNN sequence with F/W/Y)
# =================================================================
# The forwarded binder sequence (row['sequence'] / has_aromatic) is only the
# top-scoring ProteinMPNN candidate (step2_mpnn.seqs_to_validate). The other
# seqs_per_target candidates ProteinMPNN generated for the same backbone are
# never structurally predicted, but they're still sitting in the step-2
# fasta — cheap to re-scan for a design whose forwarded sequence has no
# aromatic. No fallback exists for backbone_generator=complexa (step 2 is
# skipped entirely there — no ProteinMPNN pool to pull from).

def _find_mpnn_fasta(mpnn_root, design_name):
    """Locate <02_ProteinMPNN>/gpu{N}/seqs/<design>.fa for a design name
    (mirrors contract.find_backbone's gpu-prefix resolution)."""
    m = re.match(r'^(gpu\d+)_', design_name)
    if m:
        path = os.path.join(mpnn_root, m.group(1), "seqs", f"{design_name}.fa")
        if os.path.exists(path):
            return path
    hits = glob.glob(os.path.join(mpnn_root, "**", "seqs", f"{design_name}.fa"), recursive=True)
    return hits[0] if hits else None


def _best_aromatic_alt_sequence(fasta_path):
    """Among ALL ProteinMPNN candidates for this design (not just the one
    forwarded to prediction), return the lowest-score (best) sequence that
    contains at least one aromatic residue — same score field/ordering as
    02_ProteinMPNN.py's get_top_n_seqs (lower ProteinMPNN score = better).
    None if the fasta is missing or no candidate qualifies."""
    if not fasta_path or not os.path.exists(fasta_path):
        return None
    with open(fasta_path) as f:
        entries = f.read().split('>')[1:]
    candidates = []
    for entry in entries:
        lines = entry.split('\n')
        header = lines[0]
        seq = "".join(lines[1:]).strip()
        if "original" in header.lower():
            continue
        score_part = [p for p in header.split(',') if "score=" in p]
        if not score_part:
            continue
        try:
            score = float(score_part[0].split('=')[1])
        except (IndexError, ValueError):
            continue
        if any(c in AROMATIC_RESIDUES for c in seq):
            candidates.append((score, seq))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def add_aromatic_fallback(df: pd.DataFrame, mpnn_root: str) -> pd.DataFrame:
    """For every row with has_aromatic == False, add the best-scoring
    ProteinMPNN candidate sequence (for that same backbone) that does
    contain an aromatic residue, if one exists."""
    if "has_aromatic" not in df.columns:
        return df
    df["best_seq_with_aromatic"] = ""
    needs_fallback = df.index[~df["has_aromatic"].astype(bool)]
    for idx in needs_fallback:
        # MPNN fastas are keyed by the bare backbone name — "design" carries
        # the predicted-structure "_model" suffix, same distinction _refold_base_name
        # exists for.
        backbone_name = _refold_base_name(df.at[idx, "design"])
        fasta = _find_mpnn_fasta(mpnn_root, backbone_name)
        alt_seq = _best_aromatic_alt_sequence(fasta)
        if alt_seq:
            df.at[idx, "best_seq_with_aromatic"] = alt_seq
    return df


# =================================================================
# 2. HELPER: SILENCER
# =================================================================
@contextlib.contextmanager
def suppress_output():
    """Redirects C++ level stdout/stderr to /dev/null."""
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(sys.stdout.fileno())
        old_stderr = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout, sys.stdout.fileno())
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stdout)   # close the saved descriptors
            os.close(old_stderr)

# =================================================================
# 3. HELPER FUNCTIONS
# =================================================================
def get_parent_base_name(filename):
    name = filename.replace('.cif', '').replace('.pdb', '')
    if "_seed-" in name: name = name.split('_seed-')[0]
    if "_model" in name: name = name.split('_model')[0]
    if "_seq_" in name: name = name.split('_seq_')[0]
    return name

def find_reference_pdb(rf_root, base_name):
    # Backbone resolution is centralised in contract.py (shared with the dashboard).
    return find_backbone(rf_root, base_name)

def get_chains_in_order(structure):
    chains = []
    for model in structure:
        for chain in model:
            chains.append(chain)
        break
    return chains

def get_chain_by_id(structure, chain_id):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return chain
        break
    return None

def extract_sequence_and_atoms(chain):
    seq = ""; ca_atoms = []
    for res in chain:
        if is_aa(res) and 'CA' in res:
            seq += AA_MAP.get(res.get_resname(), 'X')
            ca_atoms.append(res['CA'])
    return seq, ca_atoms

def align_and_get_matching_atoms(ref_chain, mod_chain):
    ref_seq, ref_ca = extract_sequence_and_atoms(ref_chain)
    mod_seq, mod_ca = extract_sequence_and_atoms(mod_chain)
    
    if not ref_seq or not mod_seq: return [], []
        
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2; aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5; aligner.extend_gap_score = -0.1
    
    alignments = aligner.align(ref_seq, mod_seq)
    if not alignments: return [], []
        
    best_aln = alignments[0]
    ref_matched, mod_matched = [], []
    
    for (ref_start, ref_end), (mod_start, mod_end) in zip(best_aln.aligned[0], best_aln.aligned[1]):
        for i, j in zip(range(ref_start, ref_end), range(mod_start, mod_end)):
            ref_matched.append(ref_ca[i])
            mod_matched.append(mod_ca[j])
            
    return ref_matched, mod_matched

def positional_ca_match(chain1, chain2):
    """Return all CA atoms paired by residue order (index i → index i).
    The backbone PDB and AlphaFast CIF always have the same length for both
    binder (ProteinMPNN/Complexa designs to exact backbone length) and target
    (same protein in both), so positional matching is always correct.
    Sequence alignment is avoided: for RFdiffusion designs the backbone residues
    differ from the MPNN-designed sequence, and alignment gaps would silently
    drop residues and bias the RMSD.
    """
    ca1 = [r['CA'] for r in chain1 if is_aa(r) and 'CA' in r]
    ca2 = [r['CA'] for r in chain2 if is_aa(r) and 'CA' in r]
    n   = min(len(ca1), len(ca2))
    return ca1[:n], ca2[:n]


def iterative_superimpose(ref_atoms, mod_atoms, max_cycles=5, cutoff=2.0):
    if len(ref_atoms) != len(mod_atoms) or len(ref_atoms) < 3: return 999.9
    active_indices = list(range(len(ref_atoms)))
    si = Superimposer()
    
    for cycle in range(max_cycles):
        cur_ref = [ref_atoms[i] for i in active_indices]
        cur_mod = [mod_atoms[i] for i in active_indices]
        si.set_atoms(cur_ref, cur_mod)
        rot, tran = si.rotran
        new_indices = []
        for i in range(len(ref_atoms)):
            mod_coord = np.dot(mod_atoms[i].coord, rot) + tran
            if np.linalg.norm(ref_atoms[i].coord - mod_coord) <= cutoff:
                new_indices.append(i)
        if len(new_indices) == len(active_indices) or len(new_indices) < 3: break
        active_indices = new_indices
        
    si.set_atoms([ref_atoms[i] for i in active_indices], [mod_atoms[i] for i in active_indices])
    return round(si.rms, 3)

def calculate_chain_rmsd(ref_structure, model_structure, chain_id):
    try:
        ref_chain = get_chain_by_id(ref_structure, chain_id)
        mod_chain = get_chain_by_id(model_structure, chain_id)
        if ref_chain is None or mod_chain is None: return 999.9
        ref_atoms, mod_atoms = positional_ca_match(ref_chain, mod_chain)
        return iterative_superimpose(ref_atoms, mod_atoms)
    except: return 999.9

def calculate_complex_rmsd(ref_structure, model_structure):
    try:
        ref_a = get_chain_by_id(ref_structure, 'A')
        ref_b = get_chain_by_id(ref_structure, 'B')
        mod_a = get_chain_by_id(model_structure, 'A')
        mod_b = get_chain_by_id(model_structure, 'B')
        if None in (ref_a, ref_b, mod_a, mod_b): return 999.9
        rA, mA = positional_ca_match(ref_a, mod_a)
        rB, mB = positional_ca_match(ref_b, mod_b)
        return iterative_superimpose(rA + rB, mA + mB)
    except: return 999.9

def calculate_lrmsd(ref_structure, model_structure):
    """
    Ligand RMSD: superimpose the target (B), then measure binder (A) deviation
    without any re-fitting of the binder. Both steps use positional CA matching
    (index i → index i) — see positional_ca_match for rationale.
    """
    try:
        ref_b = get_chain_by_id(ref_structure, 'B')
        mod_b = get_chain_by_id(model_structure, 'B')
        ref_a = get_chain_by_id(ref_structure, 'A')
        mod_a = get_chain_by_id(model_structure, 'A')
        if None in (ref_a, ref_b, mod_a, mod_b): return 999.9

        # Superimpose target chains (positional — same protein, same length)
        r_b, m_b = positional_ca_match(ref_b, mod_b)
        if len(r_b) < 3: return 999.9
        si = Superimposer()
        si.set_atoms(r_b, m_b)   # moves m_b in-place; rot/tran maps mod→ref frame
        rot, tran = si.rotran

        # Binder step: positional matching (index i → index i), no sequence alignment
        ref_ca = [r['CA'] for r in ref_a if is_aa(r) and 'CA' in r]
        mod_ca = [r['CA'] for r in mod_a if is_aa(r) and 'CA' in r]
        n = min(len(ref_ca), len(mod_ca))
        if n < 3: return 999.9

        ref_coords = np.array([a.coord for a in ref_ca[:n]])
        mod_coords = np.array([a.coord for a in mod_ca[:n]])   # original, untransformed
        transformed = np.dot(mod_coords, rot) + tran
        return round(float(np.sqrt(np.mean(np.sum((ref_coords - transformed) ** 2, axis=1)))), 3)
    except: return 999.9

# =================================================================
# 3b. HELPER: ipSAE and ipAE from AF3 full_data.json
# =================================================================
def _ipsae_d0(n: int) -> float:
    """TM-score d0 length correction applied to interface size."""
    if n <= 0:
        return 1.0
    if n > 21:
        return 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8
    return max(0.5 * (n ** 0.5), 0.1)


def _ipsae_one_direction(pae: np.ndarray, query_len: int, aligned_len: int,
                         pae_cutoff: float = 10.0) -> float:
    """
    Compute ipSAE for one direction (query chain → aligned chain).

    For each query residue i, average the TM-score-like term
    1/(1+(PAE_ij/d0)^2) over aligned residues j where PAE_ij < pae_cutoff,
    where d0 scales with the number of such j. Return the MAX over all i.

    Formula: Dunbrack 2025 / Overath et al. 2025 (bioRxiv 2025.08.14).
    Output is 0–1, higher = better (unlike raw PAE which is in Angstroms).
    """
    block = pae[:query_len, query_len:query_len + aligned_len]
    best = 0.0
    for i in range(query_len):
        row = block[i]
        confident = row[row < pae_cutoff]
        n_j = len(confident)
        if n_j == 0:
            continue
        d0 = _ipsae_d0(n_j)
        score = float(np.mean(1.0 / (1.0 + (confident / d0) ** 2)))
        if score > best:
            best = score
    return best


def compute_ipsae(pae_json_path: str, chain_a_len: int, chain_b_len: int,
                  pae_cutoff: float = 10.0) -> tuple[float, float]:
    """
    Compute ipSAE_min and ipAE from an AF3 *_full_data.json.

    ipSAE_min (0–1, higher = better):
        The best single-metric predictor of experimental binding success
        per Overath et al. 2025 (3,766 binders, 15 targets).
        Applies TM-score functional to PAE values, restricts to
        high-confidence pairs (PAE < pae_cutoff), and takes the minimum
        of the two asymmetric directional scores (weakest link).
        Threshold to keep: > 0.61 (from paper's F1-maximisation).

    ipAE (in Å, lower = better):
        Simple mean cross-chain PAE — cheaper but ~1.4× less predictive.
        Included for reference and backward compatibility.

    Both require AF3's *_full_data.json (not *_summary_confidences.json).
    Returns (ipsae_min, ipae) — NaN for both on any failure.
    """
    nan = float('nan')
    try:
        with open(pae_json_path) as f:
            data = json.load(f)
        pae_raw = data.get("pae")
        if pae_raw is None:
            return nan, nan

        N = chain_a_len + chain_b_len
        pae = np.array(pae_raw, dtype=np.float32)
        if pae.ndim == 1:
            if len(pae_raw) != N * N:
                return nan, nan
            pae = pae.reshape(N, N)
        elif pae.shape != (N, N):
            return nan, nan

        # ipAE: simple mean of cross-chain blocks (in Å)
        ipae = float((pae[:chain_a_len, chain_a_len:].mean() +
                      pae[chain_a_len:, :chain_a_len].mean()) / 2.0)

        # ipSAE_min: TM-score functional, PAE cutoff, weakest-link direction
        score_ab = _ipsae_one_direction(pae, chain_a_len, chain_b_len, pae_cutoff)
        # For B→A direction, transpose the matrix so query=B, aligned=A
        pae_T = pae.T
        score_ba = _ipsae_one_direction(pae_T, chain_b_len, chain_a_len, pae_cutoff)
        ipsae_min = min(score_ab, score_ba)

        return round(ipsae_min, 4), round(ipae, 3)
    except Exception:
        return nan, nan

# =================================================================
# 4. METRIC CALCULATORS
# =================================================================
def analyze_rosetta_physics(clean_pdb_path):
    try:
        with suppress_output():
            sfxn = ScoreFunctionFactory.create_score_function("ref2015")

            pose = pose_from_file(clean_pdb_path)

            # Side-chain minimize to relieve clashes in AF3/Boltz structures.
            # Backbone AND inter-chain jump fixed — preserves predicted geometry, docking, and RMSD validity.
            # NOTE: jump was True; that let the rigid-body jump slide the chains during minimization,
            # systematically depressing sc (binder 75: 0.656 -> 0.619, below the 0.62 gate). See RF_OLD_vs_NEW analysis.
            mm = MoveMap()
            mm.set_bb(False)
            mm.set_chi(True)
            mm.set_jump(False)
            MinMover(mm, sfxn, "lbfgs_armijo_nonmonotone", 0.001, True).apply(pose)

            iam = InterfaceAnalyzerMover(1)
            iam.set_scorefunction(sfxn)
            iam.set_pack_separated(True)
            iam.set_compute_packstat(True)
            iam.set_compute_interface_sc(True)
            iam.set_calc_hbond_sasaE(True)
            iam.apply(pose)
            
            data = iam.get_all_data()
            # sc_value = shape complementarity (Lawrence & Colman).
            # get_interface_packstat() is a different metric — do NOT use as fallback.
            sc = data.sc_value if hasattr(data, 'sc_value') else 0.0
            dG = iam.get_interface_dG()
            dSASA = iam.get_interface_delta_sasa()
            unsat = data.delta_unsatHbonds if hasattr(data, 'delta_unsatHbonds') else 0
            
            return round(sc, 3), round(dG, 3), round(dSASA, 2), unsat
    except Exception as e:
        logging.error(f"Rosetta calculation failed for {clean_pdb_path}: {e}")
        return 0.0, 0.0, 0.0, 0

def get_bio_metrics(clean_pdb_path):
    metrics = {"sequence": "", "pDockQ": 0.0, "hb": 0, "sb": 0, "Polar": 0, "Hydrophobic": 0, "Charged": 0, "contact_pairs": 0, "Num_intf_residues": 0, "clashes": 0}
    try:
        parser = PDBParser(QUIET=True)
        s = parser.get_structure("temp", clean_pdb_path)
        chain_a = get_chain_by_id(s, 'A')
        chain_b = get_chain_by_id(s, 'B')
        if chain_a is None or chain_b is None: return metrics
        metrics["sequence"] = "".join([AA_MAP.get(r.get_resname(), 'X') for r in chain_a if is_aa(r)])
        
        atoms_a = [a for a in chain_a.get_atoms() if is_aa(a.get_parent())]
        ns_b = NeighborSearch([a for a in chain_b.get_atoms() if is_aa(a.get_parent())])
        
        intf_res = set(); hb, sb, contacts, clashes = 0, 0, 0, 0
        cat, an = {'LYS', 'ARG', 'HIS'}, {'ASP', 'GLU'}
        
        for a in atoms_a:
            nbs = ns_b.search(a.get_coord(), 5.0)
            if not nbs: continue
            contacts += len(nbs)
            intf_res.add(a.get_parent())
            
            for n in nbs:
                dist = a - n
                # HBonds
                if a.name[0] in ['N','O'] and n.name[0] in ['N','O'] and dist < 3.5: hb += 1
                # Clashes (Heavy atoms closer than 2.2A, excluding Hbonds)
                if dist < 2.2 and not (a.name[0] in ['N','O'] and n.name[0] in ['N','O']): clashes += 1
                
            r_a = a.get_parent().get_resname()
            if r_a in cat or r_a in an:
                for n in nbs:
                    r_b = n.get_parent().get_resname()
                    if ((r_a in cat and r_b in an) or (r_a in an and r_b in cat)) and (a-n)<4.0: sb += 1
        
        metrics.update({"hb": hb, "sb": sb, "contact_pairs": contacts, "Num_intf_residues": len(intf_res), "clashes": clashes})
        
        for r in intf_res:
            n = r.get_resname()
            if n in {'SER','THR','TYR','ASN','GLN','CYS','HIS'}: metrics["Polar"]+=1
            elif n in {'ALA','VAL','ILE','LEU','MET','PHE','TRP','PRO','GLY'}: metrics["Hydrophobic"]+=1
            if n in {'ASP','GLU','LYS','ARG'}: metrics["Charged"]+=1
        
        # pDockQ (Bryant et al. 2022): canonical interface = Cβ–Cβ < 8 Å (Cα for GLY).
        # The logistic constants below were FIT to x computed this way, so the contact
        # definition must match — do NOT reuse the 5 Å all-atom `contacts` above here.
        def _cb_atom(res):
            if 'CB' in res:
                return res['CB']
            if res.get_resname() == 'GLY' and 'CA' in res:
                return res['CA']
            return None

        cb_a = [(r, _cb_atom(r)) for r in chain_a if is_aa(r)]
        cb_b = [(r, _cb_atom(r)) for r in chain_b if is_aa(r)]
        cb_a = [(r, a) for r, a in cb_a if a is not None]
        cb_b = [(r, a) for r, a in cb_b if a is not None]
        if cb_a and cb_b:
            ns_cb = NeighborSearch([a for _, a in cb_b])
            intf_cb = set()
            n_cb_contacts = 0
            for r, a in cb_a:
                near = ns_cb.search(a.get_coord(), 8.0)
                if near:
                    intf_cb.add(r)
                    n_cb_contacts += len(near)
                    for nb in near:
                        intf_cb.add(nb.get_parent())
            if n_cb_contacts > 0:
                plddts = [(rr['CB'] if 'CB' in rr else rr['CA']).get_bfactor()
                          for rr in intf_cb if ('CB' in rr or 'CA' in rr)]
                x = float(np.mean(plddts)) * math.log10(n_cb_contacts)
                metrics["pDockQ"] = round(0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018, 3)
            
    except Exception as e:
        logging.error(f"Biopython calculation failed for {clean_pdb_path}: {e}")
    return metrics

# =================================================================
# 5. WORKER PROCESS
# =================================================================
def init_worker():
    """Runs exactly ONCE per CPU core when the worker pool is created."""
    try:
        with suppress_output(): 
            pyrosetta.init("-mute all")
    except Exception as e:
        # If this fails, the worker is dead anyway, but we log it.
        pass
def process_design_worker(args):
    cif_path, rf_root = args
    pid = multiprocessing.current_process().pid

    try:
        filename = os.path.basename(cif_path)
        job_name = filename.replace('.cif', '').replace('.pdb', '')
        parent_base_name = get_parent_base_name(filename)
        rf_pdb_path = find_reference_pdb(rf_root, parent_base_name)
        
        if not rf_pdb_path:
            logging.error(f"Missing Reference PDB for {filename}. Expected: {parent_base_name}.pdb")
        
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(f"struct_{pid}", cif_path)
        
        rmsd_target = 999.9; rmsd_binder = 999.9; rmsd_complex = 999.9; lrmsd = 999.9
        if rf_pdb_path:
            ref_parser = PDBParser(QUIET=True)
            ref_structure = ref_parser.get_structure("reference", rf_pdb_path)
            rmsd_binder = calculate_chain_rmsd(ref_structure, structure, 'A')
            rmsd_target = calculate_chain_rmsd(ref_structure, structure, 'B')
            rmsd_complex = calculate_complex_rmsd(ref_structure, structure)
            lrmsd = calculate_lrmsd(ref_structure, structure)

        temp_pdb = f"temp_{pid}_{job_name}.pdb"
        io = PDBIO(); io.set_structure(structure); io.save(temp_pdb)

        # JSON Confidence Parsing
        metrics = {"iptm": 0.0, "ptm": 0.0, "complex_iplddt": 0.0, "complex_plddt": 0.0, "complex_pde": 0.0,
            "af3_fraction_disordered": 0.0, "af3_has_clash": False}
        ranking_score = 0.0 
        search_dir = os.path.dirname(cif_path)
        
        boltz_conf_path = cif_path.replace(".cif", "_confidence.json")
        af3_conf_path = cif_path.replace("_model.cif", "_summary_confidences.json")
        
        if not os.path.exists(boltz_conf_path) and not os.path.exists(af3_conf_path):
            possible_jsons = glob.glob(os.path.join(search_dir, "*confidence*.json"))
            for pj in possible_jsons:
                if "summary_confidences" in pj: af3_conf_path = pj
                elif "confidence" in pj: boltz_conf_path = pj

        if os.path.exists(boltz_conf_path):
            with open(boltz_conf_path, 'r') as f:
                conf = json.load(f)
                ranking_score = conf.get("confidence_score", 0.0)
                metrics.update({"iptm": conf.get("iptm", 0.0), "ptm": conf.get("ptm", 0.0),
                                "complex_iplddt": conf.get("complex_iplddt", 0.0),
                                "complex_plddt": conf.get("complex_plddt", 0.0), "complex_pde": conf.get("complex_pde", 0.0)})
        elif os.path.exists(af3_conf_path):
            with open(af3_conf_path, 'r') as f:
                conf = json.load(f)
                ranking_score = conf.get("ranking_score", 0.0)
                metrics.update({
                    "iptm": conf.get("iptm", 0.0),
                    "ptm": conf.get("ptm", 0.0),
                    "af3_fraction_disordered": conf.get("fraction_disordered", 0.0),
                    "af3_has_clash": conf.get("has_clash", False)
                })

        # --- ipSAE_min + ipAE: from AF3 full_data.json when available ---
        # ipSAE_min (0-1, higher=better): best single predictor of experimental
        #   binding (Overath et al. 2025, 3766 binders). Uses TM-score functional
        #   on PAE with cutoff=10A. Keep designs where ipSAE_min > 0.61.
        # ipAE (Angstroms, lower=better): simple mean cross-chain PAE for reference.
        ipsae_min = float('nan')
        ipae = float('nan')
        af3_full_path = cif_path.replace("_model.cif", "_full_data.json")
        if not os.path.exists(af3_full_path):
            af3_full_path = cif_path.replace("_model.cif", "_confidences.json")
        if os.path.exists(af3_full_path):
            try:
                chains = get_chains_in_order(structure)
                if len(chains) >= 2:
                    chain_a_len = sum(1 for r in chains[0] if is_aa(r))
                    chain_b_len = sum(1 for r in chains[1] if is_aa(r))
                    ipsae_min, ipae = compute_ipsae(af3_full_path, chain_a_len, chain_b_len)
            except Exception as e:
                logging.error(f"ipSAE failed for {filename}: {e}")

        sc, dG, dSASA, unsat = analyze_rosetta_physics(temp_pdb)
        bio = get_bio_metrics(temp_pdb)
        
        model_confidence = (0.8 * metrics["iptm"]) + (0.2 * metrics["ptm"])
        ratio = model_confidence / bio["pDockQ"] if bio["pDockQ"] > 0.01 else 0.0

        if os.path.exists(temp_pdb): os.remove(temp_pdb)

        return {
            "design": job_name,
            "rmsd_complex": rmsd_complex, "rmsd_target": rmsd_target, "rmsd_binder": rmsd_binder, "lrmsd": lrmsd,
            "generic_score": round(ranking_score, 3), # Will be renamed in main()
            "iptm": round(metrics["iptm"], 3), "ptm": round(metrics["ptm"], 3),
            "ipsae_min": ipsae_min, "ipae": ipae,
            "af3_frac_disordered": round(metrics["af3_fraction_disordered"], 3),
            "af3_has_clash": metrics["af3_has_clash"],
            "complex_iplddt": round(metrics["complex_iplddt"], 3), "complex_plddt": round(metrics["complex_plddt"], 3), "complex_pde": round(metrics["complex_pde"], 3),
            "pDockQ": bio["pDockQ"], "model_confidence": round(model_confidence, 3),
            "dG": dG, "dSASA": dSASA, "sc_score": sc, "unsat_hbonds": unsat, "clashes": bio["clashes"],
            "ratio_mp_pDockQ": round(ratio, 2),
            "hb_count": bio["hb"], "sb_count": bio["sb"], "contacts": bio["contact_pairs"], "intf_residues": bio["Num_intf_residues"],
            "polar_res": bio["Polar"], "hydro_res": bio["Hydrophobic"], "charged_res": bio["Charged"],
            "has_aromatic": any(c in AROMATIC_RESIDUES for c in bio["sequence"]),
            "sequence": bio["sequence"]
        }
    except Exception as e: 
        logging.error(f"Failed to process {cif_path}: {e}", exc_info=True)
        return None

# =================================================================
# MAIN EXECUTION
# =================================================================
def main():
    ap = argparse.ArgumentParser(description="Step 4: Scoring + BinderScore + Binder Refold")
    ap.add_argument('--config',    default=CONFIG_FILENAME)
    ap.add_argument('--no-refold', action='store_true',
                    help="Skip binder refold (AlphaFast monomer) phase")
    ap.add_argument('--top',       type=int, default=None,
                    help="Override top-N PASS designs to refold (default: scoring.refold_top_n in config)")
    args = ap.parse_args()

    print("------------------------------------------------")
    print("🧬 STEP 4: Scoring Analysis (v4.0 - AF3 & Boltz)")
    print("------------------------------------------------")

    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    _apply_scoring_config(cfg)

    WORK_DIR = cfg.get('work_dir', './')
    # Unified backbone dir: step1_backbone (new) with fallback to old per-generator keys
    _outputs = cfg.get('outputs', {})
    _gen     = cfg.get('backbone_generator', 'rfdiffusion').lower()
    RF_ROOT  = os.path.join(WORK_DIR, _outputs.get('step1_backbone',
               _outputs.get('step1_complexa' if _gen == 'complexa' else 'step1_rfd',
               'outputs/01_BackboneGeneration')))
    SCORE_DIR = os.path.join(WORK_DIR, cfg['outputs']['step4_scoring'])
    os.makedirs(SCORE_DIR, exist_ok=True)

    # Logging Setup
    log_file = os.path.join(SCORE_DIR, "scoring_errors.log")
    logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - [%(processName)s] - %(message)s')

    predictor = cfg.get('predictor', 'boltz').lower()
    if predictor == 'alphafast':
        PRED_DIR = os.path.join(WORK_DIR, cfg['outputs']['step3_alphafast'], "results")
    else:
        PRED_DIR = os.path.join(WORK_DIR, cfg['outputs']['step3_boltz'])

    all_cif_files = glob.glob(os.path.join(PRED_DIR, "**", "*.cif"), recursive=True)
    
    # 🧹 QUICK TRIAGE FILTER: Only keep the top-level representative model
    cif_files = []
    for f in all_cif_files:
        path_parts = f.split(os.sep)
        filename = os.path.basename(f)
        
        # 1. Ignore Boltz sub-folder (usually called 'predictions')
        if "predictions" in path_parts:
            continue
            
        # 2. Ignore sub-folders like 'seed-1_sample-1'
        if any("seed-" in part or "sample-" in part for part in path_parts[:-1]):
            continue
            
        # 3. If everything is in one folder, ignore sample-1, sample-2, etc. 
        # (We keep it if it doesn't have 'sample' in the name, OR if it's explicitly 'sample-0')
        if "sample-" in filename and "sample-0" not in filename:
            continue
            
        cif_files.append(f)

    if not cif_files:
        print(f"❌ ERROR: No valid top-level CIF files found in {PRED_DIR}")
        sys.exit(1)

    print(f"🔍 Found {len(cif_files)} designs to score from {predictor.upper()}.")
    
    results = []
    NUM_CORES = max(1, multiprocessing.cpu_count() - 2)

    with multiprocessing.Pool(processes=NUM_CORES, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_design_worker, [(f, RF_ROOT) for f in cif_files])):
            if res: results.append(res)
            if i % 5 == 0 or i == len(cif_files)-1:
                percent = ((i+1) / len(cif_files)) * 100
                bar = '█' * int(30 * percent / 100)
                sys.stdout.write(f"\r[{bar:<30}] {percent:.1f}%")
                sys.stdout.flush()

    print(f"\n------------------------------------------------")
    if not results:
        print("❌ ERROR: No valid scores generated.")
        return

    df = pd.DataFrame(results)

    # Rename primary score column, drop predictor-specific extras
    score_col_name = 'af3_ranking_score' if predictor == 'alphafast' else 'boltz_confidence_score'
    df.rename(columns={'generic_score': score_col_name}, inplace=True)
    if predictor == 'alphafast':
        df.drop(columns=[c for c in ['complex_iplddt', 'complex_plddt', 'complex_pde'] if c in df.columns], inplace=True)
    elif predictor == 'boltz':
        df.drop(columns=[c for c in ['af3_frac_disordered', 'af3_has_clash'] if c in df.columns], inplace=True)

    # Aromatic fallback: for any design whose forwarded sequence has no F/W/Y,
    # look up the best-scoring alternate ProteinMPNN candidate that does.
    # No-op (empty column) for backbone_generator=complexa — no MPNN pool there.
    MPNN_ROOT = os.path.join(WORK_DIR, _outputs.get('step2_mpnn', 'outputs/02_ProteinMPNN'))
    df = add_aromatic_fallback(df, MPNN_ROOT)

    project_name = cfg.get('project_name', os.path.basename(os.path.abspath(WORK_DIR))) or "project"

    # Traceable binder_id (last column) — {binder_name}-{run_id:02}-{n}, stable per run
    df = assign_binder_ids(df, cfg)

    # --- Phase 1 output: raw summary CSV ---
    out_summary = os.path.join(SCORE_DIR, f"{project_name}_summary.csv")
    df.to_csv(out_summary, index=False)
    print(f"   💾 Summary CSV: {os.path.basename(out_summary)}")

    # --- Phase 2: apply hard eliminators + BinderScore ---
    print("   🔬 Applying hard eliminators + computing BinderScore...")
    df = apply_hard_eliminators(df)
    df = compute_binder_scores(df)

    n_pass = int((df['status'] == 'PASS').sum())
    n_elim = int((df['status'] == 'ELIMINATED').sum())
    print(f"   📊 {n_pass} PASS | {n_elim} ELIMINATED  (of {len(df)} total)")
    if n_pass > 0:
        tier_counts = df[df['status'] == 'PASS']['candidate_tier'].value_counts()
        for tier in ['Tier1', 'Tier2', 'Tier3']:
            print(f"      {tier}: {tier_counts.get(tier, 0)}")

    df_pass = df[df['status'] == 'PASS'].sort_values('BinderScore', ascending=False)
    df_elim = df[df['status'] == 'ELIMINATED'].sort_values('elimination_reason')
    df_ranked = pd.concat([df_pass, df_elim], ignore_index=True)
    df_ranked = _reorder_df(df_ranked)

    # --- Phase 3: binder refold (outputs go into DASH_DIR alongside dashboard files) ---
    do_refold = (not args.no_refold) and (predictor == 'alphafast')
    if not args.no_refold and predictor != 'alphafast':
        print("   ℹ️  Binder refold skipped (requires alphafast predictor).")

    if do_refold:
        top_n    = args.top or cfg.get('scoring', {}).get('refold_top_n', 10)
        DASH_DIR = os.path.join(WORK_DIR, cfg['outputs'].get('step5_dash', 'outputs/05_Dashboard'))
        print(f"   🔄 Binder refold: top {top_n} PASS candidates → {DASH_DIR}/refold_*/")
        df_ranked = run_binder_refold(df_ranked, cfg, WORK_DIR, top_n, DASH_DIR)

    out_ranked = os.path.join(SCORE_DIR, f"{project_name}_ranked.csv")
    df_ranked.to_csv(out_ranked, index=False)
    print(f"   💾 Ranked CSV:   {os.path.basename(out_ranked)}")

    print(f"✅ SCORING COMPLETE!")
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        print(f"⚠️  Some structures failed. Check '{log_file}' for details.")

if __name__ == "__main__":
    main()