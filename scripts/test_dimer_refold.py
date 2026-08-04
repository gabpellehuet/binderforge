"""
==============================================================================
   🧬 SECONDARY VALIDATION: Homomer Self-Refold (AlphaFast)
==============================================================================
   Two modes, sharing the same AlphaFast homooligomer machinery:

   1. REAL-RUN mode (--ranked-csv): re-predicts the top-N PASS binders from a
      completed 04_Scoring run as a HOMODIMER (chain A = chain B = binder
      sequence, no target) — the AF3-JSON equivalent of the ":"-joined
      ColabFold multimer FASTA trick: separate "protein" entries, one id per
      copy, same sequence + MSA (mirrors how 03_AlphaFast.py builds the
      binder/target heterodimer, just with the binder sequence on every
      chain instead of binder+target). A binder that folds into a confident
      homodimer on its own is a red flag for a sticky/non-specific sequence
      rather than a genuine, target-specific interface.

   2. BENCHMARK mode (--benchmark): scores a labeled dataset of sequences
      with KNOWN oligomeric state (monomer / dimer / both / other) as both a
      monomer ("foldability") and a homodimer, then reports which AlphaFast
      metric (ipTM, model_confidence, ranking_score, ...) — monomer or
      dimer-refold — best separates monomer from dimer ground truth (ROC AUC,
      no sklearn dependency). Use this to validate the homodimer-refold idea
      itself before trusting it on real designs with unknown ground truth.

   run_homomer_refold_on() is the shared core (n_copies=1 monomer / 2 dimer,
   no filtering) — run_binder_dimer_refold() (real-run) and run_benchmark()
   (benchmark) are both thin wrappers around it, so it can be wired into
   04_Scoring.py later as an additional phase once proven useful.

   Usage:
     # Real-run mode
     python test_dimer_refold.py \\
         --config <run>/outputs/.resolved_config.yaml \\
         --ranked-csv <run>/outputs/04_Scoring/<project>_ranked.csv \\
         [--top 10] [--out-dir <run>/outputs/dimer_refold]

     # Benchmark mode — dataset is a CSV (id,sequence,state) or a folder
     # (labels.csv + per-id .pdb/.cif/.fasta)
     python test_dimer_refold.py \\
         --config <run>/outputs/.resolved_config.yaml \\
         --benchmark path/to/dataset[.csv or folder] \\
         [--no-monomer] [--no-dimer] [--out-dir path/to/benchmark_out]
==============================================================================
"""

import os
import glob
import json
import argparse
import subprocess

import yaml
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, is_aa

AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


# =================================================================
# HELPERS
# =================================================================

def _refold_base_name(design):
    """Strip trailing _model suffix used in CSV design names (no-op for
    benchmark IDs, which don't carry this suffix — safe to apply either way)."""
    design = str(design)
    return design[:-6] if design.endswith('_model') else design


def _build_homomer_inputs(records, in_dir, af_in="/data/af_input", n_copies=2):
    """Write one AF3 JSON per (name, sequence): n_copies identical chains.
    n_copies=1 -> monomer ("foldability") refold; n_copies=2 -> homodimer
    self-association check. Same single-sequence pseudo-MSA for every copy —
    de novo/candidate sequences have no real evolutionary homologs to search
    for (same reasoning as the monomer refold in 04_Scoring.py)."""
    os.makedirs(in_dir, exist_ok=True)
    empty_a3m = os.path.join(in_dir, "empty_paired.a3m")
    with open(empty_a3m, 'w') as f:
        f.write("")

    chain_ids = ["A", "B", "C", "D"][:n_copies]
    n = 0
    for name, seq in records:
        seq = str(seq).strip()
        if not seq or seq.lower() == 'nan':
            print(f"   ⚠️  {name}: no sequence — skipped.")
            continue

        seq_a3m = f"{name}_seq.a3m"
        with open(os.path.join(in_dir, seq_a3m), 'w') as f:
            f.write(f">{name}_seq\n{seq}\n")

        chain = {
            "sequence": seq,
            "unpairedMsaPath": f"{af_in}/{seq_a3m}",
            "pairedMsaPath":   f"{af_in}/empty_paired.a3m",
            "templates": [],
        }
        af3_dict = {
            "name": name,
            "sequences": [{"protein": {"id": [cid], **chain}} for cid in chain_ids],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 3,
        }
        with open(os.path.join(in_dir, f"{name}.json"), 'w') as f:
            json.dump(af3_dict, f, indent=2)
        n += 1
    return n


def _run_alphafast_batch(cfg, in_dir, results_dir, label="batch"):
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
    print(f"\n🚀 LAUNCHING ALPHAFAST ({label})...")
    subprocess.run(cmd, cwd=af['alphafast_dir'], check=True)
    print("✨ Batch complete.\n")


def _find_summary(results_dir, name):
    p = os.path.join(results_dir, name, f"{name}_summary_confidences.json")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(results_dir, "**", f"*{name}*summary_confidences.json"), recursive=True)
    return sorted(hits)[0] if hits else None


def _parse_confidences(summary_path):
    """Pull iptm/ptm/ranking_score/fraction_disordered/has_clash out of an
    AF3 summary_confidences.json — same keys 04_Scoring.py's main complex
    scoring already reads. None if the file is missing."""
    if not summary_path or not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        s = json.load(f)
    iptm = s.get('iptm', float('nan'))
    ptm  = s.get('ptm',  float('nan'))
    model_confidence = float('nan')
    if pd.notna(iptm) and pd.notna(ptm):
        model_confidence = round(0.8 * iptm + 0.2 * ptm, 3)
    return {
        'iptm': iptm, 'ptm': ptm, 'model_confidence': model_confidence,
        'ranking_score': s.get('ranking_score', float('nan')),
        'fraction_disordered': s.get('fraction_disordered', float('nan')),
        'has_clash': bool(s.get('has_clash', False)),
    }


def _read_sequence_file(path):
    """One-letter sequence of the first protein chain in a .fasta/.fa/.pdb/.cif
    file (only the first chain — for a dataset entry that's itself deposited
    as a dimer, we still only want ONE protomer's sequence to test)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.fasta', '.fa'):
        with open(path) as f:
            lines = [l.strip() for l in f if not l.startswith('>')]
        seq = "".join(lines).strip()
        return seq or None

    parser = MMCIFParser(QUIET=True) if ext == '.cif' else PDBParser(QUIET=True)
    structure = parser.get_structure("x", path)
    for model in structure:
        for chain in model:
            seq = "".join(AA_MAP.get(r.get_resname(), 'X') for r in chain if is_aa(r))
            if seq:
                return seq
    return None


# =================================================================
# CORE: homooligomer self-refold on an arbitrary DataFrame (no filtering)
# =================================================================

def run_homomer_refold_on(df: pd.DataFrame, cfg: dict, out_dir: str, *,
                           id_col: str = 'design', seq_col: str = 'sequence',
                           n_copies: int = 2, prefix: str = 'dimer_') -> pd.DataFrame:
    """
    Run AlphaFast on EVERY row of df as an n_copies-mer homooligomer of
    df[seq_col] (n_copies=1: monomer/"foldability" self-refold; n_copies=2:
    homodimer self-association check). No PASS/top-N filtering — callers
    (run_binder_dimer_refold for a real run, run_benchmark for a labeled
    dataset) decide what subset of rows to pass in.
    Adds {prefix}iptm/ptm/model_confidence/ranking_score/frac_disordered/has_clash.
    """
    tag = {1: "monomer", 2: "homodimer"}.get(n_copies, f"{n_copies}-mer")
    in_dir      = os.path.join(out_dir, f"{prefix}inputs")
    results_dir = os.path.join(out_dir, f"{prefix}results")

    records = [(_refold_base_name(row[id_col]), row[seq_col]) for _, row in df.iterrows()]
    n_in = _build_homomer_inputs(records, in_dir,
                                  cfg.get('tools', {}).get('af_input_dir', '/data/af_input'),
                                  n_copies=n_copies)
    print(f"   ✅ {n_in} {tag} JSON inputs written to {in_dir}")
    _run_alphafast_batch(cfg, in_dir, results_dir, label=f"{tag} refold")

    defaults = {
        f"{prefix}iptm": float('nan'), f"{prefix}ptm": float('nan'),
        f"{prefix}model_confidence": float('nan'), f"{prefix}ranking_score": float('nan'),
        f"{prefix}frac_disordered": float('nan'), f"{prefix}has_clash": False,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for idx, row in df.iterrows():
        name = _refold_base_name(row[id_col])
        conf = _parse_confidences(_find_summary(results_dir, name))
        if conf is None:
            print(f"   ⚠️  {name}: {tag} summary_confidences.json MISSING.")
            continue
        df.at[idx, f"{prefix}iptm"]            = conf['iptm']
        df.at[idx, f"{prefix}ptm"]              = conf['ptm']
        df.at[idx, f"{prefix}model_confidence"] = conf['model_confidence']
        df.at[idx, f"{prefix}ranking_score"]    = conf['ranking_score']
        df.at[idx, f"{prefix}frac_disordered"]  = conf['fraction_disordered']
        df.at[idx, f"{prefix}has_clash"]        = conf['has_clash']

    done = int(df[f"{prefix}iptm"].notna().sum())
    print(f"   ✅ {tag.capitalize()} refold complete: {done}/{len(df)} scored")
    return df


# =================================================================
# REAL-RUN MODE — secondary validation of a completed 04_Scoring run
# =================================================================

def run_binder_dimer_refold(df_ranked: pd.DataFrame, cfg: dict, top_n: int, out_dir: str) -> pd.DataFrame:
    """
    Run AlphaFast on the top-N PASS binders as a homodimer (A=B=binder seq).
    Outputs land under out_dir: dimer_inputs/ (AF3 JSONs), dimer_results/
    (AlphaFast predictions). Adds dimer_iptm/ptm/model_confidence (+
    dimer_ranking_score/frac_disordered/has_clash) to df_ranked (NaN/False
    for rows outside the top-N).
    """
    top = (df_ranked[df_ranked['status'] == 'PASS']
           .sort_values('BinderScore', ascending=False)
           .head(top_n))
    if top.empty:
        print("   ⚠️  No PASS candidates for dimer refold.")
        return df_ranked

    print(f"\n   🔬 Dimer Refold: top {len(top)} PASS candidates "
          f"(BinderScore {top['BinderScore'].min():.3f}–{top['BinderScore'].max():.3f})")

    top = run_homomer_refold_on(top, cfg, out_dir, id_col='design', seq_col='sequence',
                                 n_copies=2, prefix='dimer_')

    dimer_defaults = {
        'dimer_iptm': float('nan'), 'dimer_ptm': float('nan'),
        'dimer_model_confidence': float('nan'), 'dimer_ranking_score': float('nan'),
        'dimer_frac_disordered': float('nan'), 'dimer_has_clash': False,
    }
    for col, default in dimer_defaults.items():
        if col not in df_ranked.columns:
            df_ranked[col] = default
    df_ranked.update(top[list(dimer_defaults)])
    return df_ranked


# =================================================================
# BENCHMARK MODE — labeled monomer/dimer/both/other dataset
# =================================================================

def load_benchmark_dataset(path: str) -> pd.DataFrame:
    """
    Load a labeled monomer/dimer/both/other dataset for metric validation.
    `path` may be:
      - a CSV file with columns  id, sequence, state
      - a folder containing a `labels.csv` (id, state[, sequence]) plus, if
        `sequence` isn't in labels.csv, one <id>.pdb / <id>.cif / <id>.fasta
        per row (first protein chain's sequence is extracted from structures).
    `state` is a free-text label; only "monomer", "dimer", and "both" are
    used for the binary AUC evaluation (see _binary_label) — anything else
    ("other", "aggregate", "unknown", ...) is reported separately.
    Returns a DataFrame with columns: id, sequence, state (str id, no NaN sequences).
    """
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        labels_csv = os.path.join(path, "labels.csv")
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"No labels.csv found in benchmark folder: {path}")
        df = pd.read_csv(labels_csv)
        if 'sequence' not in df.columns:
            seqs = []
            for _id in df['id']:
                seq = None
                for ext in ('.fasta', '.fa', '.pdb', '.cif'):
                    cand = os.path.join(path, f"{_id}{ext}")
                    if os.path.exists(cand):
                        seq = _read_sequence_file(cand)
                        break
                seqs.append(seq)
            df['sequence'] = seqs

    required = {'id', 'sequence', 'state'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark dataset missing required column(s): {sorted(missing)}")

    n_before = len(df)
    df = df.dropna(subset=['sequence']).copy()
    if len(df) < n_before:
        print(f"   ⚠️  {n_before - len(df)} row(s) dropped — no sequence found.")
    df['id'] = df['id'].astype(str)
    return df.reset_index(drop=True)


def run_benchmark(df: pd.DataFrame, cfg: dict, out_dir: str,
                   run_monomer: bool = True, run_dimer: bool = True) -> pd.DataFrame:
    """Run monomer ("foldability") and/or homodimer (self-association) refold
    on EVERY row of a labeled benchmark dataset — no filtering, every row is
    ground truth we want a metric reading for."""
    if run_monomer:
        print("\n   🔬 Benchmark: monomer (\"foldability\") refold — all rows")
        df = run_homomer_refold_on(df, cfg, out_dir, id_col='id', seq_col='sequence',
                                    n_copies=1, prefix='mono_')
    if run_dimer:
        print("\n   🔬 Benchmark: homodimer (self-association) refold — all rows")
        df = run_homomer_refold_on(df, cfg, out_dir, id_col='id', seq_col='sequence',
                                    n_copies=2, prefix='dimer_')
    return df


# =================================================================
# METRIC EVALUATION — which AlphaFast metric best predicts oligomeric state?
# =================================================================

def _binary_label(state):
    """monomer -> 0, dimer/both -> 1 (both = dimerization-competent, which is
    the practical QC question: would this design show any self-association
    tendency), anything else -> None (excluded from the binary comparison)."""
    s = str(state).strip().lower()
    if s == 'monomer':
        return 0
    if s in ('dimer', 'both'):
        return 1
    return None


def _auc(y_true, scores):
    """ROC AUC via the Mann-Whitney U / rank-sum equivalence — no sklearn
    dependency (not part of any conda env this pipeline installs). None if
    a class is empty after dropping NaN scores."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = ~np.isnan(s) & ~np.isnan(y)
    y, s = y[mask], s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(s).rank(method='average').values
    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_metrics_vs_state(df: pd.DataFrame, label_col: str = 'state',
                               metric_cols=None) -> pd.DataFrame:
    """
    For each candidate AlphaFast metric (monomer-refold "foldability" columns
    AND homodimer-refold columns, whichever were computed), report how well
    it separates dimer-like (dimer/both) from monomer ground truth by ROC AUC
    (0.5 = random, 1.0 = perfect; "higher metric => more dimer-like" — below
    0.5 means the relationship runs the other way, still informative, just
    inverted). 'other'/unrecognized labels are excluded from AUC but counted.
    Ranked by |AUC - 0.5| descending (biggest signal first, either direction).
    """
    if metric_cols is None:
        candidates = ['mono_iptm', 'mono_ptm', 'mono_model_confidence', 'mono_ranking_score',
                      'mono_frac_disordered', 'dimer_iptm', 'dimer_ptm', 'dimer_model_confidence',
                      'dimer_ranking_score', 'dimer_frac_disordered']
        metric_cols = [c for c in candidates if c in df.columns]

    y = df[label_col].map(_binary_label)
    n_other = int(y.isna().sum())

    rows = []
    for col in metric_cols:
        auc = _auc(y.values, df[col].values)
        if auc is None:
            continue
        n_pos = int(((y == 1) & df[col].notna()).sum())
        n_neg = int(((y == 0) & df[col].notna()).sum())
        rows.append({
            "metric": col, "n_monomer": n_neg, "n_dimer_or_both": n_pos,
            "auc": round(auc, 3), "abs_deviation_from_chance": round(abs(auc - 0.5), 3),
            "direction": "higher = more dimer-like" if auc >= 0.5 else "higher = more monomer-like",
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("abs_deviation_from_chance", ascending=False).reset_index(drop=True)

    print(f"\n📊 Metric-vs-state evaluation ({n_other} 'other'/unlabeled row(s) excluded from AUC):")
    if result.empty:
        print("   ⚠️  No metric had both classes present — nothing to rank "
              "(check --benchmark labels use 'monomer' and 'dimer'/'both').")
    else:
        print(result.to_string(index=False))
        best = result.iloc[0]
        print(f"\n🏆 Best predictor of oligomeric state: {best['metric']} "
              f"(AUC={best['auc']}, {best['direction']})")
    return result


# =================================================================
# CLI
# =================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Homomer self-refold with AlphaFast. Real-run mode: secondary "
                    "validation of a BinderForge run's top-N PASS binders as a "
                    "homodimer. Benchmark mode: score a labeled monomer/dimer/both/"
                    "other dataset and report which AlphaFast metric best predicts "
                    "oligomeric state.")
    ap.add_argument('--config', required=True,
                    help="Resolved config, e.g. <run>/outputs/.resolved_config.yaml "
                         "(needs step3_alphafast + tools.af_input_dir)")

    real_run = ap.add_argument_group("real-run mode")
    real_run.add_argument('--ranked-csv',
                    help="Path to <project>_ranked.csv from a completed 04_Scoring run")
    real_run.add_argument('--top', type=int, default=None,
                    help="Top-N PASS designs to test (default: scoring.refold_top_n in config, else 10)")

    bench = ap.add_argument_group("benchmark mode")
    bench.add_argument('--benchmark',
                    help="A labeled dataset: a CSV (id,sequence,state) or a folder "
                         "(labels.csv + per-id .pdb/.cif/.fasta files)")
    bench.add_argument('--no-monomer', action='store_true',
                    help="Skip the monomer (\"foldability\") refold in benchmark mode")
    bench.add_argument('--no-dimer', action='store_true',
                    help="Skip the homodimer refold in benchmark mode")

    ap.add_argument('--out-dir', default=None,
                    help="Where refold inputs/results land (default: next to "
                         "--ranked-csv or --benchmark)")
    args = ap.parse_args()

    if bool(args.ranked_csv) == bool(args.benchmark):
        ap.error("Pass exactly one of --ranked-csv (real-run mode) or --benchmark (benchmark mode).")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- Real-run mode ---
    if args.ranked_csv:
        df_ranked = pd.read_csv(args.ranked_csv)
        top_n   = args.top or cfg.get('scoring', {}).get('refold_top_n', 10)
        out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ranked_csv)), "dimer_refold")

        df_ranked = run_binder_dimer_refold(df_ranked, cfg, top_n, out_dir)

        cols = ['design', 'BinderScore', 'status', 'dimer_iptm', 'dimer_ptm',
                'dimer_model_confidence', 'dimer_ranking_score', 'dimer_frac_disordered',
                'dimer_has_clash']
        result = df_ranked[[c for c in cols if c in df_ranked.columns]]

        out_csv = os.path.join(out_dir, "dimer_refold_scores.csv")
        result.to_csv(out_csv, index=False)
        print(f"\n💾 Dimer refold scores: {out_csv}")
        print(result[result['dimer_iptm'].notna()]
              .sort_values('BinderScore', ascending=False)
              .to_string(index=False))
        return

    # --- Benchmark mode ---
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.benchmark)) if os.path.isfile(args.benchmark) else args.benchmark,
        "refold_benchmark"
    )
    df = load_benchmark_dataset(args.benchmark)
    print(f"   📋 Loaded {len(df)} labeled sequences ({df['state'].value_counts().to_dict()})")

    df = run_benchmark(df, cfg, out_dir, run_monomer=not args.no_monomer, run_dimer=not args.no_dimer)

    out_csv = os.path.join(out_dir, "benchmark_scores.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Benchmark scores: {out_csv}")

    ranking = evaluate_metrics_vs_state(df)
    ranking_csv = os.path.join(out_dir, "metric_ranking.csv")
    ranking.to_csv(ranking_csv, index=False)
    print(f"💾 Metric ranking: {ranking_csv}")


if __name__ == "__main__":
    main()
