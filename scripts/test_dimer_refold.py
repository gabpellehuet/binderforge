"""
==============================================================================
   🧬 SECONDARY VALIDATION: Binder Homodimer Refold (AlphaFast)
==============================================================================
   Standalone / ad hoc check, run against an already-completed 04_Scoring
   output. Re-predicts the top-N PASS binder sequences as a HOMODIMER
   (chain A = chain B = binder sequence, no target) with AlphaFast — the
   AF3-JSON equivalent of the ":"-joined ColabFold multimer FASTA trick:
   two separate "protein" entries, id ["A"] / id ["B"], same sequence +
   MSA (mirrors how 03_AlphaFast.py builds the binder/target heterodimer,
   just with the binder sequence on both sides instead of binder+target).

   Reports ipTM + model_confidence only (0.8*ipTM + 0.2*pTM, same formula
   as the main complex scoring in 04_Scoring.py). A binder that folds into
   a confident homodimer on its own is a red flag for a sticky/non-specific
   sequence rather than a genuine, target-specific interface.

   run_binder_dimer_refold() takes/returns a ranked df like run_binder_refold()
   in 04_Scoring.py, so it can be wired in there later as an additional
   phase (e.g. behind a --dimer-refold flag) once it's proven useful.

   Usage:
     python test_dimer_refold.py \
         --config <run>/outputs/.resolved_config.yaml \
         --ranked-csv <run>/outputs/04_Scoring/<project>_ranked.csv \
         [--top 10] [--out-dir <run>/outputs/dimer_refold]
==============================================================================
"""

import os
import glob
import json
import argparse
import subprocess

import yaml
import pandas as pd


# =================================================================
# HELPERS
# =================================================================

def _refold_base_name(design):
    """Strip trailing _model suffix used in CSV design names."""
    return design[:-6] if design.endswith('_model') else design


def _build_dimer_inputs(top_df, in_dir, af_in="/data/af_input"):
    """Write one AF3 JSON per design: a homodimer of the binder sequence
    (chain A = chain B), same MSA supplied to both copies."""
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

        chain = {
            "sequence": seq,
            "unpairedMsaPath": f"{af_in}/{binder_a3m}",
            "pairedMsaPath":   f"{af_in}/empty_paired.a3m",
            "templates": [],
        }
        af3_dict = {
            "name": name,
            "sequences": [
                {"protein": {"id": ["A"], **chain}},
                {"protein": {"id": ["B"], **chain}},
            ],
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


# =================================================================
# MAIN ENTRY POINT (reusable — same shape as run_binder_refold in 04_Scoring.py)
# =================================================================

def run_binder_dimer_refold(df_ranked: pd.DataFrame, cfg: dict, top_n: int, out_dir: str) -> pd.DataFrame:
    """
    Run AlphaFast on the top-N PASS binders as a homodimer (A=B=binder seq).
    Outputs land under out_dir:
      dimer_inputs/   — homodimer AF3 JSONs
      dimer_results/  — AlphaFast dimer predictions
    Adds dimer_iptm, dimer_ptm, dimer_model_confidence to df_ranked (NaN for
    rows outside the top-N).
    """
    in_dir      = os.path.join(out_dir, "dimer_inputs")
    results_dir = os.path.join(out_dir, "dimer_results")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    top = (df_ranked[df_ranked['status'] == 'PASS']
           .sort_values('BinderScore', ascending=False)
           .head(top_n))
    if top.empty:
        print("   ⚠️  No PASS candidates for dimer refold.")
        return df_ranked

    print(f"\n   🔬 Dimer Refold: top {len(top)} PASS candidates "
          f"(BinderScore {top['BinderScore'].min():.3f}–{top['BinderScore'].max():.3f})")

    n_in = _build_dimer_inputs(top, in_dir, cfg.get('tools', {}).get('af_input_dir', '/data/af_input'))
    print(f"   ✅ {n_in} homodimer JSON inputs written to {in_dir}")
    _run_alphafast_batch(cfg, in_dir, results_dir, label="homodimer refold")

    dimer_cols = {
        "dimer_iptm":             float('nan'),
        "dimer_ptm":              float('nan'),
        "dimer_model_confidence": float('nan'),
    }
    for col, default in dimer_cols.items():
        if col not in df_ranked.columns:
            df_ranked[col] = default

    for _, row in top.iterrows():
        name = _refold_base_name(row['design'])
        summ = _find_summary(results_dir, name)

        iptm = ptm = model_confidence = float('nan')
        if summ:
            with open(summ) as f:
                s = json.load(f)
            iptm = s.get('iptm', float('nan'))
            ptm  = s.get('ptm',  float('nan'))
            if pd.notna(iptm) and pd.notna(ptm):
                model_confidence = round(0.8 * iptm + 0.2 * ptm, 3)
        else:
            print(f"   ⚠️  {name}: dimer summary_confidences.json MISSING.")

        mask = df_ranked['design'] == row['design']
        df_ranked.loc[mask, 'dimer_iptm']             = iptm
        df_ranked.loc[mask, 'dimer_ptm']              = ptm
        df_ranked.loc[mask, 'dimer_model_confidence'] = model_confidence

    done = int(df_ranked['dimer_iptm'].notna().sum())
    print(f"   ✅ Dimer refold complete: {done} scored")
    return df_ranked


def main():
    ap = argparse.ArgumentParser(
        description="Secondary validation: refold top-N PASS binders as a homodimer "
                    "(chain A = chain B = binder sequence) with AlphaFast; reports "
                    "ipTM + model_confidence only.")
    ap.add_argument('--config', required=True,
                    help="Run's resolved config, e.g. <run>/outputs/.resolved_config.yaml "
                         "(needs step3_alphafast + tools.af_input_dir)")
    ap.add_argument('--ranked-csv', required=True,
                    help="Path to <project>_ranked.csv from a completed 04_Scoring run")
    ap.add_argument('--top', type=int, default=None,
                    help="Top-N PASS designs to test (default: scoring.refold_top_n in config, else 10)")
    ap.add_argument('--out-dir', default=None,
                    help="Where dimer_inputs/ and dimer_results/ land "
                         "(default: a dimer_refold/ folder next to --ranked-csv)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df_ranked = pd.read_csv(args.ranked_csv)
    top_n   = args.top or cfg.get('scoring', {}).get('refold_top_n', 10)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.ranked_csv)), "dimer_refold")

    df_ranked = run_binder_dimer_refold(df_ranked, cfg, top_n, out_dir)

    cols   = ['design', 'BinderScore', 'status', 'dimer_iptm', 'dimer_ptm', 'dimer_model_confidence']
    result = df_ranked[[c for c in cols if c in df_ranked.columns]]

    out_csv = os.path.join(out_dir, "dimer_refold_scores.csv")
    result.to_csv(out_csv, index=False)
    print(f"\n💾 Dimer refold scores: {out_csv}")
    print(result[result['dimer_iptm'].notna()]
          .sort_values('BinderScore', ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    main()

