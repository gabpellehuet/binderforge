"""
==============================================================================
   STEP 05: DASHBOARD GENERATION v5.0
==============================================================================
   Reads the pre-scored _ranked.csv written by 04_Scoring.py.
   Generates:
     - Interactive HTML (4 screening plots + score correlation heatmap)
     - List_1_AF3_Candidates.xlsx (all designs, PyMOL links, refold links)
     - PyMOL_Ready/  complex .pml sessions (generated here)
   Binder refold .pml sessions are written by 04_Scoring.py into
   outputs/06_BinderRefold/PyMOL_Ready/ and linked here.
==============================================================================
"""

import os
import sys
import yaml
import glob
import re
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from identity import assign_binder_ids
from contract import find_backbone, backbone_root

try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CONFIG_FILENAME = "config.yaml"

# ==============================================================================
# HEATMAP CONFIGURATION
# ==============================================================================

# Metrics included in the score correlation heatmap (order = display order)
HEATMAP_METRICS = [
    "BinderScore",
    "ipsae_min", "iptm", "model_confidence", "pDockQ",
    "dG", "sc_score", "dSASA", "unsat_hbonds", "clashes",
    "lrmsd", "rmsd_binder",
]

# Polarity: +1 = higher is better, -1 = lower is better (negated before correlation)
HEATMAP_POLARITY = {
    "BinderScore":  +1,
    "ipsae_min":    +1,
    "iptm":         +1,
    "model_confidence":      +1,
    "pDockQ":       +1,
    "dG":           -1,
    "sc_score":     +1,
    "dSASA":        +1,
    "unsat_hbonds": -1,
    "clashes":      -1,
    "lrmsd":        -1,
    "rmsd_binder":  -1,
}

HEATMAP_LABELS = {
    "BinderScore":  "BinderScore",
    "ipsae_min":    "ipSAE_min",
    "iptm":         "ipTM",
    "model_confidence":      "Model Conf.",
    "pDockQ":       "pDockQ",
    "dG":           "dG (neg.)",
    "sc_score":     "Shape Comp.",
    "dSASA":        "dSASA",
    "unsat_hbonds": "Unsat. HBonds (neg.)",
    "clashes":      "Clashes (neg.)",
    "lrmsd":        "L-RMSD (neg.)",
    "rmsd_binder":  "RMSD Binder (neg.)",
}

# ==============================================================================
# COLUMN ORDER — applied to both ranked CSV and Excel output
# ==============================================================================

from columns import COLUMN_ORDER, reorder_df as _reorder_df  # shared source of truth


# ==============================================================================
# SCORE CORRELATION HEATMAP
# ==============================================================================

def build_score_heatmap(df_pass: pd.DataFrame) -> "go.Figure | None":
    """
    Polarity-corrected Spearman correlation heatmap for PASS designs.
    Metrics with polarity -1 are negated so that higher always means better,
    making positive r mean the two metrics agree on binder quality.
    Upper triangle only; lower triangle masked to NaN (shown blank).
    Returns None if scipy is unavailable or fewer than 3 metrics are present.
    """
    if not HAS_SCIPY or df_pass.empty:
        return None

    available = [m for m in HEATMAP_METRICS if m in df_pass.columns and df_pass[m].notna().any()]
    if len(available) < 3:
        return None

    # Polarity correction: flip "lower is better" metrics so all point the same way
    corrected = pd.DataFrame({
        m: df_pass[m].astype(float) * HEATMAP_POLARITY.get(m, 1)
        for m in available
    })

    n = len(available)
    corr_m = np.eye(n)
    pval_m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            xi = corrected.iloc[:, i].dropna()
            xj = corrected.iloc[:, j].dropna()
            common = xi.index.intersection(xj.index)
            if len(common) >= 5:
                r, p = _scipy_stats.spearmanr(xi.loc[common], xj.loc[common])
                corr_m[i, j] = corr_m[j, i] = float(r)
                pval_m[i, j] = pval_m[j, i] = float(p)
            else:
                corr_m[i, j] = corr_m[j, i] = np.nan
                pval_m[i, j] = pval_m[j, i] = 1.0

    labels = [HEATMAP_LABELS.get(m, m) for m in available]

    # Mask lower triangle
    z = corr_m.copy().astype(float)
    for i in range(n):
        for j in range(i):
            z[i, j] = np.nan

    # Build per-cell hover text
    hover = []
    for i in range(n):
        row_text = []
        for j in range(n):
            r = corr_m[i, j]
            p = pval_m[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if np.isnan(r):
                row_text.append(f"<b>{labels[i]}</b> vs <b>{labels[j]}</b><br>r = n/a")
            else:
                row_text.append(
                    f"<b>{labels[i]}</b> vs <b>{labels[j]}</b><br>"
                    f"Spearman r = {r:.3f}{sig}<br>"
                    f"p = {p:.4f} | n = {len(df_pass)}"
                )
        hover.append(row_text)

    cell_size = 70
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale="RdBu_r",
        zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title="Spearman r", thickness=14, len=0.7),
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"Score Correlation Heatmap — Polarity-Corrected Spearman  "
                f"(n = {len(df_pass)} PASS designs)<br>"
                "<sup>Metrics flipped so higher = better | Positive r = metrics agree on quality | "
                "Hover for r and p-value | * p&lt;0.05  ** p&lt;0.01  *** p&lt;0.001</sup>"
            ),
            font=dict(size=14),
        ),
        width=max(600, n * cell_size + 120),
        height=max(520, n * cell_size + 100),
        xaxis=dict(tickangle=-40, side="bottom", tickfont=dict(size=12)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        font=dict(size=12),
        template="plotly_white",
        margin=dict(l=30, r=30, t=100, b=30),
    )
    return fig


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_config(path):
    if not os.path.exists(path):
        path_up = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
        if os.path.exists(path_up):
            return yaml.safe_load(open(path_up))
        print(f"❌ Error: Config '{path}' not found.")
        sys.exit(1)
    return yaml.safe_load(open(path))


def find_prediction_file(pred_dir, design_name):
    af_path = os.path.join(pred_dir, design_name, f"{design_name}_model.cif")
    if os.path.exists(af_path):
        return af_path
    boltz_path = os.path.join(pred_dir, f"boltz_results_{design_name}", f"{design_name}.cif")
    if os.path.exists(boltz_path):
        return boltz_path
    recursive_matches = glob.glob(os.path.join(pred_dir, "**", f"*{design_name}*.cif"), recursive=True)
    if recursive_matches:
        return sorted(recursive_matches)[0]
    return None


def create_pymol_session(design_raw, work_dir, cfg, output_dir, tier=""):
    """Generates a .pml file for instant viewing in PyMOL."""
    clean_name  = re.sub(r'_seed-\d+_sample-\d+.*', '', design_raw)
    design_name = clean_name.replace('_model', '').replace('.cif', '').replace('.pdb', '')

    # Backbone location + resolution: centralised in contract.py (shared with scoring).
    backbone_dir = backbone_root(work_dir, cfg)
    hit = find_backbone(backbone_dir, design_name)
    if not hit:
        return None
    rfd_path = os.path.abspath(hit)

    predictor = cfg.get('predictor', 'boltz').lower()
    if predictor == 'alphafast':
        pred_dir = os.path.join(work_dir, cfg['outputs']['step3_alphafast'], "results")
    else:
        pred_dir = os.path.join(work_dir, cfg['outputs']['step3_boltz'])
    pred_path = find_prediction_file(pred_dir, design_name)
    if not pred_path:
        return None
    pred_path = os.path.abspath(pred_path)

    tier_label = f"_{tier}" if tier else ""
    pml_path   = os.path.join(output_dir, f"View_{design_name}{tier_label}.pml")
    pml_content = f"""# {design_name}  |  {tier if tier else 'PASS'}
load {rfd_path}, rfd_model
load {pred_path}, pred_model

hide everything, all
show cartoon, all

# Binder = chain A  |  Target = chain B
color white,    rfd_model  and chain B
color gray70,   pred_model and chain B
color cyan,     rfd_model  and chain A
color tv_green, pred_model and chain A

align pred_model and chain B, rfd_model and chain B
zoom all
bg_color black
"""
    with open(pml_path, "w") as f:
        f.write(pml_content)
    return os.path.abspath(pml_path)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_FILENAME)
    args = parser.parse_args()

    cfg          = load_config(args.config)
    WORK_DIR     = cfg.get('work_dir', './')
    PROJECT_NAME = cfg.get('project_name', 'Binder_Design')

    cutoffs   = cfg.get('cutoffs', {})
    CUT_RMSD  = cutoffs.get('rmsd', 1.2)
    CUT_DSASA = cutoffs.get('dsasa', 1000)

    SCORE_DIR = os.path.join(WORK_DIR, cfg.get('outputs', {}).get('step4_scoring', 'outputs/04_Scoring'))
    DASH_DIR  = os.path.join(WORK_DIR, cfg.get('outputs', {}).get('step5_dash', 'outputs/05_Dashboard'))
    PYMOL_DIR = os.path.join(DASH_DIR, "PyMOL_Ready")
    os.makedirs(DASH_DIR,  exist_ok=True)
    os.makedirs(PYMOL_DIR, exist_ok=True)

    INPUT_CSV   = os.path.join(SCORE_DIR, f"{PROJECT_NAME}_ranked.csv")
    OUTPUT_HTML = os.path.join(DASH_DIR,  f"{PROJECT_NAME}_dashboard.html")
    OUTPUT_XLSX = os.path.join(DASH_DIR,  "List_1_AF3_Candidates.xlsx")
    # Refold .pml files written by 04_Scoring.py land in the same PyMOL_Ready dir
    REFOLD_PYMOL = PYMOL_DIR

    print("------------------------------------------------")
    print("📊 STEP 5: Dashboard (v5.0)")
    print("------------------------------------------------")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Run Step 4 (scoring) first.")
        sys.exit(1)

    # _ranked.csv is already scored and ranked by 04_Scoring.py
    df_ranked = pd.read_csv(INPUT_CSV)

    # binder_id (traceable identity) — normally written by 04_Scoring.py; recompute
    # here if reading an older ranked CSV that predates the identity column.
    if 'binder_id' not in df_ranked.columns:
        df_ranked = assign_binder_ids(df_ranked, cfg)

    if 'af3_ranking_score' in df_ranked.columns:
        score_col = 'af3_ranking_score';    pred_name = 'AF3'
    elif 'boltz_confidence_score' in df_ranked.columns:
        score_col = 'boltz_confidence_score'; pred_name = 'Boltz'
    else:
        score_col = None; pred_name = 'AI'

    # Safety defaults for plot columns that may be absent
    for col, val in {'rmsd_complex': 99.9, 'iptm': 0.0, 'dSASA': 0.0,
                     'pDockQ': 0.0, 'model_confidence': 0.0, 'dG': 0.0}.items():
        if col not in df_ranked.columns:
            df_ranked[col] = val
    if score_col and score_col not in df_ranked.columns:
        df_ranked[score_col] = 0.0

    n_pass = int((df_ranked['status'] == 'PASS').sum())
    n_elim = int((df_ranked['status'] == 'ELIMINATED').sum())
    print(f"   📊 {n_pass} PASS | {n_elim} ELIMINATED  (of {len(df_ranked)} total)")
    tier_counts = {}
    if n_pass > 0:
        tier_counts = df_ranked[df_ranked['status'] == 'PASS']['candidate_tier'].value_counts().to_dict()
        for tier in ['Tier1', 'Tier2', 'Tier3']:
            print(f"      {tier}: {tier_counts.get(tier, 0)}")

    df_pass_sorted = df_ranked[df_ranked['status'] == 'PASS']

    # df_clean for plots (exclude extreme RMSD outliers)
    df_clean = df_ranked[df_ranked['rmsd_complex'] < 50].copy()

    # PyMOL sessions: normally one per PASS design. In scoring.pass_all test mode (everything
    # PASSes) cap to the top-N by BinderScore so a stray long run can't emit thousands.
    pass_all = bool(cfg.get('scoring', {}).get('pass_all', False))
    sessions_df = df_pass_sorted
    if pass_all:
        limit = int(cfg.get('scoring', {}).get('pass_all_limit', 10))
        if 'BinderScore' in sessions_df.columns:
            sessions_df = sessions_df.sort_values('BinderScore', ascending=False)
        sessions_df = sessions_df.head(limit)

    # --- COMPLEX PyMOL SESSIONS ---
    if pass_all:
        print(f"   🎨 pass_all: PyMOL sessions for top {len(sessions_df)} of {len(df_pass_sorted)} "
              f"designs (cap {limit})...")
    else:
        print("   🎨 Generating complex PyMOL sessions for PASS candidates...")
    pymol_link_map   = {}
    terminal_cmd_map = {}

    for _, row in sessions_df.iterrows():
        design   = row['design']
        tier     = row.get('candidate_tier', '')
        pml_path = create_pymol_session(design, WORK_DIR, cfg, PYMOL_DIR, tier=tier)
        pymol_link_map[design]   = f'=HYPERLINK("{pml_path}", "Open PyMOL")' if pml_path else "Files missing"
        terminal_cmd_map[design] = f"python scripts/view_design.py {design}"

    # --- REFOLD PyMOL LINKS (from 04_Scoring.py output) ---
    refold_link_map = {}
    if os.path.isdir(REFOLD_PYMOL):
        for pml in glob.glob(os.path.join(REFOLD_PYMOL, "Refold_*.pml")):
            basename = os.path.basename(pml)
            # "Refold_{name}_{tier}.pml" or "Refold_{name}.pml"
            inner = basename[len("Refold_"):-len(".pml")]
            name  = re.sub(r'_(Tier[123])$', '', inner)
            refold_link_map[name] = f'=HYPERLINK("{os.path.abspath(pml)}", "Refold PyMOL")'

    # --- EXCEL OUTPUT ---
    df_ranked['pymol_link']   = df_ranked['design'].map(pymol_link_map).fillna("")
    df_ranked['terminal_cmd'] = df_ranked['design'].map(terminal_cmd_map).fillna("")
    # Map refold links using the base name (strip _model suffix)
    df_ranked['refold_link']  = df_ranked['design'].apply(
        lambda d: refold_link_map.get(d[:-6] if d.endswith('_model') else d, "")
    )
    front_extra = ['pymol_link', 'refold_link', 'terminal_cmd']
    df_output   = _reorder_df(df_ranked, front_extra=front_extra)

    try:
        df_output.to_excel(OUTPUT_XLSX, index=False, engine='openpyxl')
        excel_status = ".xlsx saved"
    except ImportError:
        excel_status = "(pip install openpyxl for Excel output)"

    refold_note = f" | {len(refold_link_map)} refold links" if refold_link_map else ""
    print(f"   🎯 {n_pass} PASS / {len(df_ranked)} total{refold_note} → {os.path.basename(OUTPUT_XLSX)} {excel_status}")

    # Counts for subplot annotations
    dsasa_mask       = (df_clean['rmsd_complex'] < CUT_RMSD) & (df_clean['dSASA'] > CUT_DSASA)
    num_dsasa        = int(dsasa_mask.sum())
    PDOCK_CUT, MPDOCK_CUT = 0.6, 0.6
    num_elite        = int(((df_clean['pDockQ'] >= PDOCK_CUT) & (df_clean['model_confidence'] >= MPDOCK_CUT)).sum())
    FUNNEL_RMSD_CUT  = 5.0

    # Data-driven dG Y axis — Rosetta dG can reach 10k-100k REU for failed structures,
    # which completely buries the meaningful negative-dG range on an auto-scaled axis.
    # Use p2 → min(p95, 50 REU) with small buffers; hard-cap at ±150.
    _dg = df_clean['dG'].dropna()
    dg_y_min = float(np.floor(max(_dg.quantile(0.02) * 1.10, -150.0)))
    dg_y_max = float(np.ceil(min(_dg.quantile(0.95), 50.0)))
    dg_y_max = max(dg_y_max, 20.0)   # always include zero line + buffer

    # Funnel tip = negative binding energy (dG < 0) AND low RMSD — genuine binders
    num_funnel = int(((df_clean['rmsd_complex'] < FUNNEL_RMSD_CUT) & (_dg < 0.0)).sum())

    tier1_n = tier_counts.get('Tier1', 0)
    tier2_n = tier_counts.get('Tier2', 0)
    tier3_n = tier_counts.get('Tier3', 0)

    # =================================================================
    # PLOTTING
    # =================================================================
    print("   Generating visualizations...")

    titles = (
        f"<b>1. Final Candidate Selection</b><br>"
        f"<span style='font-size:12px;color:gray'>RMSD vs {pred_name} Conf | "
        f"{n_pass} PASS  (T1:{tier1_n}  T2:{tier2_n}  T3:{tier3_n})</span>",
        f"<b>2. Interface Surface Area</b><br>"
        f"<span style='font-size:12px;color:gray'>RMSD vs dSASA | {num_dsasa} pass dSASA cut</span>",
        f"<b>3. Physics vs AI Confidence</b><br>"
        f"<span style='font-size:12px;color:gray'>pDockQ vs Model Conf. | {num_elite} in Elite Zone</span>",
        f"<b>4. The Master Funnel</b><br>"
        f"<span style='font-size:12px;color:gray'>Binding Energy vs RMSD | "
        f"{num_funnel} in tip (dG&lt;0, RMSD&lt;{FUNNEL_RMSD_CUT}Å)</span>",
    )

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles,
                        vertical_spacing=0.15, horizontal_spacing=0.1)

    # --- PLOT 1: RMSD vs AI Conf — ELIMINATED (gray) + PASS (colored by BinderScore) ---
    df_elim_plot = df_clean[df_clean['status'] == 'ELIMINATED']
    df_pass_plot = df_clean[df_clean['status'] == 'PASS']

    if not df_elim_plot.empty and score_col:
        fig.add_trace(go.Scatter(
            x=df_elim_plot['rmsd_complex'], y=df_elim_plot[score_col], mode='markers',
            marker=dict(size=6, color='lightgray', opacity=0.5),
            text=df_elim_plot['design'],
            hovertemplate=(
                "<b>%{text}</b><br>RMSD: %{x:.2f} Å<br>Conf: %{y:.2f}<br>"
                "<i>ELIMINATED: %{customdata}</i><extra></extra>"
            ),
            customdata=df_elim_plot['elimination_reason'],
            showlegend=False,
        ), row=1, col=1)

    if not df_pass_plot.empty and score_col:
        bs_vals = df_pass_plot['BinderScore'] if 'BinderScore' in df_pass_plot.columns else df_pass_plot['iptm']
        fig.add_trace(go.Scatter(
            x=df_pass_plot['rmsd_complex'], y=df_pass_plot[score_col], mode='markers',
            marker=dict(
                size=9, color=bs_vals, colorscale='Viridis', showscale=True, opacity=0.9,
                colorbar=dict(title="BinderScore", x=0.45, y=0.8, len=0.35, thickness=10),
            ),
            text=df_pass_plot['design'],
            hovertemplate=(
                "<b>%{text}</b><br>RMSD: %{x:.2f} Å<br>" + pred_name + " Conf: %{y:.2f}<br>"
                "BinderScore: %{marker.color:.3f}<br>Tier: %{customdata}<extra></extra>"
            ),
            customdata=df_pass_plot['candidate_tier'],
            showlegend=False,
        ), row=1, col=1)

    fig.add_annotation(
        x=0.5, y=0.97, xref='x domain', yref='y domain',
        text=f"PASS: {n_pass}  (T1:{tier1_n}  T2:{tier2_n}  T3:{tier3_n})",
        showarrow=False, font=dict(color="darkgreen", size=10), row=1, col=1,
    )

    # --- PLOT 2: RMSD vs dSASA ---
    if score_col:
        fig.add_trace(go.Scatter(
            x=df_clean['rmsd_complex'], y=df_clean['dSASA'], mode='markers',
            marker=dict(
                size=8, color=df_clean[score_col], colorscale='Plasma', showscale=True, opacity=0.8,
                colorbar=dict(title=f"{pred_name} Conf", x=1.0, y=0.8, len=0.35, thickness=10),
            ),
            text=df_clean['design'],
            hovertemplate=(
                "<b>%{text}</b><br>RMSD: %{x:.2f} Å<br>dSASA: %{y:.0f}<br>"
                + pred_name + " Conf: %{marker.color:.2f}<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=2)

    max_dsasa = df_clean['dSASA'].max() if not df_clean.empty else 2000
    fig.add_shape(type="rect", x0=0, y0=CUT_DSASA, x1=CUT_RMSD, y1=max_dsasa * 1.1,
                  line=dict(color="green", width=2, dash="dash"), fillcolor="rgba(0,255,0,0.1)", row=1, col=2)
    fig.add_annotation(x=CUT_RMSD / 2, y=CUT_DSASA + 150, text=f"Ideal Interface<br>({num_dsasa})",
                       showarrow=False, font=dict(color="green", size=10), row=1, col=2)

    # --- PLOT 3: pDockQ vs model_confidence ---
    fig.add_trace(go.Scatter(
        x=df_clean['pDockQ'], y=df_clean['model_confidence'], mode='markers',
        marker=dict(
            size=9, color=df_clean['rmsd_complex'], colorscale='Turbo_r', showscale=True, opacity=0.8,
            colorbar=dict(title="Global RMSD", x=0.45, y=0.2, len=0.35, thickness=10),
        ),
        text=df_clean['design'],
        hovertemplate=(
            "<b>%{text}</b><br>pDockQ: %{x:.2f}<br>Model Conf.: %{y:.2f}<br>"
            "RMSD: %{marker.color:.2f} Å<extra></extra>"
        ),
        showlegend=False,
    ), row=2, col=1)

    fig.add_shape(type="rect", x0=PDOCK_CUT, y0=MPDOCK_CUT, x1=1.0, y1=1.0,
                  line=dict(color="purple", width=2, dash="dash"), fillcolor="rgba(128,0,128,0.1)", row=2, col=1)
    fig.add_annotation(x=PDOCK_CUT + 0.15, y=MPDOCK_CUT + 0.05, text=f"Elite Zone<br>({num_elite})",
                       showarrow=False, font=dict(color="purple", size=10), row=2, col=1)

    # --- PLOT 4: Master Funnel ---
    if score_col:
        fig.add_trace(go.Scatter(
            x=df_clean['rmsd_complex'], y=df_clean['dG'], mode='markers',
            marker=dict(
                size=8, color=df_clean[score_col], colorscale='Magma', showscale=True,
                colorbar=dict(title=f"{pred_name} Conf", x=1.0, y=0.2, len=0.35, thickness=10),
            ),
            text=df_clean['design'],
            hovertemplate="<b>%{text}</b><br>RMSD: %{x:.2f} Å<br>dG: %{y:.2f} REU<extra></extra>",
            showlegend=False,
        ), row=2, col=2)

    # Zero-energy reference line
    fig.add_hline(y=0, line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot"), row=2, col=2)

    # Funnel tip highlight: negative dG (real binding) + low RMSD
    fig.add_shape(type="rect",
                  x0=0, y0=dg_y_min, x1=FUNNEL_RMSD_CUT, y1=0,
                  line=dict(color="orange", width=2, dash="dot"),
                  fillcolor="rgba(255,165,0,0.08)", row=2, col=2)
    tip_label_y = dg_y_min + (0 - dg_y_min) * 0.15   # 15% from bottom
    fig.add_annotation(
        x=FUNNEL_RMSD_CUT / 2, y=tip_label_y,
        text=f"Funnel Tip<br>{num_funnel} (dG<0, RMSD<{FUNNEL_RMSD_CUT}Å)",
        showarrow=False, font=dict(color="darkorange", size=10), row=2, col=2,
    )
    fig.add_annotation(
        x=FUNNEL_RMSD_CUT * 1.3, y=dg_y_min + (dg_y_max - dg_y_min) * 0.08,
        text=(
            "<b>How to read the funnel:</b><br>"
            "Binders converge to low RMSD<br>"
            "and negative dG at the tip.<br>"
            f"Y axis: p2–p95 of dG<br>"
            f"(outliers up to {int(_dg.max()):,} REU hidden)"
        ),
        showarrow=False, align="left", bordercolor="#ccc", borderwidth=1,
        borderpad=5, bgcolor="rgba(255,255,255,0.9)", font=dict(size=9), row=2, col=2,
    )

    # --- LAYOUT ---
    fig.update_layout(
        title_text=f"<b>{PROJECT_NAME} Pre-AF3 Selection Dashboard ({pred_name} Engine)</b>",
        height=1000, width=1400, template="plotly_white", showlegend=False,
    )
    fig.update_xaxes(title_text="Complex RMSD (Å)", row=1, col=1, range=[0, 5])
    fig.update_yaxes(title_text=f"{pred_name} Confidence", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Complex RMSD (Å)", row=1, col=2, range=[0, 5])
    fig.update_yaxes(title_text="Buried Area (dSASA)", row=1, col=2)
    fig.update_xaxes(title_text="Physics Check (pDockQ)", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="AI Check (Model Conf.)", row=2, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Complex RMSD (Å)", row=2, col=2)
    fig.update_yaxes(title_text="Binding Energy (dG, REU)", row=2, col=2,
                     range=[dg_y_min, dg_y_max])

    # --- SCORE CORRELATION HEATMAP ---
    heatmap_fig  = build_score_heatmap(df_pass_sorted)
    heatmap_div  = (
        heatmap_fig.to_html(full_html=False, include_plotlyjs=False)
        if heatmap_fig is not None
        else "<p style='color:#888'>Score correlation heatmap unavailable — install scipy: pip install scipy</p>"
    )
    if heatmap_fig is None and HAS_SCIPY:
        heatmap_div = "<p style='color:#888'>Score correlation heatmap skipped — fewer than 3 metrics available.</p>"

    # --- COMBINED HTML (screening plots + heatmap in one file) ---
    screening_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
    html_content  = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{PROJECT_NAME} Dashboard</title>
  <style>
    body  {{ font-family: Arial, sans-serif; background:#f5f5f5; margin:24px; color:#222; }}
    h1    {{ margin-bottom:6px; }}
    .sub  {{ color:#666; font-size:14px; margin-bottom:20px; }}
    .card {{ background:white; border-radius:10px; padding:24px 28px; margin-bottom:28px;
             box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
    h2    {{ margin-top:0; color:#444; border-bottom:2px solid #eee; padding-bottom:8px; }}
  </style>
</head>
<body>
  <h1>🧬 {PROJECT_NAME}</h1>
  <p class="sub">Pre-AF3 Selection Dashboard — {pred_name} Engine &nbsp;|&nbsp;
     {n_pass} PASS &nbsp;(T1:{tier1_n} T2:{tier2_n} T3:{tier3_n}) &nbsp;|&nbsp;
     {n_elim} ELIMINATED</p>

  <div class="card">
    <h2>Screening Plots</h2>
    {screening_div}
  </div>

  <div class="card">
    <h2>Score Correlation Heatmap</h2>
    <p style="font-size:13px;color:#666;margin-top:-4px">
      Polarity-corrected Spearman r — all metrics flipped so higher = better binder.
      <b>Positive r</b>: two metrics agree on quality.
      <b>Near zero</b>: independent signals.
      <b>Negative r</b>: conflicting signals.
    </p>
    {heatmap_div}
  </div>
</body>
</html>"""

    with open(OUTPUT_HTML, 'w') as f:
        f.write(html_content)
    print(f"   ✅ Dashboard generated: {os.path.basename(OUTPUT_HTML)}")
    print("------------------------------------------------")


if __name__ == "__main__":
    main()
