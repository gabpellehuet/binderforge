import os
import sys
import yaml
import glob
import argparse
import subprocess
import re

# ==============================================================================
# ⚙️ USER CONFIGURATION
# ==============================================================================
# PyMOL executable — primary source is site.yaml `tools.pymol` (read from config);
# this is only the fallback when that's unset, resolving `pymol` from PATH.
PYMOL_PATH = "pymol"

# ==============================================================================
# 🧬 HOTSPOT PIPELINE - 3D VISUALIZATION
# ==============================================================================

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Config file '{config_path}' not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_prediction_file(pred_dir, design_name):
    # 1. Exact AlphaFast layout
    af_path = os.path.join(pred_dir, design_name, f"{design_name}_model.cif")
    if os.path.exists(af_path):
        return af_path
        
    # 2. Standard Boltz layout
    boltz_path = os.path.join(pred_dir, f"boltz_results_{design_name}", f"{design_name}.cif")
    if os.path.exists(boltz_path):
        return boltz_path
        
    # 3. Recursive fallback
    recursive_matches = glob.glob(os.path.join(pred_dir, "**", f"*{design_name}*.cif"), recursive=True)
    if recursive_matches:
        return sorted(recursive_matches)[0]

    return None

def main():
    parser = argparse.ArgumentParser(description="View a specific design in PyMOL (RFdiffusion vs Prediction).")
    parser.add_argument("design_name", type=str, help="Name of the base design (e.g., 0-NF-test_B0_S0)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    
    args = parser.parse_args()
    
    # --- NETTOYAGE AUTOMATIQUE DU NOM ---
    raw_name = args.design_name
    clean_name = re.sub(r'_seed-\d+_sample-\d+.*', '', raw_name)
    clean_name = clean_name.replace('_model', '').replace('.cif', '').replace('.pdb', '')
    design_name = clean_name

    print(f"🔍 Recherche des fichiers pour le design : {design_name}...")

    # 1. Load configuration
    config = load_config(args.config)
    work_dir = config.get('work_dir', './')

    # 2. Find RFdiffusion Backbone (PDB)
    rfd_dir = os.path.join(work_dir, config['outputs']['step1_rfd'])
    rfd_matches = glob.glob(os.path.join(rfd_dir, "**", f"{design_name}.pdb"), recursive=True)
    
    if not rfd_matches:
        print(f"❌ ERROR: RFdiffusion backbone '{design_name}.pdb' not found in {rfd_dir} or its subfolders.")
        sys.exit(1)
    rfd_path = rfd_matches[0]

    # 3. Find Prediction Output (CIF)
    predictor = config.get('predictor', 'boltz').lower()
    
    if predictor == 'alphafast':
        pred_dir = os.path.join(work_dir, config['outputs']['step3_alphafast'], "results")
        engine_name = "AlphaFold 3"
    else:
        pred_dir = os.path.join(work_dir, config['outputs']['step3_boltz'])
        engine_name = "Boltz-1"
        
    pred_path = find_prediction_file(pred_dir, design_name)

    if not pred_path:
        print(f"❌ ERROR: {engine_name} prediction file not found for '{design_name}' in {pred_dir}")
        sys.exit(1)

    print(f"✅ Found RFdiffusion backbone: {rfd_path}")
    print(f"✅ Found {engine_name} prediction: {pred_path}")
    print("🚀 Création du script PyMOL et lancement...")

# 4. Generate a PyMOL Macro Script (.pml)
    pml_script_path = "temp_view.pml"
    pml_content = f"""
# Load structures
load {rfd_path}, rfd_model
load {pred_path}, pred_model

# Clean up view
hide everything, all
show cartoon, all

# Set colors (Binder = A, Target = B for BOTH)
color white, rfd_model and chain B
color gray70, pred_model and chain B
color cyan, rfd_model and chain A
color tv_green, pred_model and chain A

# Align target (Anchor on Chain B)
align pred_model and chain B, rfd_model and chain B

# Center view
zoom all
bg_color black
"""
    with open(pml_script_path, "w") as f:
        f.write(pml_content)

    print("\n✨ Visualization Setup Complete ✨")
    print("------------------------------------------------")
    print("🎨 Color Legend:")
    print("   - Target (RFdiffusion): White")
    print("   - Target (Prediction) : Gray")
    print("   - Binder (RFdiffusion): Cyan")
    print("   - Binder (Prediction) : Green")
    print("------------------------------------------------")
    print("⚠️ Fermez la fenêtre PyMOL pour rendre la main au terminal.")

    # 5. Launch PyMOL using the hardcoded path
    try:
        # Prefer site.yaml tools.pymol; else the module fallback; else `pymol` on PATH.
        cfg_pymol = config.get('tools', {}).get('pymol')
        pymol_cmd = (cfg_pymol if (cfg_pymol and os.path.exists(cfg_pymol))
                     else (PYMOL_PATH if os.path.exists(PYMOL_PATH) else "pymol"))
        subprocess.run([pymol_cmd, pml_script_path])
    except FileNotFoundError:
        print(f"❌ ERROR: Exécutable PyMOL introuvable à : {pymol_cmd}")
        print("Vérifiez la variable PYMOL_PATH au début du script.")
    finally:
        # Clean up the temporary script
        if os.path.exists(pml_script_path):
            os.remove(pml_script_path)

if __name__ == "__main__":
    main()