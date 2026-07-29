"""
==============================================================================
   STEP 03: AlphaFast (Native Multi-GPU Batch + Clean MSA)
==============================================================================
"""

import os
import sys
import time
import threading
import yaml
import glob
import json
import subprocess

CONFIG_PATH = os.environ.get("PIPELINE_CONFIG", "config.yaml")

def load_config():
    with open(CONFIG_PATH, 'r') as f: return yaml.safe_load(f)

def clean_and_extract_msa(msa_abs_path, dest_msa_path):
    print(f"🎯 Nettoyage et extraction depuis : {os.path.basename(msa_abs_path)}")
    with open(msa_abs_path, 'r', encoding='utf-8', errors='ignore') as f_in:
        raw_text = f_in.read().replace('\x00', '')

    lines = raw_text.splitlines()
    valid_lines = []
    query_length = 0
    target_ref_seq = ""

    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith(">"):
            valid_lines.append(line)
        else:
            seq = "".join([c.upper() for c in line if c.isalpha() or c == "-"])

            if query_length == 0:
                query_length = len(seq)
                target_ref_seq = seq.replace("-", "")

            if len(seq) == query_length:
                valid_lines.append(seq)
            else:
                if valid_lines: valid_lines.pop()

    with open(dest_msa_path, 'w', newline='\n') as f_out:
        f_out.write("\n".join(valid_lines) + "\n")

    print(f"   -> Séquence Target prête (Longueur: {len(target_ref_seq)}).")
    return target_ref_seq

def main():
    cfg = load_config()
    WORK_DIR = cfg['work_dir']

    MPNN_DIR       = os.path.join(WORK_DIR, cfg['outputs']['step2_mpnn'])
    AF_OUT_DIR     = os.path.join(WORK_DIR, cfg['outputs']['step3_alphafast'])
    AF_IN_DIR      = os.path.join(AF_OUT_DIR, "inputs")
    AF_RESULTS_DIR = os.path.join(AF_OUT_DIR, "results")

    os.makedirs(AF_IN_DIR, exist_ok=True)
    os.makedirs(AF_RESULTS_DIR, exist_ok=True)

    # 0. CLEAN AND EXTRACT TARGET MSA
    msa_path = cfg.get('target', {}).get('msa', '')
    if not os.path.isfile(msa_path):
        print("❌ CRITICAL ERROR: Target MSA is missing.")
        sys.exit(1)

    dest_msa_path  = os.path.join(AF_IN_DIR, "target_msa.a3m")
    target_ref_seq = clean_and_extract_msa(msa_path, dest_msa_path)

    empty_a3m_name = "empty_paired.a3m"
    with open(os.path.join(AF_IN_DIR, empty_a3m_name), 'w') as f:
        f.write("")

    # Path AlphaFast/AF3 sees INSIDE its container (AF_IN_DIR is mounted here).
    af_in = cfg.get('tools', {}).get('af_input_dir', '/data/af_input')

    # 1. BUILD AF3 INPUTS FROM STEP-3 YAML SELECTIONS
    # Step 3 already ran get_top_n_seqs() and stored the best binder sequence
    # per design in the Boltz YAMLs. !! Boltz YAMLs are just the name of the format, inherited from the Boltz pipeline
    yaml_files = glob.glob(os.path.join(MPNN_DIR, "**", "*.yaml"), recursive=True)
    print(f"\n🔄 Preparing {len(yaml_files)} AF3 JSON inputs from Step-3 selections...")

    json_names = set()
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            boltz_data = yaml.safe_load(f)

        binder_seq = boltz_data['sequences'][0]['protein']['sequence']

        # Derive JSON name from YAML stem: strip _seq_1 suffix for the primary
        # sequence so naming matches the original FASTA-based convention.
        yaml_stem = os.path.splitext(os.path.basename(yaml_file))[0]
        json_name = yaml_stem[:-6] if yaml_stem.endswith('_seq_1') else yaml_stem

        binder_a3m_name = f"{json_name}_binder.a3m"
        with open(os.path.join(AF_IN_DIR, binder_a3m_name), 'w') as f:
            f.write(f">{json_name}_binder\n{binder_seq}\n")

        af3_dict = {
            "name": json_name,
            "sequences": [
                {"protein": {
                    "id": ["A"], "sequence": binder_seq,
                    "unpairedMsaPath": f"{af_in}/{binder_a3m_name}",
                    "pairedMsaPath":   f"{af_in}/{empty_a3m_name}",
                    "templates": []
                }},
                {"protein": {
                    "id": ["B"], "sequence": target_ref_seq,
                    "unpairedMsaPath": f"{af_in}/target_msa.a3m",
                    "pairedMsaPath":   f"{af_in}/{empty_a3m_name}",
                    "templates": []
                }}
            ],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 3
        }

        with open(os.path.join(AF_IN_DIR, f"{json_name}.json"), 'w') as f:
            json.dump(af3_dict, f, indent=2)
        json_names.add(json_name)

    json_count = len(json_names)
    print(f"✅ {json_count} JSON inputs ready.")

    # 2. RUN ALPHAFAST NATIVE BATCH
    af_cfg      = cfg['step3_alphafast']
    script_path = os.path.join(af_cfg['alphafast_dir'], "scripts/run_alphafast.sh")

    cmd = [
        script_path,
        "--input_dir",   AF_IN_DIR,
        "--output_dir",  AF_RESULTS_DIR,
        "--db_dir",      af_cfg['db_dir'],
        "--weights_dir", af_cfg['weights_dir'],
        "--num_gpus",    str(af_cfg['num_gpus']),
        "--backend",     "docker",
        "--container",   "romerolabduke/alphafast:latest"
    ]

    print(f"\n🚀 LAUNCHING ALPHAFAST NATIVE BATCH  ({json_count} designs)")
    print(f"   Progress tracked by counting completed *_model.cif files.\n")

    # --- Progress monitor (background thread, polls every 10 s) ---
    stop_event = threading.Event()

    def _monitor():
        start = time.time()
        while not stop_event.wait(10):
            # Count unique designs finished, not raw CIF files.
            # AF3 writes multiple CIFs per design (seeds, ranked copies, etc.),
            # so naively counting files causes percentage > 100%.
            cif_files = glob.glob(
                os.path.join(AF_RESULTS_DIR, "**", "*_model.cif"), recursive=True)
            done_names = set()
            for f in cif_files:
                basename = os.path.basename(f)
                for name in json_names:
                    if basename.startswith(name):
                        done_names.add(name)
                        break
            done = len(done_names)
            pct  = done / json_count * 100 if json_count > 0 else 0
            bar  = '█' * int(30 * done / json_count) if json_count > 0 else ''
            elapsed = int(time.time() - start)
            eta = ""
            if done > 0 and elapsed > 0:
                secs_left = int(elapsed / done * (json_count - done))
                eta = f"  ETA ~{secs_left // 60}m{secs_left % 60:02d}s"
            sys.stdout.write(
                f"\r   [{bar:<30}] {done}/{json_count} ({pct:.1f}%){eta}   ")
            sys.stdout.flush()

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    try:
        proc = subprocess.Popen(cmd, cwd=af_cfg['alphafast_dir'])
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        stop_event.set()
        print("\n⚠️  Interrupted.")
        sys.exit(1)
    finally:
        stop_event.set()
        monitor.join(timeout=2)

    cif_files = glob.glob(
        os.path.join(AF_RESULTS_DIR, "**", "*_model.cif"), recursive=True)
    done_names = {n for f in cif_files for n in json_names if os.path.basename(f).startswith(n)}
    done = len(done_names)
    print(f"\r   [{'█' * 30}] {done}/{json_count} — done{' ' * 20}")

    if proc.returncode != 0:
        print(f"❌ AlphaFast exited with code {proc.returncode}")
        sys.exit(1)
    print("✨ ALPHAFAST BATCH PREDICTION COMPLETE ✨")

if __name__ == "__main__":
    main()
