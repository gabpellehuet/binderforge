"""
==============================================================================
   STEP 2: ProteinMPNN — sequence design + prediction-input prep
==============================================================================
   Designs sequences for the Step-1 backbones with ProteinMPNN (binder = chain A
   redesigned, target = chain B fixed), keeps the best sequence per design, and
   writes the per-design YAML that Step 3 (Boltz/AlphaFast) consumes.

   In : backbones  <01_BackboneGeneration>/gpu{N}/*.pdb
   Out: <02_ProteinMPNN>/gpu{N}/seqs/*.fa  and  yaml/*_seq_*.yaml

   Run via the orchestrator (`Run_Pipeline.py 2`); standalone needs the `mlfold` env.
   Authors: Naïs Sermet, Jean-Marie Bourhis.
==============================================================================
"""

import os
import sys
import yaml
import json
import subprocess
import glob
import time
from multiprocessing import Process
from Bio import SeqIO

# =================================================================
# 1. CONFIGURATION
# =================================================================
CONFIG_FILE = os.environ.get("PIPELINE_CONFIG", "config.yaml")

if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: {CONFIG_FILE} not found.")
    sys.exit(1)

with open(CONFIG_FILE, 'r') as f: cfg = yaml.safe_load(f)

WORK_DIR = cfg['work_dir']
IN_DIR   = os.path.join(WORK_DIR, cfg['outputs'].get('step1_backbone', cfg['outputs'].get('step1_rfd', 'outputs/01_BackboneGeneration')))
OUT_DIR  = os.path.join(WORK_DIR, cfg['outputs']['step2_mpnn'])

# UPDATED: Read from 'step2_mpnn' block
MPNN_SCRIPT = os.path.expanduser(cfg['tools']['mpnn_script'])
MPNN_DIR    = os.path.dirname(MPNN_SCRIPT)
MPNN_HELPER = os.path.join(MPNN_DIR, "helper_scripts", "parse_multiple_chains.py")

def adjust_batch_size(requested, n_seqs):
    """ProteinMPNN requires num_seq_per_target to be a multiple of batch_size.
    Snap the requested batch size to the nearest divisor of n_seqs (ties -> larger),
    e.g. 1000 sequences with batch_size 23 -> 25 (1000 / 25 = 40 batches)."""
    requested = max(1, int(requested))
    if n_seqs % requested == 0:
        return requested
    divisors = [d for d in range(1, n_seqs + 1) if n_seqs % d == 0]
    # Closest divisor to the request; on a tie prefer the larger batch size.
    best = min(divisors, key=lambda d: (abs(d - requested), -d))
    print(f"   ⚙️  batch_size {requested} does not divide {n_seqs} sequences; "
          f"auto-adjusted to {best} ({n_seqs // best} batches).")
    return best

SEQS_PER_TARGET = int(cfg['step2_mpnn']['seqs_per_target'])
BATCH_SIZE      = adjust_batch_size(cfg['step2_mpnn'].get('batch_size', 24), SEQS_PER_TARGET)

# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================

def get_target_seq():
    target_pdb = cfg['target']['pdb'] # Global target
    for record in SeqIO.parse(target_pdb, "pdb-atom"):
        return str(record.seq)
    return "ERROR_SEQ_NOT_FOUND"

def get_top_n_seqs(fasta_path, n_keep=1):
    if not os.path.exists(fasta_path): return []
    candidates = []
    with open(fasta_path, 'r') as f:
        entries = f.read().split('>')[1:] 
        for entry in entries: 
            lines = entry.split('\n')
            header = lines[0]
            seq = "".join(lines[1:]).strip()
            if "original" in header.lower(): continue
            try:
                parts = header.split(',')
                score_part = [p for p in parts if "score=" in p]
                if score_part:
                    score = float(score_part[0].split('=')[1])
                    candidates.append((score, seq))
            except: continue
    candidates.sort(key=lambda x: x[0])
    return candidates[:n_keep]

def create_boltz_yaml(design_base_name, seq_id, binder_seq, target_seq, output_folder):
    target_msa_path = cfg['target']['msa'] # Global target
    filename = f"{design_base_name}_seq_{seq_id}.yaml"
    output_path = os.path.join(output_folder, filename)
    boltz_data = {
        "version": 1,
        "sequences": [
            { "protein": { "id": "A", "sequence": binder_seq, "msa": "empty" } },
            { "protein": { "id": "B", "sequence": target_seq, "msa": target_msa_path } }
        ]
    }
    with open(output_path, 'w') as f: yaml.dump(boltz_data, f, sort_keys=False)

# =================================================================
# 3. CORE LOGIC
# =================================================================

def run_gpu_task(gpu_id, target_sequence):
    rfd_gpu   = os.path.join(IN_DIR, f"gpu{gpu_id}")
    mpnn_gpu  = os.path.join(OUT_DIR, f"gpu{gpu_id}")
    yaml_dest = os.path.join(mpnn_gpu, "yaml")
    seqs_dest = os.path.join(mpnn_gpu, "seqs")
    
    os.makedirs(mpnn_gpu, exist_ok=True)
    os.makedirs(yaml_dest, exist_ok=True)
    os.makedirs(seqs_dest, exist_ok=True)

    if not os.path.exists(rfd_gpu): return

    # Check for existing work
    existing_fastas = glob.glob(os.path.join(seqs_dest, "*.fa"))
    if len(existing_fastas) > 0:
        run_mpnn_success = True
    else:
        run_mpnn_success = False

    # ---------------------------------------------------------
    # STRATEGY A: BATCH MODE
    # ---------------------------------------------------------
    if not run_mpnn_success and os.path.exists(MPNN_HELPER):
        
        jsonl_path = os.path.join(mpnn_gpu, "parsed_pdbs.jsonl")
        chain_id_path = os.path.join(mpnn_gpu, "chain_id.jsonl")
        
        try:
            # 1. Parse PDBs
            subprocess.run(
                f"python {MPNN_HELPER} --input_path {rfd_gpu} --output_path {jsonl_path}", 
                shell=True, check=True, stderr=subprocess.DEVNULL
            )
            
            # 2. Create Chain Dict
            chain_dict = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    name = entry['name']
                    chain_dict[name] = [["A"], ["B"]] 
            
            with open(chain_id_path, 'w') as f:
                json.dump(chain_dict, f)

            # 3. Run MPNN (UPDATED KEYS)
            cmd_run = (f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
                       f"python {MPNN_SCRIPT} "
                       f"--jsonl_path {jsonl_path} "
                       f"--chain_id_jsonl {chain_id_path} "
                       f"--out_folder {mpnn_gpu} "
                       f"--num_seq_per_target {SEQS_PER_TARGET} "
                       f"--sampling_temp '{cfg['step2_mpnn']['sampling_temp']}' "
                       f"--batch_size {BATCH_SIZE} --suppress_print 1")

            subprocess.run(cmd_run, shell=True, check=True)
            run_mpnn_success = True

        except Exception:
            run_mpnn_success = False

    # ---------------------------------------------------------
    # STRATEGY B: LOOP MODE
    # ---------------------------------------------------------
    if not run_mpnn_success:
        pdb_files = glob.glob(os.path.join(rfd_gpu, "*.pdb"))
        if not pdb_files: pdb_files = glob.glob(os.path.join(rfd_gpu, "*.cif"))

        for i, pdb in enumerate(pdb_files):
            cmd = (f"export CUDA_VISIBLE_DEVICES={gpu_id} && "
                   f"python {MPNN_SCRIPT} "
                   f"--pdb_path {pdb} "
                   f"--pdb_path_chains 'A' "
                   f"--out_folder {mpnn_gpu} "
                   f"--num_seq_per_target {SEQS_PER_TARGET} "
                   f"--sampling_temp '{cfg['step2_mpnn']['sampling_temp']}' "
                   f"--batch_size {BATCH_SIZE} --suppress_print 1")
            subprocess.run(cmd, shell=True)

    # -----------------------------------------
    # Process Output -> YAMLs
    # -----------------------------------------
    fasta_files = glob.glob(os.path.join(seqs_dest, "*.fa"))
    N_TO_KEEP = cfg['step2_mpnn']['seqs_to_validate'] # Updated Key

    for fasta in fasta_files:
        design_name = os.path.splitext(os.path.basename(fasta))[0]
        top_variants = get_top_n_seqs(fasta, n_keep=N_TO_KEEP)
        
        for i, (score, binder_seq) in enumerate(top_variants):
            create_boltz_yaml(
                design_base_name=design_name,
                seq_id=i+1,
                binder_seq=binder_seq,
                target_seq=target_sequence,
                output_folder=yaml_dest
            )

if __name__ == "__main__":
    print("⏳ Loading Target Chain B Sequence...")
    target_seq = get_target_seq()
    
    input_files_0 = glob.glob(os.path.join(IN_DIR, "gpu0", "*"))
    input_files_1 = glob.glob(os.path.join(IN_DIR, "gpu1", "*"))
    total_inputs = len([f for f in input_files_0 + input_files_1 if f.endswith('.pdb') or f.endswith('.cif')])
    
    print(f"   Target: {total_inputs} designs to process.")
    print("   🚀 Running ProteinMPNN...")

    p0 = Process(target=run_gpu_task, args=(0, target_seq))
    p1 = Process(target=run_gpu_task, args=(1, target_seq))
    p0.start(); p1.start()
    
    while p0.is_alive() or p1.is_alive():
        out_0 = len(glob.glob(os.path.join(OUT_DIR, "gpu0", "seqs", "*.fa")))
        out_1 = len(glob.glob(os.path.join(OUT_DIR, "gpu1", "seqs", "*.fa")))
        current_total = out_0 + out_1
        
        if total_inputs > 0: percent = int((current_total / total_inputs) * 100)
        else: percent = 0
            
        sys.stdout.write(f"\r   ⏳ Progress: {current_total} / {total_inputs} ({percent}%) ")
        sys.stdout.flush()
        time.sleep(2)
    
    p0.join(); p1.join()
    print("\n✅ [Step 3] ProteinMPNN & YAML Generation Complete.")