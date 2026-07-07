"""
==============================================================================
   STEP 01: BACKBONE GENERATION — Proteina Complexa
==============================================================================

DESCRIPTION:
    Runs Proteina Complexa inside Docker to generate binder backbone structures.

    The project target PDB is mounted directly into Docker — no task needs to
    be configured inside the Complexa installation. All project-specific
    parameters (target, binder length, hotspots) are injected at runtime via
    Hydra ++ overrides into Complexa's target_dict_cfg.

    Outputs land directly in the project output directory via volume mount.
    PDBs are rechained after generation (binder B→A, target A→B) to match
    the convention expected by all downstream steps (03+).

USAGE:
    Called by 01_Run_Pipeline.py when backbone_generator = "complexa"
    Can also be run standalone: python scripts/02_ProteinaComplexa.py

PREREQUISITES:
    - Docker running with the image named in config (step1_complexa.docker_image)
    - Complexa installed at step1_complexa.complexa_dir with checkpoints in ckpts/
    - reward_model: null already set in Complexa's binder_generate.yaml

==============================================================================
"""

import argparse
import os
import re
import shutil
import sys
import time
import threading
import yaml
import subprocess
import glob

from contract import design_name as canonical_design_name

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

CONFIG_PATH = os.environ.get("PIPELINE_CONFIG", "config.yaml")

# Minimal Complexa pipeline YAML — generation only (evaluate/analyze stages removed).
# reward_model must be set in the file; it cannot be reliably overridden via ++.
COMPLEXA_YAML_TEMPLATE = """\
defaults:
  - pipeline/binder/binder_generate@generation
  - _self_

run_name: {run_name}
ckpt_path: ./ckpts
ckpt_name: complexa.ckpt
autoencoder_ckpt_path: ./ckpts/complexa_ae.ckpt
gen_njobs: {njobs}
seed: {seed}
{generation_block}"""

# Inner blocks (indented for nesting under a single `generation:` key).
# Must live under generation: because inf_cfg = cfg.generation in generate.py.

_REWARD_AF2_INNER = """\
  reward_model:
    _target_: proteinfoundation.rewards.alphafold2_reward.AF2RewardModel
    protocol: binder
    use_multimer: true
    af_params_dir: {af2_params_dir}
    num_recycles: 1
    use_initial_guess: true
    use_initial_atom_pos: false
    reward_weights:
      i_pae: -1.0
      plddt: 0.1
      i_ptm: 0.0
      con: 0.0
      dgram_cce: 0.0
"""

_REWARD_BIOINFORMATICS_INNER = """\
  reward_model:
    _target_: proteinfoundation.rewards.bioinformatics_reward.BioinformaticsRewardModel
    reward_weights:
      interface_sc: 1.0
      interface_dSASA: 0.5
      interface_hydrophobicity: 0.0
    reward_thresholds:
      interface_sc: 0.45
      interface_dSASA: 1.0
    reward_threshold_modes:
      interface_sc: max
      interface_dSASA: max
    structure_source: null
"""

_REFINEMENT_INNER = """\
  refinement:
    algorithm: sequence_hallucination
    refine_targets: final
    save_pre_refinement: none
    enable_soft_optimization: false
    enable_greedy_optimization: true
    n_temp_iters: 45
    n_hard_iters: 5
    n_recycles: 3
    n_greedy_iters: 15
    greedy_percentage: 1
    loss_weights:
      pae: 0.4
      plddt: 0.1
      i_pae: 0.1
      con: 1.0
      i_con: 1.0
      dgram_cce: 0.0
      rg: 0.3
      i_ptm: 0.05
      helix_binder: -0.3
"""


AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def extract_sequence(pdb_path, chain_id):
    """Read one-letter sequence for chain_id from ATOM records (residue order)."""
    seq = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            if line[21] != chain_id:
                continue
            res_key = line[22:27]   # resnum + insertion code
            if res_key in seen:
                continue
            seen.add(res_key)
            seq.append(AA3TO1.get(line[17:20].strip(), 'X'))
    return ''.join(seq)


def create_prediction_inputs(gpu_pdb_files, mpnn_out_dir, msa_path):
    """
    Write FASTA + structure-prediction YAML for each Complexa PDB,
    bypassing ProteinMPNN. Outputs land in step2_mpnn dirs so step 4
    can run unchanged.
    """
    for pdb_path in gpu_pdb_files:
        gpu_label   = os.path.basename(os.path.dirname(pdb_path))   # gpu0 / gpu1
        seqs_dir    = os.path.join(mpnn_out_dir, gpu_label, "seqs")
        yaml_dir    = os.path.join(mpnn_out_dir, gpu_label, "yaml")
        os.makedirs(seqs_dir, exist_ok=True)
        os.makedirs(yaml_dir, exist_ok=True)

        # The flattened PDB basename is already the canonical design name
        # (gpu{G}_{project}_B{batch}_S{sample}), so use it as-is.
        design_name = os.path.splitext(os.path.basename(pdb_path))[0]
        binder_seq  = extract_sequence(pdb_path, 'A')
        target_seq  = extract_sequence(pdb_path, 'B')

        # FASTA — minimal record, not parsed by downstream steps
        with open(os.path.join(seqs_dir, f"{design_name}.fa"), 'w') as f:
            f.write(f">{design_name}\n{binder_seq}\n")

        # Boltz / AlphaFast YAML — same format as ProteinMPNN output
        boltz_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": binder_seq, "msa": "empty"}},
                {"protein": {"id": "B", "sequence": target_seq, "msa": msa_path}},
            ],
        }
        with open(os.path.join(yaml_dir, f"{design_name}_seq_1.yaml"), 'w') as f:
            yaml.dump(boltz_data, f, sort_keys=False)

    return len(gpu_pdb_files)


def rechain_pdb(path):
    """
    Swap chain labels in-place: binder B→A, target A→B.
    Complexa outputs binder as chain B; pipeline expects binder as chain A.
    Uses chain X as temporary to avoid A↔B collision.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Pass 1: A→X, B→A
    intermediate = []
    for line in lines:
        if line.startswith(('ATOM', 'HETATM', 'TER')):
            ch = line[21]
            if ch == 'A':
                line = line[:21] + 'X' + line[22:]
            elif ch == 'B':
                line = line[:21] + 'A' + line[22:]
        intermediate.append(line)

    # Pass 2: X→B
    final = []
    for line in intermediate:
        if line.startswith(('ATOM', 'HETATM', 'TER')) and line[21] == 'X':
            line = line[:21] + 'B' + line[22:]
        final.append(line)

    with open(path, 'w') as f:
        f.writelines(final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 1 GPU, 2 samples, 20 steps")
    args = parser.parse_args()

    cfg = load_config()

    proj     = cfg['project_name']
    work_dir = cfg['work_dir']
    cx       = cfg['step1_complexa']
    gpus     = cfg.get('gpu_devices', [0, 1])
    out_base = cfg['outputs'].get('step1_backbone', cfg['outputs'].get('step1_complexa', 'outputs/01_BackboneGeneration'))

    complexa_dir    = cx['complexa_dir']
    docker_image    = cx['docker_image']
    target_input    = cx['target_input']
    binder_length   = cx['binder_length']       # [min, max]
    hotspot_res     = cx.get('hotspot_residues', [])
    nsteps          = cx['nsteps']
    nsamples        = cx['nsamples']
    batch_size      = cx.get('batch_size', 8)   # samples per forward pass; lower if CUDA OOM
    algorithm       = cx['algorithm']
    seed            = cx.get('seed', 5)
    reward_model    = cx.get('reward_model', None)   # None / "bioinformatics" / "af2"
    af2_params_dir  = cx.get('af2_params_dir', '/data/AF2/params')
    bon_replicas    = cx.get('best_of_n_replicas', 2)
    n_recycle       = cx.get('n_recycle', 0)
    use_refinement  = cx.get('refinement', False)
    target_pdb      = cfg['target']['pdb']      # absolute path from project config

    if args.test:
        nsamples  = 2
        nsteps    = 20
        gpus      = gpus[:1]
        print("⚠️  TEST MODE: nsamples=2, nsteps=20, 1 GPU")

    njobs     = len(gpus)
    run_name  = f"{proj}_complexa"
    task_name = proj                            # task label = project name; no Complexa config needed

    out_dir  = os.path.join(work_dir, out_base)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    use_bioinformatics = (reward_model == "bioinformatics")
    use_af2 = (reward_model == "af2")
    needs_af2_docker = use_af2 or use_refinement

    print("=" * 60)
    print("🧬 STEP 1: Backbone Generation — Proteina Complexa")
    print(f"   Target PDB:    {target_pdb}")
    print(f"   Target input:  {target_input}")
    print(f"   Binder length: {binder_length[0]}–{binder_length[1]} residues")
    print(f"   Run name:      {run_name}")
    print(f"   Jobs/GPUs:     {njobs}")
    print(f"   Steps:         {nsteps}")
    print(f"   Samples:       {nsamples} per job  ({nsamples * njobs} total)")
    print(f"   Algorithm:     {algorithm}" + (f"  (replicas={bon_replicas})" if algorithm == "best-of-n" else ""))
    print(f"   Reward model:  {reward_model or 'none'}" +
          (f"  (params: {af2_params_dir})" if use_af2 else ""))
    print(f"   Refinement:    {'sequence_hallucination' if use_refinement else 'off'}")
    print(f"   n_recycle:     {n_recycle}")
    print(f"   Output:        {out_dir}")
    print("=" * 60)

    if not os.path.exists(target_pdb):
        print(f"❌ Target PDB not found: {target_pdb}")
        sys.exit(1)

    # Write auto-generated Complexa pipeline YAML
    af2_params_in_docker = "/workspace/af2_params"
    generation_parts = []
    if use_af2:
        generation_parts.append(_REWARD_AF2_INNER.format(af2_params_dir=af2_params_in_docker))
    elif use_bioinformatics:
        generation_parts.append(_REWARD_BIOINFORMATICS_INNER)
    if use_refinement:
        generation_parts.append(_REFINEMENT_INNER)
    generation_block = "generation:\n" + "".join(generation_parts) if generation_parts else ""
    yaml_content = COMPLEXA_YAML_TEMPLATE.format(
        run_name=run_name, njobs=njobs, seed=seed, generation_block=generation_block
    )
    temp_yaml = os.path.join(work_dir, ".complexa_run.yaml")
    with open(temp_yaml, 'w') as f:
        f.write(yaml_content)

    workspace        = "/workspace/protein-foundation-models"
    target_in_docker = "/workspace/project_target.pdb"
    # generate.py writes PDBs to ./inference/project_pipeline_{task}_{run}/
    # Mount that exact path to the project output dir so PDBs land there directly.
    inference_subdir = f"project_pipeline_{task_name}_{run_name}"

    # Build Hydra ++ overrides to inject project target into Complexa's target registry
    binder_len_str = f"[{binder_length[0]},{binder_length[1]}]"
    overrides = [
        f"++generation.task_name={task_name}",
        f"++generation.target_dict_cfg.{task_name}.target_path={target_in_docker}",
        f"++generation.target_dict_cfg.{task_name}.target_input={target_input}",
        f"++generation.target_dict_cfg.{task_name}.binder_length={binder_len_str}",
        # source/target_filename/pdb_id are not used when target_path is set, but
        # OmegaConf.to_container(resolve=True) eagerly resolves the oc.select fallback
        # in binder_generate.yaml:pdb_path, so these must exist to avoid InterpolationKeyError.
        f"++generation.target_dict_cfg.{task_name}.source=custom",
        f"++generation.target_dict_cfg.{task_name}.target_filename=target",
        f"++generation.target_dict_cfg.{task_name}.pdb_id=null",
        f"++generation.dataloader.batch_size={batch_size}",
        f"++generation.args.nsteps={nsteps}",
        f"++generation.dataloader.dataset.nres.nsamples={nsamples}",
        f"++generation.search.algorithm={algorithm}",
        # Hydra defaults to writing outputs/{date}/{time}/ in the Docker workdir (the
        # Complexa install). With --user, that dir is not writable. Redirect to the
        # inference subdir, which is already mounted as our project output dir.
        f"hydra.run.dir={workspace}/inference/{inference_subdir}",
    ]
    if algorithm == "best-of-n":
        overrides.append(f"++generation.search.best_of_n.replicas={bon_replicas}")
    if n_recycle > 0:
        overrides.append(f"++generation.n_recycle={n_recycle}")
    # hotspot_residues is always referenced in binder_generate.yaml so must always be set
    if hotspot_res:
        hs_str = "[" + ",".join(f'"{r}"' for r in hotspot_res) + "]"
    else:
        hs_str = "[]"
    overrides.append(f"++generation.target_dict_cfg.{task_name}.hotspot_residues={hs_str}")

    sc_binary = f"{workspace}/src/proteinfoundation/result_analysis/sc"
    docker_cmd = [
        "docker", "run",
        "--gpus", "all",
        "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-w", workspace,
        "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "-e", "PYTHONUNBUFFERED=1",
        *(["-e", f"SC_EXEC={sc_binary}"] if use_bioinformatics else []),
        *(["-e", f"AF2_DIR={af2_params_in_docker}",
           "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
           "-v", f"{af2_params_dir}:{af2_params_in_docker}:ro"] if needs_af2_docker else []),
        "-v", f"{complexa_dir}:{workspace}",
        "-v", f"{complexa_dir}/ckpts:{workspace}/checkpoints",
        # Map the exact inference subdir generate.py writes to → project output dir
        "-v", f"{out_dir}:{workspace}/inference/{inference_subdir}",
        # Redirect Complexa's ./logs/ into the project step2 output dir
        "-v", f"{logs_dir}:{workspace}/logs",
        # Auto-generated pipeline config injected into Complexa's config tree
        "-v", f"{temp_yaml}:{workspace}/configs/project_pipeline.yaml",
        # Project target PDB mounted into Docker
        "-v", f"{target_pdb}:{target_in_docker}",
        docker_image,
        "complexa", "generate", "configs/project_pipeline.yaml",
        *overrides,
    ]

    print(f"\n   Overrides:")
    for o in overrides:
        print(f"     {o}")
    print("-" * 60)
    print("   🚀 Running Complexa (streaming Docker output)...\n")

    stop_counter = threading.Event()
    start_time = time.time()
    _console = Console()

    def _parse_log_fraction(log_path, max_bytes=32768):
        try:
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - max_bytes))
                tail = f.read().decode('utf-8', errors='replace')
            cur = tot = 0
            for line in tail.splitlines():
                m = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
            return cur / tot if tot > 0 else 0.0
        except Exception:
            return 0.0

    _progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Complexa[/bold cyan]"),
        BarColumn(bar_width=50),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=_console,
    )
    _task = _progress.add_task("generate", total=100)

    def _counter_loop():
        while not stop_counter.wait(10):
            logs = [
                lf for lf in glob.glob(os.path.join(logs_dir, "*.log"))
                if os.path.getmtime(lf) > start_time - 5
            ]
            fracs = [_parse_log_fraction(lf) for lf in logs]
            if fracs:
                avg_pct = sum(fracs) / len(fracs) * 100
                _progress.update(_task, completed=avg_pct)

    counter_thread = threading.Thread(target=_counter_loop, daemon=True)
    counter_thread.start()

    try:
        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        with Live(_progress, console=_console, refresh_per_second=2):
            for line in proc.stdout:
                _console.print(f"   [complexa] {line}", end='', markup=False, highlight=False)
        proc.wait()
    except FileNotFoundError:
        stop_counter.set()
        _console.print("❌ Docker not found. Is Docker running?")
        os.remove(temp_yaml)
        sys.exit(1)
    finally:
        stop_counter.set()

    os.remove(temp_yaml)

    if proc.returncode != 0:
        print(f"\n❌ Docker exited with code {proc.returncode}.")
        sys.exit(1)

    # Post-process: rechain then flatten into gpu{n}/ for MPNN compatibility
    print("\n" + "-" * 60)
    print("   Post-processing: rechaining PDBs (binder B→A, target A→B)...")

    # Exclude gpu*/ subdirs so we don't re-process on subsequent runs
    pdb_files = [
        p for p in glob.glob(os.path.join(out_dir, "**", "*.pdb"), recursive=True)
        if not re.search(r"[/\\]gpu\d+[/\\]", p)
    ]

    if not pdb_files:
        print(f"   ⚠️  No PDB files found in {out_dir}")
    else:
        for pdb in pdb_files:
            rechain_pdb(pdb)
        print(f"   ✅ {len(pdb_files)} designs rechained.")

        # Flatten into gpu{N}/ with the canonical basename (see contract.design_name):
        #   gpu{G}_{project}_B{idx:04d}_S0   (Complexa has no sample dimension → S0)
        print("   Flattening into gpu{n}/ dirs...")
        job_dirs_to_remove = set()
        gpu_counters = {}
        for pdb in sorted(pdb_files):
            m = re.search(r"[/\\]job_(\d+)_", pdb)
            job_id = int(m.group(1)) if m else 0
            gpu_dir = os.path.join(out_dir, f"gpu{job_id}")
            os.makedirs(gpu_dir, exist_ok=True)
            idx = gpu_counters.get(job_id, 0)
            new_name = canonical_design_name(job_id, proj, f"{idx:04d}", 0) + ".pdb"
            shutil.copy2(pdb, os.path.join(gpu_dir, new_name))
            gpu_counters[job_id] = idx + 1
            job_dirs_to_remove.add(os.path.dirname(pdb))

        for job_dir in job_dirs_to_remove:
            shutil.rmtree(job_dir, ignore_errors=True)
        print(f"   ✅ {len(pdb_files)} PDBs in gpu dirs, job subdirs removed.")

        # Write FASTAs + structure-prediction YAMLs directly — skips ProteinMPNN
        print("   Extracting sequences and writing prediction inputs...")
        gpu_pdbs    = glob.glob(os.path.join(out_dir, "gpu*", "*.pdb"))
        mpnn_out    = os.path.join(work_dir, cfg['outputs']['step2_mpnn'])
        n = create_prediction_inputs(gpu_pdbs, mpnn_out, cfg['target']['msa'])
        print(f"   ✅ {n} FASTA + YAML pairs written to {mpnn_out}")

    print(f"\n✅ Step 2 (Complexa) complete.")
    print(f"   Outputs: {out_dir}")


if __name__ == "__main__":
    main()
