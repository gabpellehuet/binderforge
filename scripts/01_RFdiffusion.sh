#!/bin/bash
# ==============================================================================
# 🧬 STEP 01: RFdiffusion BACKBONE GENERATION
# ==============================================================================
# All parameters read from config.yaml — no separate JSON file required.
# A temporary input JSON is generated at runtime and removed on exit.
# ==============================================================================

shopt -s expand_aliases
source ~/.bashrc 2>/dev/null
CONFIG="${PIPELINE_CONFIG:-config.yaml}"

# Helper to read YAML values
get_val() { python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))$1)" 2>/dev/null; }

# Conda source comes from the resolved config (orchestrator writes conda_source);
# fall back to the legacy default only if it's absent.
CONDA_SRC=$(get_val "['conda_source']")
[ -z "$CONDA_SRC" ] || [ "$CONDA_SRC" = "None" ] && CONDA_SRC="/usr/local/miniforge3/etc/profile.d/conda.sh"

# ── Load Configuration ────────────────────────────────────────────────────────
WORK_DIR=$(get_val "['work_dir']")
OUT_BASE=$(get_val "['outputs']['step1_backbone']")
PROJ=$(get_val "['project_name']")
CONV_ENV=$(get_val "['envs']['mpnn']")

BATCH=$(get_val "['step1_rfd']['batch_size']")
NUM=$(get_val "['step1_rfd']['num_batches']")
STEP_SCALE=$(get_val "['step1_rfd']['step_scale']")
GAMMA=$(get_val "['step1_rfd']['gamma_0']")
CKPT=$(get_val "['checkpoints']['rfd']")

OUT_DIR="$WORK_DIR/$OUT_BASE"
mkdir -p "$OUT_DIR/gpu0" "$OUT_DIR/gpu1"
TOTAL_EXPECTED=$((BATCH * NUM * 2))

# ── Generate temporary input JSON from config ─────────────────────────────────
TMP_JSON="$OUT_DIR/rfd_input_$$.json"

# Register cleanup: runs on normal exit AND Ctrl+C (SIGINT/SIGTERM)
trap 'rm -f "$TMP_JSON"' EXIT

export _CFG="$CONFIG"
export _TMP_JSON="$TMP_JSON"

python3 <<'PYEOF'
import yaml, json, os, sys

cfg = yaml.safe_load(open(os.environ["_CFG"]))
proj      = cfg["project_name"]
pdb       = cfg["target"]["pdb"]
rfd       = cfg.get("step1_rfd", {})

blen      = rfd.get("binder_length", [50, 150])
tgt_res   = rfd.get("target_residues", "")
hotspot   = rfd.get("hotspot_residues", [])
non_loopy = rfd.get("is_non_loopy", True)
ori_token = rfd.get("ori_token", None)          # explicit [x,y,z] origin override; None = let strategy infer

# Contig: e.g. "50-150,/0,A1-346"
contig = f"{blen[0]}-{blen[1]},/0,{tgt_res}"

# RFd3 select_hotspots = {residue: atoms}, e.g. {"A271": "ALL"} (docs/input.md).
# User gives residues only (["A271","A244"]) → target ALL atoms of each.
hs_dict = {res: "ALL" for res in hotspot}

# RFd3 infer_ori_strategy accepts "com" or "hotspots" ("hotspots" works best WITH hotspots).
# Default is auto: hotspots when hotspots are given, else com. Set ori_strategy in config to force one.
ori_strategy = rfd.get("ori_strategy") or ("hotspots" if hs_dict else "com")

# Only include select_hotspots (HS mode) and ori_token (explicit origin) when set —
# otherwise "com" infers the origin from THIS target's center of mass. A hardcoded
# ori_token would anchor the binder to a fixed point unrelated to the target.
entry = {
    "dialect": 2,
    "infer_ori_strategy": ori_strategy,
    "input": pdb,
    "contig": contig,
    "is_non_loopy": non_loopy,
}
if hs_dict:
    entry["select_hotspots"] = hs_dict
if ori_token is not None:
    entry["ori_token"] = ori_token
task = {proj: entry}

with open(os.environ["_TMP_JSON"], "w") as f:
    json.dump(task, f, indent=2)

print(f"   Target PDB:      {pdb}")
print(f"   Binder length:   {blen[0]}–{blen[1]} residues")
print(f"   Target residues: {tgt_res}")
print(f"   Hotspots:        {hotspot if hotspot else 'none (full surface)'}")
print(f"   Origin:          {ori_strategy}" + (f"  ori_token={ori_token}" if ori_token is not None else "  (inferred from target)"))
print(f"   Contig:          {contig}")
print(f"   is_non_loopy:    {non_loopy}")
PYEOF

if [ ! -f "$TMP_JSON" ]; then
    echo "❌ Failed to generate RFdiffusion input JSON."
    echo "   Check config.yaml: target.pdb, step1_rfd.binder_length, step1_rfd.target_residues"
    exit 1
fi

echo "------------------------------------------------"
echo "🧬 STEP 1: RFdiffusion"
echo "   Output:  $OUT_DIR"
echo "   Params:  Step Scale=$STEP_SCALE | Gamma=$GAMMA"
echo "   Target:  $TOTAL_EXPECTED designs ($BATCH × $NUM batches × 2 GPUs)"
echo "------------------------------------------------"

# ── GPU 0 ─────────────────────────────────────────────────────────────────────
echo "   Launching GPU 0..."
CUDA_VISIBLE_DEVICES=0 rfd3 design \
    inputs="$TMP_JSON" \
    out_dir="$OUT_DIR/gpu0" \
    diffusion_batch_size=$BATCH \
    n_batches=$NUM \
    ckpt_path=$CKPT \
    inference_sampler.step_scale=$STEP_SCALE \
    inference_sampler.gamma_0=$GAMMA \
    skip_existing=False > "$OUT_DIR/gpu0.log" 2>&1 &
PID0=$!

# ── GPU 1 ─────────────────────────────────────────────────────────────────────
echo "   Launching GPU 1..."
CUDA_VISIBLE_DEVICES=1 rfd3 design \
    inputs="$TMP_JSON" \
    out_dir="$OUT_DIR/gpu1" \
    diffusion_batch_size=$BATCH \
    n_batches=$NUM \
    ckpt_path=$CKPT \
    inference_sampler.step_scale=$STEP_SCALE \
    inference_sampler.gamma_0=$GAMMA \
    skip_existing=False > "$OUT_DIR/gpu1.log" 2>&1 &
PID1=$!

# ── Monitor ───────────────────────────────────────────────────────────────────
echo "   🚀 Running... (Press Ctrl+C to stop)"
while kill -0 $PID0 2>/dev/null || kill -0 $PID1 2>/dev/null; do
    count0=$(ls -1 "$OUT_DIR/gpu0"/*.cif.gz 2>/dev/null | wc -l)
    count1=$(ls -1 "$OUT_DIR/gpu1"/*.cif.gz 2>/dev/null | wc -l)
    current_total=$((count0 + count1))
    if [ "$TOTAL_EXPECTED" -gt 0 ]; then
        percent=$((current_total * 100 / TOTAL_EXPECTED))
    else
        percent=0
    fi
    echo -ne "\r   ⏳ Progress: $current_total / $TOTAL_EXPECTED ($percent%) "
    sleep 10
done

echo -e "\n   ✅ Generation finished. Starting processing..."
wait
# trap will remove TMP_JSON on exit

# ── Post-Processing: decompress, convert CIF→PDB, rename ─────────────────────
echo "------------------------------------------------"
echo "   Switching env to '$CONV_ENV' for conversion & renaming..."

source "$CONDA_SRC"
conda activate "$CONV_ENV"

python3 -c "
import os, glob, gzip, concurrent.futures
from Bio.PDB import MMCIFParser, PDBIO

BASE_DIR  = '$OUT_DIR'
PROJ_NAME = '$PROJ'

def process_gpu_folder(args):
    gpu_name, prefix = args
    folder_path = os.path.join(BASE_DIR, gpu_name)
    if not os.path.exists(folder_path): return

    for gz in glob.glob(os.path.join(folder_path, '*.gz')):
        try:
            out = gz[:-3]
            with gzip.open(gz, 'rb') as fi, open(out, 'wb') as fo:
                fo.write(fi.read())
            os.remove(gz)
        except Exception: pass

    source_files = glob.glob(os.path.join(folder_path, '*.cif'))
    if not source_files:
        source_files = glob.glob(os.path.join(folder_path, '*.pdb'))

    parser = MMCIFParser(QUIET=True)
    io     = PDBIO()
    count  = 0
    for src in source_files:
        try:
            filename    = os.path.basename(src)
            # Canonical design name / basename (see scripts/contract.py DESIGN_NAME_RE):
            #   gpu{G}_{project}_B{batch}_S{sample}
            if filename.startswith(f'gpu{prefix}_'): continue
            base_no_ext = os.path.splitext(filename)[0]
            parts       = base_no_ext.split('_')
            if 'model' in parts:
                idx      = parts.index('model')
                batch    = parts[idx - 1]
                sample   = parts[idx + 1]
                new_base = f'gpu{prefix}_{PROJ_NAME}_B{batch}_S{sample}'
            else:
                new_base = f'gpu{prefix}_{PROJ_NAME}_{base_no_ext}'
            pdb_out = os.path.join(folder_path, new_base + '.pdb')
            if src.endswith('.cif'):
                io.set_structure(parser.get_structure('tmp', src))
                io.save(pdb_out)
                os.remove(src)
            elif src.endswith('.pdb') and src != pdb_out:
                os.rename(src, pdb_out)
            count += 1
        except Exception: pass

    print(f'   ✅ {gpu_name}: Processed {count} designs.')

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_gpu_folder, [('gpu0', '0'), ('gpu1', '1')])
"

echo "------------------------------------------------"
echo "✅ RFdiffusion Complete."
