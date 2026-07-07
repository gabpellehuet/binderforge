#!/bin/bash
# 04_Boltz.sh
# ==============================================================================
# 🧬 STEP 03: BOLTZ STRUCTURE PREDICTION
# ==============================================================================
#
#   DESCRIPTION:
#   ---------------------------------------------------------------------------
#   This script executes the structure prediction step using Boltz-1. It takes
#   the YAML files generated in Step 03 (which contain the designed binder 
#   sequence + fixed target sequence & MSA) and predicts the 3D complex structure.
#
#   KEY FUNCTIONS:
#   1. PARALLEL EXECUTION: Launches two separate Boltz processes, one for each
#      GPU (gpu0 and gpu1), to maximize throughput.
#   2. PREDICTION: Uses the provided MSA and recycling steps to generate high-
#      confidence models of the Binder-Target complex.
#   3. OUTPUT: Generates .cif or .pdb files and confidence JSONs (pae/plddt).
#
#   INPUTS:
#   - YAML files from: outputs/03_ProteinMPNN/gpu{0,1}/yaml/
#
#   OUTPUTS:
#   - Structure files: outputs/04_Boltz/gpu{0,1}/{design_name}/
#   - Logs: outputs/04_Boltz/boltz_gpu{0,1}.log
#
#   USAGE:
#   ---------------------------------------------------------------------------
#   Managed by Orchestrator:
#       python 01_Run_Pipeline.py 4
#
#   Standalone (requires 'boltz' env):
#       ./scripts/04_Boltz.sh
#
# Written by Naïs Sermet, jean-Marie Bourhis
# ==============================================================================


CONFIG_FILE="${PIPELINE_CONFIG:-config.yaml}"

# Helper to read YAML
get_val() { python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))$1)" 2>/dev/null; }

WORK_DIR=$(get_val "['work_dir']")
IN_BASE=$(get_val "['outputs']['step2_mpnn']")
OUT_BASE=$(get_val "['outputs']['step3_boltz']")

# UPDATED KEYS
REC=$(get_val "['step3_boltz']['recycling']")
SAMPLES=$(get_val "['step3_boltz']['samples']")

# Paths
IN_DIR="$WORK_DIR/$IN_BASE"
OUT_DIR="$WORK_DIR/$OUT_BASE"
mkdir -p "$OUT_DIR/gpu0" "$OUT_DIR/gpu1"

# Calculate Target
count_in_0=$(ls -1 "$IN_DIR/gpu0/yaml/"*.yaml 2>/dev/null | wc -l)
count_in_1=$(ls -1 "$IN_DIR/gpu1/yaml/"*.yaml 2>/dev/null | wc -l)
TOTAL_EXPECTED=$((count_in_0 + count_in_1))

echo "------------------------------------------------"
echo "🧬 STEP 3: Boltz Prediction"
echo "   Input:  $IN_DIR"
echo "   Output: $OUT_DIR"
echo "   Target: $TOTAL_EXPECTED predictions"
echo "------------------------------------------------"

# Launch Jobs
echo "   Launching GPU 0..."
CUDA_VISIBLE_DEVICES=0 boltz predict "$IN_DIR/gpu0/yaml" \
    --out_dir "$OUT_DIR/gpu0" \
    --recycling_steps $REC \
    --diffusion_samples $SAMPLES \
    --override \
    --write_full_pae > "$OUT_DIR/boltz_gpu0.log" 2>&1 &
PID0=$!

echo "   Launching GPU 1..."
CUDA_VISIBLE_DEVICES=1 boltz predict "$IN_DIR/gpu1/yaml" \
    --out_dir "$OUT_DIR/gpu1" \
    --recycling_steps $REC \
    --diffusion_samples $SAMPLES \
    --override \
    --write_full_pae > "$OUT_DIR/boltz_gpu1.log" 2>&1 &
PID1=$!

# Monitor
echo "   🚀 Running..."
while kill -0 $PID0 2>/dev/null || kill -0 $PID1 2>/dev/null; do
    current_0=$(find "$OUT_DIR/gpu0" -name "confidence_*.json" 2>/dev/null | wc -l)
    current_1=$(find "$OUT_DIR/gpu1" -name "confidence_*.json" 2>/dev/null | wc -l)
    current_total=$((current_0 + current_1))
    
    if [ "$TOTAL_EXPECTED" -gt 0 ]; then
        percent=$((current_total * 100 / TOTAL_EXPECTED))
    else
        percent=0
    fi
    echo -ne "\r   ⏳ Progress: $current_total / $TOTAL_EXPECTED ($percent%) "
    sleep 30
done

wait
echo -e "\n✅ Boltz Complete."