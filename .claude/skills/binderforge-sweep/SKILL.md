---
name: binderforge-sweep
description: >
  Planning a multi-batch or multi-generator "how many designs do I actually
  need" campaign in BinderForge: comparing backbone_generator options
  (rfdiffusion vs complexa) for throughput and quality, running a campaign in
  batches with a plateau-based stopping rule instead of an arbitrary
  num_batches, and aggregating multiple runs into one comparison. Reach for
  this skill whenever the user says "run a sweep", "compare rfdiffusion and
  complexa", "how many designs should I generate", "ultimate generation",
  "when should I stop generating", or "bake-off between generators/predictors".
compatibility: "one or more configured, doctor-clean run folders"
allowed-tools: Bash, Read, Write
---

# BinderForge Sweep Skill

## There is no adaptive loop here — say so up front

RFdiffusion/Complexa + ProteinMPNN sample independently each batch; nothing
from step 4 scoring feeds back into what step 1/2 generate next
(`step1_rfd.num_batches` is a fixed count, not an adaptive budget). So a
"sweep" here means tracking a **yield curve** (best-of-N / Tier1 count vs.
total designs generated) and stopping on diminishing returns or a hard
budget — not gradient-style convergence. Set that expectation before the
user commits multiple weeks to a campaign.

## Comparing generators or predictors fairly

Hold everything except the variable under test constant:
- same `target_residues` / `hotspot_residues` / `binder_length`
- same GPU count / `gpu_devices`
- same total number of designs attempted (not the same `num_batches` —
  `batch_size × num_batches × nGPUs` differs in meaning between
  `rfdiffusion` and `complexa`, see `config.yaml`'s two `step1_*` blocks)

Report **both** raw hit rate (`Tier1 count / total attempted`) and hit rate
per GPU-hour — a generator that's faster but lower-quality can look like it
"wins" on raw count alone.

## Running in batches with a stopping rule

There's no built-in batch-runner — do it as repeated, incremented runs:
1. Run a fixed-size batch (e.g. one `num_batches` chunk) through steps 1→4.
2. Read `{project}_ranked.csv`, record cumulative Tier1 count and best
   `BinderScore` so far.
3. Compare against the previous batch's numbers — stop when the marginal
   Tier1 yield per additional batch drops below whatever threshold the user
   sets (or a fixed compute-hour budget, whichever they actually care about).
4. Don't run this unattended for days without a checkpoint — check in after
   each batch rather than committing the whole budget up front.

## Aggregating across runs

Each generator/predictor combination produces its own `{project}_ranked.csv`
under its own `NN-Label/outputs/04_Scoring/`. There's no cross-run
aggregation step in `scripts/05_Dashboard.py` — to build one comparison
table, concatenate the ranked CSVs, tag each row with its source run
(generator/predictor), and re-sort on `BinderScore`. This is a small script
that doesn't exist yet; offer to write one scoped to the specific runs being
compared rather than a speculative general-purpose merger.

## Saving the best designs at the end

The per-run dashboard/Excel/PyMOL outputs already do this per run
(`outputs/05_Dashboard/`) — for a multi-run campaign, the "final" artifact is
the aggregated comparison table above, plus copying forward the PyMOL
sessions for whichever designs end up in the merged top-N.
