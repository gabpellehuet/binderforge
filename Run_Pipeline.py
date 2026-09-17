"""
==============================================================================
   🧬 DESIGN PIPELINE — MASTER ORCHESTRATOR (convention-based)
==============================================================================

   Layout (folder structure IS the convention):

     <TOOL ROOT>/                 scripts/ + this Run_Pipeline.py
     └── <ProjectName>/           one target/binder
         ├── target.pdb           shared by all configs
         ├── target_msa.a3m       created on 1st run (step 0) if missing
         └── <NN-Label>/          one config/run
             ├── config.yaml      PARAMETERS ONLY
             └── outputs/         this run's outputs + manifest + summary

   USAGE — cd into a config folder, then run this shared orchestrator:
       cd MyProject/00-RFdiffusion1
       python /path/to/tool/Run_Pipeline.py all
       python /path/to/tool/Run_Pipeline.py 4      # single step (scoring)

   The orchestrator derives, from the folder location:
     - binder_name  = the Project folder name
     - config_id    = the NN- prefix of the config folder (auto-added if missing)
     - work_dir     = the config folder (outputs land in ./outputs)
     - target pdb/msa = the Project folder (target.pdb / target_msa.a3m)
   config.yaml only needs design PARAMETERS; anything above can still be overridden
   there. The resolved (complete) config is written to outputs/.resolved_config.yaml
   and every step consumes it via $PIPELINE_CONFIG.

   Conda envs (config envs:): rf3, mlfold, boltz, pyrosetta, base (+ msa for step 0).
==============================================================================
"""
import os
import sys
import re
import glob
import yaml
import subprocess
import argparse

TOOL_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TOOL_ROOT, "scripts"))
from config_schema import validate_config
from provenance import write_manifest
from run_summary import write_run_summary
from contract import OUTPUT_LAYOUT as DEFAULT_OUTPUTS   # single source of truth for the folder layout
import sitecfg   # machine layer: site.yaml load/merge, conda source, doctor

# Conda source to `source` before activating envs. Resolved at runtime from
# site.yaml / $CONDA_EXE (see sitecfg.resolve_conda_source); this is only the fallback.
CONDA_SOURCE = "/usr/local/miniforge3/etc/profile.d/conda.sh"


# ==============================================================================
# CONVENTION RESOLUTION
# ==============================================================================

def ensure_config_prefix(config_dir):
    """Guarantee this config folder has a UNIQUE 'NN-' config_id among its siblings.

    config_id must be unique per project (it's part of binder_id); labels may repeat.
    If the folder has no 'NN-' prefix, or its number is already used by another sibling
    config folder, assign the next free number (max sibling id + 1) and rename the folder,
    keeping the label. Returns the (possibly new) folder path.
    """
    base = os.path.basename(os.path.normpath(config_dir))
    parent = os.path.dirname(os.path.normpath(config_dir))
    m = re.match(r'^(\d+)-', base)
    my_id = int(m.group(1)) if m else None
    label = base[m.end():] if m else base                 # folder name without the 'NN-'

    sib_ids = []
    for name in os.listdir(parent):
        if name == base or not os.path.isdir(os.path.join(parent, name)):
            continue                                       # skip self and non-dirs
        sm = re.match(r'^(\d+)-', name)
        if sm:
            sib_ids.append(int(sm.group(1)))

    if my_id is not None and my_id not in sib_ids:
        return config_dir                                  # already prefixed and unique

    new_id = (max(sib_ids) + 1) if sib_ids else 0          # > every sibling → guaranteed free
    new_dir = os.path.join(parent, f"{new_id:02d}-{label}")
    os.rename(config_dir, new_dir)
    why = "had no NN- prefix" if my_id is None else f"had a duplicate config_id ({my_id:02d})"
    print(f"   🔢 Config folder {why} → renamed '{base}' → '{os.path.basename(new_dir)}'")
    print(f"      (your shell is still in this folder; run `cd \"{new_dir}\"` to refresh the path)")
    return new_dir


# Legacy config param-block keys → current names. The step numbers now match the
# real step order (1=Backbone, 2=Sequence, 3=Prediction). Normalizing here — before
# validation, output resolution, and writing .resolved_config.yaml — means old
# run-folder config.yaml files keep working while every step and config_schema see
# only the new keys. (Output-folder keys need no such map: they live only in
# contract.OUTPUT_LAYOUT and the resolved config is rebuilt each run.)
_LEGACY_PARAM_KEYS = {
    'step2_rfd':       'step1_rfd',
    'step2_complexa':  'step1_complexa',
    'step3_mpnn':      'step2_mpnn',
    'step4_boltz':     'step3_boltz',
    'step4_alphafast': 'step3_alphafast',
}


def find_target_pdb(project_dir):
    """Resolve the target structure in a project folder without needing a fixed name.

    Prefer `target.pdb` if present, else the SOLE `*.pdb` in the folder (outputs never
    land here, so a single .pdb is unambiguous). Returns (path, candidates); on a
    missing/ambiguous choice, path is the conventional `target.pdb` (which won't exist)
    so pre-flight can report it, and `candidates` lists what was found.
    """
    conv = os.path.join(project_dir, "target.pdb")
    if os.path.exists(conv):
        return conv, [conv]
    pdbs = sorted(glob.glob(os.path.join(project_dir, "*.pdb")))
    if len(pdbs) == 1:
        return pdbs[0], pdbs
    return conv, pdbs                       # 0 or >1 → report via check_inputs


def find_target_msa(project_dir):
    """Resolve the target MSA like the PDB: prefer `target_msa.a3m`, else the SOLE
    `*.a3m` in the project folder (any name). On none/ambiguous, return `target_msa.a3m`
    — where step 0 generates it."""
    conv = os.path.join(project_dir, "target_msa.a3m")
    if os.path.exists(conv):
        return conv
    a3ms = sorted(glob.glob(os.path.join(project_dir, "*.a3m")))
    return a3ms[0] if len(a3ms) == 1 else conv


_RES_RANGE_RE = re.compile(r'([A-Za-z])(\d+)\s*-\s*(\d+)')


def pdb_chain_ranges(pdb_path):
    """{chain: (min_resnum, max_resnum)} from a PDB's ATOM records."""
    chains = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                ch = line[21]
                try:
                    rn = int(line[22:26])
                except ValueError:
                    continue
                lo, hi = chains.get(ch, (rn, rn))
                chains[ch] = (min(lo, rn), max(hi, rn))
    return chains


def check_target_residues(target_residues, pdb_path):
    """Validate a `<chain><start>-<end>` target_residues spec against the target PDB.

    Returns (errors, warnings): ERROR if a range names a missing chain or extends beyond
    the PDB's residues (RFdiffusion silently yields 0 backbones); WARNING if it's a strict
    subset (designing against part of the target — sometimes intended).
    """
    errors, warnings = [], []
    tr = str(target_residues or "").strip()
    if not tr or not os.path.exists(pdb_path):
        return errors, warnings
    segs = _RES_RANGE_RE.findall(tr)
    if not segs:
        warnings.append(f"target_residues '{tr}' isn't in '<chain><start>-<end>' form — not validated against the PDB.")
        return errors, warnings
    ranges = pdb_chain_ranges(pdb_path)
    for ch, s, e in segs:
        start, end = int(s), int(e)
        if ch not in ranges:
            errors.append(f"target_residues chain '{ch}' is not in the target PDB "
                          f"(chains present: {', '.join(sorted(ranges)) or 'none'}).")
        elif start > end:
            errors.append(f"target_residues {ch}{start}-{end} is reversed (start > end).")
        elif start < ranges[ch][0] or end > ranges[ch][1]:
            lo, hi = ranges[ch]
            errors.append(f"target_residues {ch}{start}-{end} extends beyond the target's "
                          f"residues ({ch}{lo}-{hi}) — RFdiffusion would silently make 0 backbones.")
        elif (start, end) != ranges[ch]:
            lo, hi = ranges[ch]
            warnings.append(f"target_residues {ch}{start}-{end} is a SUBSET of the target "
                            f"({ch}{lo}-{hi}) — designing against part of it (intended?).")
    return errors, warnings


def resolve_convention(user_cfg, config_dir, project_dir):
    """Fill work_dir / identity / target / outputs from the folder convention.
    Anything already set in the user config wins (explicit override)."""
    cfg = dict(user_cfg or {})
    for old, new in _LEGACY_PARAM_KEYS.items():
        if old in cfg and new not in cfg:
            cfg[new] = cfg.pop(old)
    base = os.path.basename(os.path.normpath(config_dir))
    m = re.match(r'^(\d+)-', base)

    cfg['work_dir']     = config_dir
    cfg['binder_name']  = cfg.get('binder_name') or os.path.basename(os.path.normpath(project_dir))
    if cfg.get('config_id') is None:
        cfg['config_id'] = int(m.group(1)) if m else 0
    cfg['project_name'] = cfg.get('project_name') or base    # legacy label → output CSV base name

    tgt = dict(cfg.get('target') or {})
    if not tgt.get('pdb'):
        tgt['pdb'] = find_target_pdb(project_dir)[0]
    tgt['msa'] = tgt.get('msa') or find_target_msa(project_dir)
    cfg['target'] = tgt

    outputs = dict(DEFAULT_OUTPUTS)
    outputs.update(cfg.get('outputs') or {})
    cfg['outputs'] = outputs
    return cfg


# ==============================================================================
# STEP RUNNER
# ==============================================================================

def run_step(step_name, command, runner, soft=False):
    """Execute a step through its runner. soft=True → warn and continue on failure.

    `runner` is the execution-environment spec (the seam that lets a step later be
    moved to Docker without touching call sites). Today only the conda runner
    exists: pass the conda env name (str), or {'type': 'conda', 'env': <name>}.
    """
    if isinstance(runner, dict):
        env_name = runner.get('env')
    else:
        env_name = runner
    print(f"\n🚀 STARTING: {step_name}")
    print(f"   Environment: {env_name}")
    print(f"   Command:     {command}")
    print("-" * 60)
    full_cmd = f"source {CONDA_SOURCE} && conda activate {env_name} && {command}"
    try:
        subprocess.run(full_cmd, shell=True, check=True, executable='/bin/bash')
        print(f"✅ FINISHED: {step_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        if soft:
            print(f"⚠️  {step_name} did not complete (exit {e.returncode}) — continuing.\n")
            return False
        print(f"\n❌ ERROR in {step_name}: failed with exit code {e.returncode}.")
        sys.exit(1)


def check_inputs(cfg, config_yaml):
    """Pre-flight: validate the resolved config, then verify required input files exist."""
    print("🔍 Running Pre-flight Checks...")

    # Target structure — clearer message than the generic schema check (it's a FILE,
    # not a config edit). Only when not explicitly overridden and not yet resolved.
    pdb = cfg['target']['pdb']
    if not os.path.exists(pdb):
        project_dir = os.path.dirname(os.path.normpath(cfg['work_dir']))
        pdbs = sorted(glob.glob(os.path.join(project_dir, "*.pdb")))
        print("\n" + "!" * 60)
        if len(pdbs) > 1:
            print("❌ Multiple .pdb files in the project folder — can't pick the target:")
            for p in pdbs:
                print(f"     - {os.path.basename(p)}")
            print(f"   → rename one to target.pdb, or set  target: {{pdb: ...}}  in {config_yaml}")
        else:
            print(f"❌ No target structure found in {project_dir}")
            print("   → put your target .pdb there (any filename if it's the only one).")
        print("!" * 60)
        return False

    # target_residues sanity (RFdiffusion): must fall within the target PDB's residues,
    # else RFdiffusion silently makes 0 backbones and the failure only shows up at step 3.
    if cfg.get('backbone_generator', 'rfdiffusion').lower() == 'rfdiffusion':
        terrs, twarns = check_target_residues(
            (cfg.get('step1_rfd') or {}).get('target_residues'), pdb)
        for w in twarns:
            print(f"   ⚠️  {w}")
        if terrs:
            print("\n" + "!" * 60)
            print("❌ target_residues problem:")
            for e in terrs:
                print(f"   - {e}")
            print(f"   → fix step1_rfd.target_residues in {config_yaml}  "
                  f"(target = {os.path.basename(pdb)}).")
            print("!" * 60)
            return False

    ok, errors, warnings = validate_config(cfg)
    for w in warnings:
        print(f"   ⚠️  {w}")
    if not ok:
        print("\n" + "!" * 60)
        print("❌ CONFIG VALIDATION FAILED:")
        for e in errors:
            print(f"   - {e}")
        print("!" * 60)
        print(f"\nFix the issues in '{config_yaml}' before proceeding.")
        return False

    missing = []
    if not os.path.exists(cfg['target']['pdb']):
        missing.append(f"Target PDB: {cfg['target']['pdb']}  (put target.pdb in the Project folder)")
    if cfg.get('backbone_generator', 'rfdiffusion').lower() == 'rfdiffusion':
        ckpt = cfg.get('checkpoints', {}).get('rfd')
        if ckpt and not os.path.exists(ckpt):
            missing.append(f"RFdiffusion Checkpoint: {ckpt}")
    if missing:
        print("\n" + "!" * 60)
        print("❌ CRITICAL MISSING FILES:")
        for it in missing:
            print(f"   - {it}")
        print("!" * 60)
        return False
    print("✅ Pre-flight OK.\n")
    return True


# ==============================================================================
# 🎮 MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="🧬 Protein Design Pipeline Orchestrator")
    parser.add_argument('step', choices=['0', '1', '2', '3', '4', '5', 'all',
                                         'doctor', 'init-site', 'setup-envs', 'setup'],
                        help="0=MSA, 1=Backbone, 2=MPNN, 3=Prediction, 4=Score, 5=Dash, all; "
                             "setup=guided one-shot (envs + starter site.yaml); setup-envs=create "
                             "conda envs; doctor=check prerequisites; init-site=seed site.yaml from config")
    parser.add_argument('--config', default=None,
                        help="Explicit config path (default: ./config.yaml in the config folder you cd'd into)")
    parser.add_argument('--create', action='store_true',
                        help="With setup / setup-envs: actually create the missing envs (default: dry-run plan)")
    args = parser.parse_args()

    # Machine-setup actions — no config folder needed; handle before locating one.
    if args.step == 'setup':
        sys.exit(0 if sitecfg.setup(TOOL_ROOT, create=args.create) else 1)
    if args.step == 'setup-envs':
        sys.exit(0 if sitecfg.setup_envs(TOOL_ROOT, create=args.create) else 1)

    # 1. Locate the config folder (cwd convention, or explicit --config escape hatch)
    if args.config:
        config_yaml = os.path.abspath(args.config)
        config_dir = os.path.dirname(config_yaml)
    else:
        config_dir = os.getcwd()
        config_yaml = os.path.join(config_dir, "config.yaml")

    if not os.path.exists(config_yaml):
        print(f"❌ No config.yaml at {config_yaml}")
        print("   cd into your config/run folder  (…/<Project>/<NN-Label>/)  and run again,")
        print("   or pass --config <path>.")
        sys.exit(1)

    # 2. Auto-prefix the config folder if needed (cwd mode only)
    if not args.config:
        config_dir = ensure_config_prefix(config_dir)
        config_yaml = os.path.join(config_dir, "config.yaml")

    project_dir = os.path.dirname(os.path.normpath(config_dir))

    # 3. Load user config; init-site scrapes machine paths straight from it (pre-merge)
    with open(config_yaml) as f:
        user_cfg = yaml.safe_load(f) or {}

    if args.step == 'init-site':
        dest = os.environ.get('BINDERFORGE_SITE') or os.path.expanduser('~/.binderforge/site.yaml')
        if os.path.exists(dest):
            print(f"⚠️  {dest} already exists — not overwriting. Edit it directly or remove it first.")
            sys.exit(1)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'w') as f:
            yaml.safe_dump(sitecfg.scrape_site(user_cfg), f, sort_keys=False)
        print(f"✅ Wrote {dest} — review/edit it, then run:  python {os.path.basename(__file__)} doctor")
        sys.exit(0)

    # Merge the machine layer (site.yaml) in; per-run config wins on any conflict.
    global CONDA_SOURCE
    site, site_path = sitecfg.load_site(TOOL_ROOT)
    user_cfg = sitecfg.merge_site(user_cfg, site)
    CONDA_SOURCE = sitecfg.resolve_conda_source(site)

    # Resolve the convention → a complete config
    cfg = resolve_convention(user_cfg, config_dir, project_dir)
    cfg['conda_source'] = CONDA_SOURCE   # so step shell scripts read it from the resolved config

    if args.step == 'doctor':
        sys.exit(0 if sitecfg.print_doctor(cfg, TOOL_ROOT) else 1)

    # 4. Write the resolved config into outputs/; every step consumes it via $PIPELINE_CONFIG
    outputs_dir = os.path.join(config_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    resolved_path = os.path.join(outputs_dir, ".resolved_config.yaml")
    with open(resolved_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    os.environ["PIPELINE_CONFIG"] = resolved_path

    print("------------------------------------------------")
    print(f"🧬 Run  binder={cfg['binder_name']}  config_id={int(cfg['config_id']):02d}  "
          f"({os.path.basename(os.path.normpath(config_dir))})")
    print(f"   Project : {project_dir}")
    print(f"   Target  : {cfg['target']['pdb']}")
    print(f"   Outputs : {outputs_dir}")
    print("------------------------------------------------")

    if not check_inputs(cfg, config_yaml):
        sys.exit(1)

    # Traceability manifest (date, git, versions, config hash) into outputs/
    manifest_path = write_manifest(cfg, TOOL_ROOT, step=args.step, work_dir=outputs_dir)
    if manifest_path:
        print(f"   🧾 Run manifest: {manifest_path}\n")

    envs = cfg.get('envs', {})
    ENV_RFD   = envs.get('rfd', 'rf3')
    ENV_MPNN  = envs.get('mpnn', 'mlfold')
    ENV_BOLTZ = envs.get('boltz', 'boltz')
    ENV_SCORE = envs.get('score', 'pyrosetta')
    ENV_DASH  = envs.get('dash', 'base')
    ENV_MSA   = envs.get('msa', ENV_DASH)

    def S(name):
        return os.path.join(TOOL_ROOT, "scripts", name)

    # --- STEP 0: TARGET MSA (auto, best-effort) ---
    if args.step in ['0', 'all']:
        if os.path.exists(cfg['target']['msa']):
            print("ℹ️  Target MSA present — skipping step 0.\n")
        else:
            run_step("00_TargetMSA", f"python {S('00_TargetMSA.py')}", ENV_MSA, soft=True)

    # --- STEP 1: BACKBONE GENERATION ---
    if args.step in ['1', 'all']:
        generator = cfg.get('backbone_generator', 'rfdiffusion').lower()
        if generator == 'rfdiffusion':
            print("🦴 Selected Generator: RFDIFFUSION")
            run_step("01_RFdiffusion", f"bash {S('01_RFdiffusion.sh')}", ENV_RFD)
        elif generator == 'complexa':
            print("🦴 Selected Generator: PROTEINA COMPLEXA")
            run_step("01_ProteinaComplexa", f"python {S('01_ProteinaComplexa.py')}", ENV_DASH)
        else:
            print(f"❌ Unknown backbone_generator '{generator}'. Choose 'rfdiffusion' or 'complexa'.")
            sys.exit(1)

    # --- STEP 2: PROTEIN MPNN ---
    if args.step in ['2', 'all']:
        if cfg.get('backbone_generator', 'rfdiffusion').lower() == 'complexa':
            print("ℹ️  backbone_generator=complexa — sequences already designed; MPNN skipped.\n")
        else:
            run_step("02_ProteinMPNN", f"python {S('02_ProteinMPNN.py')}", ENV_MPNN)

    # --- STEP 3: STRUCTURE PREDICTION ---
    if args.step in ['3', 'all']:
        predictor = cfg.get('predictor', 'boltz').lower()
        if predictor == 'boltz':
            print("🧠 Selected Engine: BOLTZ")
            run_step("03_Boltz", f"bash {S('03_Boltz.sh')}", ENV_BOLTZ)
        elif predictor == 'alphafast':
            print("🧠 Selected Engine: ALPHAFAST")
            run_step("03_AlphaFast", f"python {S('03_AlphaFast.py')}", ENV_DASH)
        else:
            print(f"❌ Unknown predictor '{predictor}'. Choose 'boltz' or 'alphafast'.")
            sys.exit(1)

    # --- STEP 4: SCORING ---
    if args.step in ['4', 'all']:
        run_step("04_Scoring", f"python {S('04_Scoring.py')} --config {resolved_path}", ENV_SCORE)

    # --- STEP 5: DASHBOARD ---
    if args.step in ['5', 'all']:
        run_step("05_Dashboard", f"python {S('05_Dashboard.py')} --config {resolved_path}", ENV_DASH)

    # --- RUN SUMMARY (readable digest of parameters + output counts) ---
    try:
        summary_path = write_run_summary(cfg, outputs_dir, step=args.step, project_dir=project_dir)
        if summary_path:
            print(f"   📝 Run summary: {summary_path}")
    except Exception as e:
        print(f"   ⚠️  Could not write run summary: {e}")

    print("\n✨ PIPELINE EXECUTION COMPLETE ✨")


if __name__ == "__main__":
    main()
