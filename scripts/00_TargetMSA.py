"""
==============================================================================
   🧬 STEP 00: TARGET MSA (auto, best-effort)
==============================================================================
   If the project-level target MSA is missing, generate it once from the target
   PDB via colabfold_search, and place it at the path the pipeline expects
   (project/target_msa.a3m). Reused by every config under the same project.

   Best-effort: if the MSA already exists, or colabfold_search / the DB are not
   available, it warns and exits non-zero (the orchestrator runs this soft).

   Reads the resolved config from $PIPELINE_CONFIG.
==============================================================================
"""
import os
import sys
import glob
import shutil
import subprocess
import yaml

try:
    from Bio import SeqIO
except ImportError:
    SeqIO = None

CONFIG = os.environ.get("PIPELINE_CONFIG", "config.yaml")


def target_sequence(pdb_path):
    if SeqIO is None:
        return None
    for record in SeqIO.parse(pdb_path, "pdb-atom"):
        return str(record.seq)
    return None


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    pdb = cfg['target']['pdb']
    msa_out = cfg['target']['msa']

    if os.path.exists(msa_out):
        print(f"✅ Target MSA already exists: {msa_out}")
        return 0
    if not os.path.exists(pdb):
        print(f"❌ Target PDB not found: {pdb}")
        return 1

    db = cfg.get('tools', {}).get('msa_db')
    if not db:
        print("⚠️  tools.msa_db not set — cannot auto-generate the MSA.")
        print("    Set tools.msa_db, or place the MSA at:", msa_out)
        return 1

    seq = target_sequence(pdb)
    if not seq:
        print("⚠️  Could not extract target sequence from the PDB (need Biopython).")
        return 1

    proj_dir = os.path.dirname(os.path.abspath(msa_out))
    fasta = os.path.join(proj_dir, "target.fasta")
    with open(fasta, "w") as f:
        f.write(f">target\n{seq}\n")

    search_out = os.path.join(proj_dir, "_msa_search")
    os.makedirs(search_out, exist_ok=True)
    print(f"🔎 Running colabfold_search on the target ({len(seq)} aa)…")
    try:
        subprocess.run(["colabfold_search", fasta, db, search_out], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  colabfold_search failed or unavailable: {e}")
        print("    Generate the MSA manually and save it to:", msa_out)
        return 1

    a3ms = sorted(glob.glob(os.path.join(search_out, "*.a3m")))
    if not a3ms:
        print("⚠️  colabfold_search produced no .a3m file.")
        return 1

    shutil.copy(a3ms[0], msa_out)
    print(f"✅ Target MSA written: {msa_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
