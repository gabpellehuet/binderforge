"""
==============================================================================
   🧾 PROVENANCE / RUN MANIFEST
==============================================================================
   Stamps every pipeline run with the information needed to reproduce it:
   date, git commit of the pipeline code, software versions, host, the step
   that ran, and a hash of the exact config used.

   Writes/append to  {work_dir}/run_manifest.json  as a growing "runs" list, so
   a binder_id always traces back to the code + config that produced it.

   Usage:
     Imported:    from provenance import write_manifest
                  write_manifest(cfg, repo_dir, step="all")
     Standalone:  python scripts/provenance.py [config.yaml]   # prints provenance
==============================================================================
"""
import os
import sys
import json
import platform
import subprocess
import hashlib
import datetime


def _git_info(repo_dir):
    """Return git provenance for the pipeline code, or {is_repo: False}."""
    def run(args):
        try:
            return subprocess.check_output(
                ['git', '-C', repo_dir] + args,
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    if run(['rev-parse', '--is-inside-work-tree']) != 'true':
        return {"is_repo": False}
    return {
        "is_repo": True,
        "commit": run(['rev-parse', 'HEAD']),
        "short": run(['rev-parse', '--short', 'HEAD']),
        "branch": run(['rev-parse', '--abbrev-ref', 'HEAD']),
        "dirty": bool(run(['status', '--porcelain'])),
        "remote": run(['config', '--get', 'remote.origin.url']),
        "last_commit_date": run(['log', '-1', '--format=%cI']),
    }


def _pkg_versions():
    mods = ["numpy", "pandas", "Bio", "yaml", "plotly", "scipy", "pyrosetta", "openpyxl"]
    out = {}
    for m in mods:
        try:
            out[m] = getattr(__import__(m), "__version__", "unknown")
        except Exception:
            out[m] = None
    return out


def _config_hash(cfg):
    try:
        blob = json.dumps(cfg, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return None


def collect_provenance(cfg, repo_dir, step=None):
    """Build one provenance entry (a dict) for a single pipeline invocation."""
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "step": step,
        "binder_name": cfg.get("binder_name") or cfg.get("project_name"),
        "config_id": cfg.get("config_id", cfg.get("run_id")),
        "config_label": cfg.get("config_label"),
        "binder_id_prefix": f"{cfg.get('binder_name') or cfg.get('project_name')}"
                            f"-{int(cfg.get('config_id', cfg.get('run_id', 0)) or 0):02d}",
        "backbone_generator": cfg.get("backbone_generator"),
        "predictor": cfg.get("predictor"),
        "config_sha256_16": _config_hash(cfg),
        "git": _git_info(repo_dir),
        "python": platform.python_version(),
        "packages": _pkg_versions(),
        "host": platform.node(),
        "platform": platform.platform(),
        "argv": " ".join(sys.argv),
    }


def write_manifest(cfg, repo_dir, step=None, work_dir=None, path=None):
    """Append a provenance entry to {work_dir}/run_manifest.json. Returns the path.

    Never raises into the pipeline — provenance failures must not kill a run.
    """
    try:
        work_dir = work_dir or cfg.get("work_dir") or "."
        os.makedirs(work_dir, exist_ok=True)
        path = path or os.path.join(work_dir, "run_manifest.json")

        runs = []
        if os.path.exists(path):
            try:
                runs = json.load(open(path)).get("runs", [])
            except Exception:
                runs = []
        runs.append(collect_provenance(cfg, repo_dir, step))

        data = {
            "binder_name": cfg.get("binder_name") or cfg.get("project_name"),
            "config_id": cfg.get("config_id", cfg.get("run_id")),
            "config_label": cfg.get("config_label"),
            "runs": runs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
    except Exception as e:
        print(f"   ⚠️  Could not write run manifest: {e}")
        return None


def main():
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(collect_provenance(cfg, repo_dir, step="(standalone)"), indent=2))


if __name__ == "__main__":
    main()
