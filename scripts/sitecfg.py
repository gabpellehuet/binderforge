"""
==============================================================================
   🖥️  SITE CONFIG  (machine layer — install locations, kept out of run config)
==============================================================================
   A run's config.yaml holds only PORTABLE design parameters. Everything that is
   specific to THIS machine — conda source, per-tool conda env names, and the
   paths to installed tools / model weights / databases — lives in a separate
   `site.yaml`, written once per install. This module finds it, merges it into
   the run config (so every step still just reads `cfg[...]`), resolves the conda
   source, and provides the `doctor` preflight.

   site.yaml resolution order (first that exists wins):
     1. $BINDERDESIGN_SITE
     2. <tool root>/site.yaml
     3. ~/.binderdesign/site.yaml

   MERGE RULE: site fills machine keys; the per-run config WINS on any conflict,
   so existing run folders that still carry machine paths inline keep working.
   site.yaml mirrors the config key layout (envs:, tools:, checkpoints:,
   step1_complexa:, step3_alphafast:, gpu_devices) so the merge is a deep merge
   and no step script needs to change.
==============================================================================
"""
import os
import glob
import subprocess

try:
    import yaml
except ImportError:                       # keep import-safe for tooling
    yaml = None

# Machine-level keys that site.yaml is expected to provide (for docs / init-site).
SITE_KEYS = (
    'conda_source', 'envs', 'gpu_devices',
    'tools.mpnn_script', 'tools.msa_db', 'tools.pymol', 'tools.af_input_dir',
    'checkpoints.rfd',
    'step1_complexa.complexa_dir', 'step1_complexa.docker_image', 'step1_complexa.af2_params_dir',
    'step3_alphafast.alphafast_dir', 'step3_alphafast.db_dir', 'step3_alphafast.weights_dir',
)

_LEGACY_CONDA_SOURCE = "/usr/local/miniforge3/etc/profile.d/conda.sh"


# ── Locate & load ─────────────────────────────────────────────────────────────
def find_site_path(tool_root):
    """Return the first existing site.yaml path (env → tool root → home), or None."""
    for cand in (os.environ.get('BINDERDESIGN_SITE'),
                 os.path.join(tool_root, 'site.yaml'),
                 os.path.expanduser('~/.binderdesign/site.yaml')):
        if cand and os.path.exists(cand):
            return cand
    return None


def load_site(tool_root):
    """Return (site_dict, path_or_None)."""
    path = find_site_path(tool_root)
    if not path or yaml is None:
        return {}, path
    with open(path) as f:
        return (yaml.safe_load(f) or {}), path


# ── Merge ─────────────────────────────────────────────────────────────────────
def _deep_merge(base, override):
    """Return base updated by override; override wins on every leaf."""
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def merge_site(cfg, site):
    """Deep-merge site into cfg with cfg winning (site only fills what's absent)."""
    return _deep_merge(site or {}, cfg or {})


# ── Conda source ──────────────────────────────────────────────────────────────
def resolve_conda_source(site):
    """Resolve the conda.sh to source: explicit site/env value, else auto-detect."""
    val = os.environ.get('BINDERDESIGN_CONDA_SOURCE') or (site or {}).get('conda_source')
    if val and val != 'auto':
        return os.path.expanduser(val)
    exe = os.environ.get('CONDA_EXE')             # …/bin/conda → …/etc/profile.d/conda.sh
    if exe:
        cand = os.path.join(os.path.dirname(os.path.dirname(exe)), 'etc/profile.d/conda.sh')
        if os.path.exists(cand):
            return cand
    for base in ('/usr/local/miniforge3', '~/miniforge3', '~/miniconda3',
                 '/opt/conda', '~/anaconda3', '/opt/miniforge3'):
        cand = os.path.expanduser(os.path.join(base, 'etc/profile.d/conda.sh'))
        if os.path.exists(cand):
            return cand
    return _LEGACY_CONDA_SOURCE                    # last-resort legacy default


# ── init-site ─────────────────────────────────────────────────────────────────
def scrape_site(cfg):
    """Pull the machine-level slice out of a (pre-merge) config dict, for init-site."""
    site = {}
    if 'envs' in cfg:        site['envs'] = cfg['envs']
    if 'gpu_devices' in cfg: site['gpu_devices'] = cfg['gpu_devices']
    if 'tools' in cfg:       site['tools'] = dict(cfg['tools'])
    if 'checkpoints' in cfg: site['checkpoints'] = dict(cfg['checkpoints'])
    for blk, keys in (('step1_complexa', ('complexa_dir', 'docker_image', 'af2_params_dir')),
                      ('step3_alphafast', ('alphafast_dir', 'db_dir', 'weights_dir'))):
        sub = {k: cfg[blk][k] for k in keys if isinstance(cfg.get(blk), dict) and k in cfg[blk]}
        if sub:
            site[blk] = sub
    site.setdefault('conda_source', 'auto')
    return site


# ── Env recipes / setup-envs bootstrap ────────────────────────────────────────
def plan_envs(tool_root):
    """Return [(recipe_path, env_name, already_exists)] for envs/*.yml (read-only)."""
    recipes = sorted(glob.glob(os.path.join(tool_root, 'envs', '*.yml')))
    present = _conda_envs(resolve_conda_source(load_site(tool_root)[0]))
    rows = []
    for r in recipes:
        name = None
        if yaml is not None:
            try:
                name = (yaml.safe_load(open(r)) or {}).get('name')
            except Exception:
                name = None
        name = name or os.path.splitext(os.path.basename(r))[0]
        rows.append((r, name, name in present))
    return rows


def setup_envs(tool_root, create=False):
    """Print a create plan for envs/*.yml; with create=True, run `conda env create`
    for any that don't exist yet. Returns True on success."""
    rows = plan_envs(tool_root)
    if not rows:
        print("No env recipes found in envs/.")
        return False
    conda_source = resolve_conda_source(load_site(tool_root)[0])
    print("🧩 setup-envs — conda env recipes in envs/\n")
    to_make = []
    for recipe, name, exists in rows:
        if exists:
            print(f"   ✅ {name:<12} already present — skipping")
        else:
            to_make.append((recipe, name))
            cmd = f"conda env create -f {os.path.relpath(recipe, tool_root)}"
            print(f"   {'⏳ creating' if create else '➕ would create'} {name:<12} ({cmd})")
    ok = True
    if create:
        for recipe, name in to_make:
            print(f"\n── conda env create -f {recipe} ──")
            r = subprocess.run(f"source {conda_source} && conda env create -f {recipe}",
                               shell=True, executable='/bin/bash')
            ok = ok and (r.returncode == 0)
    elif to_make:
        print("\n   (dry run — re-run with `setup-envs --create` to create the missing envs)")
    print("\n📄 Post-create pip/git/license steps per env: see envs/README.md")
    print("   Then set paths in site.yaml and run: doctor")
    return ok


# ── Doctor ────────────────────────────────────────────────────────────────────
def _conda_envs(conda_source):
    try:
        r = subprocess.run(f"source {conda_source} && conda env list",
                           shell=True, executable='/bin/bash',
                           capture_output=True, text=True, timeout=30)
        return {ln.split()[0] for ln in r.stdout.splitlines()
                if ln.strip() and not ln.startswith('#')}
    except Exception:
        return set()


def _which(binary):
    return subprocess.run(['bash', '-lc', f'command -v {binary}'],
                          capture_output=True, text=True).returncode == 0


def doctor(cfg, tool_root):
    """Check the enabled steps' env/tool/model prerequisites.

    Returns (all_ok, rows) where each row is (status, label, detail, site_key).
    status ∈ {'OK', 'MISS', 'WARN'}.
    """
    site, site_path = load_site(tool_root)
    conda_source = resolve_conda_source(site)
    gen  = str(cfg.get('backbone_generator', 'rfdiffusion')).lower()
    pred = str(cfg.get('predictor', 'boltz')).lower()
    envs = cfg.get('envs', {})
    tools = cfg.get('tools', {})
    rows = []

    def path_row(label, value, key, required=True):
        if not value:
            rows.append(('MISS' if required else 'WARN', label, '(unset)', key))
        elif os.path.exists(os.path.expanduser(str(value))):
            rows.append(('OK', label, str(value), key))
        else:
            rows.append(('MISS' if required else 'WARN', label, f'not found: {value}', key))

    # site file + conda
    rows.append((('OK' if site_path else 'WARN'), 'site.yaml',
                 site_path or 'none found (run: init-site)', ''))
    path_row('conda source', conda_source, 'conda_source')

    # conda envs needed for the enabled path
    present = _conda_envs(conda_source)
    need = {'score': envs.get('score', 'pyrosetta'), 'dash': envs.get('dash', 'base'),
            'msa': envs.get('msa', 'msa_tools')}
    if gen == 'rfdiffusion':
        need['rfd'] = envs.get('rfd', 'rf3'); need['mpnn'] = envs.get('mpnn', 'mlfold')
    if pred == 'boltz':
        need['boltz'] = envs.get('boltz', 'boltz')
    for role, name in need.items():
        rows.append((('OK' if name in present else 'MISS'),
                     f'conda env [{role}]', name, f'envs.{role}'))

    # host binaries
    if gen == 'complexa' or pred == 'alphafast':
        rows.append((('OK' if _which('docker') else 'MISS'), 'docker', 'on PATH', ''))
    pymol = tools.get('pymol')
    if pymol:
        path_row('pymol', pymol, 'tools.pymol', required=False)
    else:
        rows.append((('OK' if _which('pymol') else 'WARN'), 'pymol', 'on PATH', 'tools.pymol'))

    # model / data paths (only those the enabled path needs)
    path_row('ProteinMPNN script', tools.get('mpnn_script'), 'tools.mpnn_script',
             required=(gen == 'rfdiffusion'))
    path_row('MSA database', tools.get('msa_db'), 'tools.msa_db', required=False)
    if gen == 'rfdiffusion':
        path_row('RFdiffusion checkpoint', cfg.get('checkpoints', {}).get('rfd'), 'checkpoints.rfd')
    if gen == 'complexa':
        cx = cfg.get('step1_complexa', {})
        path_row('Complexa dir', cx.get('complexa_dir'), 'step1_complexa.complexa_dir')
        path_row('AF2 params', cx.get('af2_params_dir'), 'step1_complexa.af2_params_dir', required=False)
    if pred == 'alphafast':
        af = cfg.get('step3_alphafast', {})
        path_row('AlphaFast dir', af.get('alphafast_dir'), 'step3_alphafast.alphafast_dir')
        path_row('AlphaFast DB', af.get('db_dir'), 'step3_alphafast.db_dir')
        path_row('AF3 weights', af.get('weights_dir'), 'step3_alphafast.weights_dir')

    all_ok = all(s != 'MISS' for s, *_ in rows)
    return all_ok, rows


def print_doctor(cfg, tool_root):
    ok, rows = doctor(cfg, tool_root)
    icon = {'OK': '✅', 'MISS': '❌', 'WARN': '⚠️ '}
    print("🩺 binderdesign doctor — checking enabled steps\n")
    for status, label, detail, key in rows:
        tail = f"   → set `{key}` in site.yaml" if status == 'MISS' and key else ''
        print(f"   {icon[status]} {label:<24} {detail}{tail}")
    print("\n" + ("✅ All required prerequisites present." if ok
                   else "❌ Missing prerequisites above — fix them in site.yaml, then re-run doctor."))
    return ok
