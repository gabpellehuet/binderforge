# Template project — copy me

This folder is a ready-to-copy scaffold for one **target**. To start a real run:

1. **Copy & rename** this folder to your target's name, e.g. `PIV3/` (the folder name
   becomes `binder_name`, the prefix of every `binder_id`).
2. **Replace `target.pdb`** with your real target structure. (`target_msa.a3m` is
   created automatically in step 0 if it isn't already here.)
3. **Edit `00-Label/config.yaml`** — set every `# TODO` field (design knobs and, for
   now, the machine paths). Rename `00-Label` if you like; keep the `NN-` prefix
   (`NN` = `config_id`, auto-added on first run if missing). Add more `NN-Label/`
   config folders to run different settings against the same target.
4. **Run** from inside the config folder:
   ```bash
   cd PIV3/00-Label
   python /path/to/ProteinBinderDesign/Run_Pipeline.py all   # or a step: 0..5
   ```

Layout:
```
ProjectName/            <- rename to your target (= binder_name)
├── target.pdb          <- replace with your target
├── target_msa.a3m      <- auto-made in step 0 (optional to provide)
└── 00-Label/           <- one config/run (NN- = config_id)
    ├── config.yaml     <- edit the TODOs
    └── outputs/        <- created on run
```
