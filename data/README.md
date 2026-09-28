# Inputs for the physics case

Place these files in this directory (or set `LATENTCAT_DATA`). Nothing here is committed.

| file | what | used by |
|---|---|---|
| `ggzz4l_bkg.csv`, `ggzz4l_sbi.csv`, `ggzz4l_sig.csv` | parton-level gg → ZZ → 4ℓ samples (background, signal + background + interference, signal), with per-event SM squared matrix elements `msq_{bkg,int,sig,sbi}_sm` and generator weight `wt` | `scripts/physics_cache_csv.py` (once) |
| `physics_cache_{bkg,sbi,sig}.npz` | compact cache written by `physics_cache_csv.py` | all physics scripts |

The Gaussian toy needs no inputs.
