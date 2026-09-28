"""One-time conversion of data/ggzz4l_{sig,bkg,sbi}.csv (17 GB, 208 columns) to data/physics_cache_{proc}.npz,
keeping only what the code uses: the 17 encoder features, the four SM squared matrix elements and the generator
weight `wt`."""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latentcat.common import data_dir
from latentcat.physics_data import FEATS

MSQ = ["msq_bkg_sm", "msq_int_sm", "msq_sig_sm", "msq_sbi_sm"]
for proc in ("sig", "bkg", "sbi"):
    t0 = time.time()
    use = FEATS + MSQ + ["wt"]
    chunks = pd.read_csv(os.path.join(data_dir(), f"ggzz4l_{proc}.csv"), usecols=use, chunksize=500000, engine="c")
    X = np.concatenate([ch[use].to_numpy(np.float32) for ch in chunks])
    wt = X[:, use.index("wt")]
    np.savez(os.path.join(data_dir(), f"physics_cache_{proc}.npz"), X=X, columns=np.array(use), weights=wt)
    print(
        f"{proc}: {X.shape} cached in {(time.time()-t0)/60:.1f} min; sum wt = {wt.astype(np.float64).sum():.5f}; "
        f"rows with NaN = {int(np.isnan(X).any(1).sum())}",
        flush=True,
    )
