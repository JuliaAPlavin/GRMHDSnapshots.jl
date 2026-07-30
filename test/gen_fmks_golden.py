#!/usr/bin/env python3
"""Regenerate the pyharm golden values used by the "iharm FMKS loader" testitem.

pyharm (https://github.com/AFD-Illinois/pyharm) is an independent reference reader for iharm3d/KHARMA
dumps. We bake a handful of its per-cell results into the Julia test as constants so the test needs no
Python at run time; this script documents exactly how those constants were produced.

Setup (avoid pyharm's `yt`/`hallmark` git submodules — not needed for reading dumps):

    python3 -m venv venv && . venv/bin/activate
    pip install numpy scipy h5py pandas matplotlib click tqdm psutil
    pip install --no-deps "https://github.com/AFD-Illinois/pyharm/archive/refs/heads/master.tar.gz"

Then:  python3 gen_fmks_golden.py data/iharm_fmks_a0.h5

Cells are 0-based native indices (i=radial, j=theta, k=phi). `bsq`, `sigma`, `Gamma` are
coordinate-frame-invariant scalars, so they compare directly against the KS-frame reconstruction.
"""
import sys
import pyharm

CELLS = [(59, 63, 9), (119, 39, 69), (29, 99, 4), (199, 109, 119)]

d = pyharm.load_dump(sys.argv[1] if len(sys.argv) > 1 else "data/iharm_fmks_a0.h5")

def at(name, i, j, k):
    a = d[name]
    return float(a[i, j, 0] if a.shape[2] == 1 else a[i, j, k])

print("# (i, j, k, r, th, RHO, bsq, sigma, Gamma) — 0-based native indices")
for (i, j, k) in CELLS:
    vals = [at(n, i, j, k) for n in ("r", "th", "RHO", "bsq", "sigma", "Gamma")]
    print(f"({i},{j},{k}, " + ", ".join(f"{v:.12g}" for v in vals) + "),")
