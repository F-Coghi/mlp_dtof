# This script reads a table (legenda.txt) listing .mat simulation files and six parameters per file,
# loads each .mat file, extracts the variable `output_all` as a 1D array, and saves one .npz per file
# containing the flattened output vector and the associated 6 parameters.

import os
import numpy as np
import scipy.io

TAB_PATH = "../sim_dtof_milano/legenda.txt"
MAT_DIR = "../sim_dtof_milano/DatasetPoisson/"
VAR_NAME = "output_all"

results = {}  # filename -> {"params": [p1..p6], "output": np.ndarray}

with open(TAB_PATH, "r") as fh:
    next(fh)
    for ln in fh:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        fname = parts[0]
        if not fname.endswith(".mat"):
            fname += ".mat"
        params = [float(x) for x in parts[1:7]]  # 6 params

        mat_path = os.path.join(MAT_DIR, fname)
        if not os.path.exists(mat_path):
            print(f"WARNING: missing file {mat_path}, skipping.")
            continue

        data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)

        if VAR_NAME not in data:
            print(f"WARNING: '{VAR_NAME}' not found in {fname}, keys: {list(data.keys())}")
            continue

        # flatten to 1D (remove those extra brackets)
        out = np.asarray(data[VAR_NAME]).ravel()

        results[fname] = {
            "params": params,   # [p1, p2, p3, p4, p5, p6]
            "output": out       # 1D numpy array
        }

# Example: access one entry
#for k, v in list(results.items())[:1]:
#    print(f"\n{k}")
#    print("params:", v["params"])
#    print("output shape:", v["output"].shape)
#    print("first 10:", v["output"][:10])

SAVE_DIR = "../ProcessedPoisson"
os.makedirs(SAVE_DIR, exist_ok=True)
for fname, payload in results.items():
    base = os.path.splitext(fname)[0]
    np.savez(os.path.join(SAVE_DIR, f"{base}.npz"),
             output=payload["output"], params=np.array(payload["params"]))