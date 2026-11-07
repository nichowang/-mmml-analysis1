"""Cross-session probe for the two evaluation participants recorded in two sessions.

For every 10 s query window of P0047 / P0048, distance (z-scored) from the query to the
block-mean of each other recording, grouped as same person + same session, same person +
other session, and other people (all other evaluation participants). Reported for static
posture and for the handcrafted descriptors.

Usage: python idea2_session_probe.py CACHE_DIR OUT_JSON
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis1 import MIN_VALID, parse_id  # noqa: E402

cache, out = sys.argv[1], sys.argv[2]
ev = {os.path.basename(f)[:-4] for f in glob.glob("data/0000_000[0-9]/*.npz")}
R = {}
for f in sorted(glob.glob(os.path.join(cache, "*.npz"))):
    rid = os.path.basename(f)[:-4]
    if rid not in ev:
        continue
    z = np.load(f)
    good = (z["vf"] >= MIN_VALID) & (z["vm"] >= MIN_VALID) & ~np.isnan(z["desc"]).any(1)
    pm = z["pmask"][..., None]
    post = (z["pose"].astype(np.float32) * pm).sum(1) / np.maximum(pm.sum(1), 1)
    R[rid] = dict(**parse_id(rid), good=np.flatnonzero(good), posture=post, desc=z["desc"])
res = {}
for feat in ("posture", "desc"):
    A = np.concatenate([R[r][feat][R[r]["good"]] for r in R])
    mu, sd = A.mean(0), A.std(0) + 1e-6
    M = {r: ((R[r][feat][R[r]["good"]] - mu) / sd).mean(0) for r in R if len(R[r]["good"])}
    for p in ("P0047", "P0048"):
        d = defaultdict(list)
        for q in [r for r in M if R[r]["participant"] == p]:
            g = set(R[q]["good"])
            X = (R[q][feat] - mu) / sd
            for w in range(len(X) // 2):
                if 2 * w in g and 2 * w + 1 in g:
                    xq = X[2 * w:2 * w + 2].mean(0)
                    for r in M:
                        if r == q:
                            continue
                        if R[r]["participant"] == p:
                            k = "same_session" if R[r]["session"] == R[q]["session"] else "other_session"
                        else:
                            k = "other_people"
                        d[k].append(float(np.linalg.norm(xq - M[r])))
        res[f"{feat}|{p}"] = {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in d.items()}
        res[f"{feat}|{p}"]["sessions"] = sorted({R[r]["session"] for r in M if R[r]["participant"] == p})
        res[f"{feat}|{p}"]["frac_other_people_closer_than_other_session"] = float(
            np.mean(np.array(d["other_people"]) < np.mean(d["other_session"])))
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print(k, {kk: (round(vv["mean"], 2) if isinstance(vv, dict) else vv) for kk, vv in v.items()})
