"""Per-block feature cache for the Idea 2 midterm baseline.

For every recording and every 5 s block it stores:
  desc / desc_jit   7 handcrafted descriptors (analysis1/2), clean and jittered (0.05 rad)
  si_latent         mean Seamless Interaction movement frame_latent (64)
  si_expr           mean expression code + FAU values (128 + 24)
  si_hyper          mean hypernet features (5120); nearly constant within a recording,
                    so it acts as an appearance / identity control rather than motion
  pose / pose_jit   upper-body pose at 10 fps for the learned encoder: 13 joint rotations
                    (6D) + chest-frame wrist positions, (50, 84), clean and jittered
  sf, vf, vm        speaking fraction, SMPL-H valid fraction, movement valid fraction

Usage: python idea2_features.py DATA_DIR CACHE_DIR [archive_glob] [n_proc]
"""
import glob
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis1 import BLOCK, MIN_VALID, fk, speaking_mask  # noqa: E402
from analysis2 import block_feats, tf  # noqa: E402

SIG = 0.05                 # jitter (rad), = analysis2 "jit_hi"
SUB = 3                    # 30 fps -> 10 fps
UPPER = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]   # SMPL joints (body_pose index j-1)


def pose_seq(bp, rel):
    T = len(bp)
    rot = R.from_rotvec(bp[:, [j - 1 for j in UPPER]].reshape(-1, 3)).as_matrix()
    six = rot[..., :, :2].reshape(T, len(UPPER) * 6)
    return np.concatenate([six, rel.reshape(T, 6)], 1)


def one(args):
    npz, out = args
    rid = os.path.basename(npz)[:-4]
    dst = os.path.join(out, rid + ".npz")
    if os.path.exists(dst):
        return rid, "cached"
    js = npz[:-4] + ".json"
    if not os.path.exists(js):
        return rid, "no json"
    z = np.load(npz)
    valid = z["smplh:is_valid"].astype(bool)
    vm = z["movement:is_valid"][:, 0] > 0
    bp = z["smplh:body_pose"].astype(np.float64)
    go = z["smplh:global_orient"].astype(np.float64)
    T = len(valid)
    speak = speaking_mask(json.load(open(js))["metadata:vad"], T)
    rel, head = fk(bp, go)
    rng = np.random.default_rng(zlib.crc32(rid.encode()))
    bpj = bp + rng.normal(0, SIG, bp.shape)
    relj, headj = fk(bpj, go + rng.normal(0, SIG, go.shape))
    P, PJ = pose_seq(bp, rel), pose_seq(bpj, relj)
    lat = z["movement:frame_latent"]
    expr = np.concatenate([z["movement:expression"], z["movement:FAUValue"]], 1)
    hyp = z["movement:hypernet_features"]
    hstep = T / max(len(hyp), 1)
    nb = T // BLOCK
    rec = {k: [] for k in ["desc", "desc_jit", "si_latent", "si_expr", "si_hyper",
                           "pose", "pose_jit", "pmask", "sf", "vf", "vm"]}
    for b in range(nb):
        s = slice(b * BLOCK, (b + 1) * BLOCK)
        v = valid[s]
        vf = v.mean()
        ok = vf >= MIN_VALID
        rec["desc"].append(tf(block_feats(rel, head, valid, s)) if ok else np.full(7, np.nan))
        rec["desc_jit"].append(tf(block_feats(relj, headj, valid, s)) if ok else np.full(7, np.nan))
        m = vm[s]
        rec["vm"].append(m.mean())
        rec["si_latent"].append(lat[s][m].mean(0) if m.any() else np.full(64, np.nan))
        rec["si_expr"].append(expr[s][m].mean(0) if m.any() else np.full(152, np.nan))
        h0, h1 = int(b * BLOCK / hstep), max(int((b + 1) * BLOCK / hstep), int(b * BLOCK / hstep) + 1)
        rec["si_hyper"].append(hyp[h0:h1].mean(0) if len(hyp[h0:h1]) else np.full(hyp.shape[1], np.nan))
        idx = np.arange(b * BLOCK, (b + 1) * BLOCK, SUB)
        rec["pose"].append(P[idx] * valid[idx, None])
        rec["pose_jit"].append(PJ[idx] * valid[idx, None])
        rec["pmask"].append(valid[idx])
        rec["sf"].append(speak[s].mean())
        rec["vf"].append(vf)
    arr = {k: np.asarray(v, np.float16 if k in ("si_hyper", "pose", "pose_jit") else np.float32)
           for k, v in rec.items()}
    arr["pmask"] = arr["pmask"].astype(bool)
    np.savez(dst, **arr)
    return rid, nb


if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else "*"
    npr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data, pat, "*.npz")))
    with Pool(npr) as pool:
        for i, (rid, r) in enumerate(pool.imap_unordered(one, [(f, out) for f in files])):
            if i % 20 == 0:
                print(i, len(files), rid, r, flush=True)
    print("done", len(files), flush=True)
