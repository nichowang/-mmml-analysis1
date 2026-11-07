"""Analysis 2: Multi-reference composition for Idea 2 (descriptor level, no model).

Same descriptors, query windows, leave-one-participant-out scaling and speaking-
fraction matching as analysis1.py. A reference SET is a few 5 s blocks of one person;
its descriptor is the equal-weight mean of its blocks (clip boundaries respected:
every block descriptor is computed inside its own block). For each query window,
Delta = mean distance to matched different-person sets (built the same way, equal
weight per person) - distance to the same-person set. Participant means, bootstrap
over participants, paired differences between conditions.

Experiments
  A. Fixed total duration D = 20 s: 1x20 s, 2x10 s, 4x5 s, drawn from 1, 2 or 3 of the
     person's other recordings.
  B. Duration sweep at a fixed count (1 clip / 2 clips from 2 recordings), 5-20 s per clip.
  C. Selection policy at a fixed count: uniform random vs greedy coverage (farthest-
     point on reference-side velocity, amplitude and head motion), same candidate pool.
  D. Saturation: k = 1..16 random 5 s blocks from the person's other recordings.
  E. Low-quality references: 4 good blocks + m added blocks that are good, jittered,
     tracking-degraded, or from a different person.
  F. Same-recording upper bound (references from the query's own recording, >=10 s
     away), reported separately.

Usage: python analysis2.py DATA_DIR OUT_DIR [archive_glob]
"""
import glob
import json
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis1 import (BLOCK, FEATS, FPS, LOG_FEATS, MIN_VALID, N_BOOT, TOL, fk,  # noqa: E402
                       parse_id, speaking_mask)

DEG_MIN = 0.5            # degraded block: valid fraction in [DEG_MIN, MIN_VALID)
GAP = 2                  # >=10 s between clips taken from one recording
JITTER = {"jit_lo": 0.02, "jit_hi": 0.05}   # rad, iid per-frame rotation noise
N_POS = 60               # positive draws per (person, query recording)
N_POS_OWN = 20           # positive draws per query (same-recording conditions)
N_NEG = 100              # negative sets per person and condition
MIN_RECS = 4             # main subset: >=4 eligible recordings (>=3 other recordings)
COV_DIMS = ([0, 1], [2, 3], [6])   # velocity, amplitude, head motion
RNG = np.random.default_rng(0)


def boot_ci(x, n=N_BOOT):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            float((m <= 0).mean()), int(len(x))]


def block_feats(rel, head, valid, s):
    v = valid[s]
    w = rel[s]
    pair = v[1:] & v[:-1]
    sp = np.linalg.norm(np.diff(w, axis=0), axis=-1)[pair] * FPS
    hq = head[s]
    dq = (hq[:-1].inv() * hq[1:]).magnitude()[pair] * FPS
    wv = w[v]
    exc = np.sqrt(((wv - wv.mean(0)) ** 2).sum(-1).mean(0))
    hh = wv[..., 1].mean(0)
    return np.array([sp[:, 0].mean(), sp[:, 1].mean(), exc[0], exc[1], hh[0], hh[1], dq.mean()])


def rec_blocks(npz_path, json_path):
    """Per block: clean feats (vf >= DEG_MIN), jittered feats (good blocks), sf, vf."""
    z = np.load(npz_path)
    valid = z["smplh:is_valid"].astype(bool)
    bp = z["smplh:body_pose"].astype(np.float64)
    go = z["smplh:global_orient"].astype(np.float64)
    T = len(valid)
    speak = speaking_mask(json.load(open(json_path))["metadata:vad"], T)
    rel, head = fk(bp, go)
    noisy = {}
    rng = np.random.default_rng(zlib.crc32(os.path.basename(npz_path).encode()))
    for k, sig in JITTER.items():
        noisy[k] = fk(bp + rng.normal(0, sig, bp.shape), go + rng.normal(0, sig, go.shape))
    out = []
    for b in range(T // BLOCK):
        s = slice(b * BLOCK, (b + 1) * BLOCK)
        vf = valid[s].mean()
        d = dict(b=b, sf=float(speak[s].mean()), vf=float(vf), f=None)
        if vf >= DEG_MIN:
            d["f"] = block_feats(rel, head, valid, s)
        if vf >= MIN_VALID:
            for k in JITTER:
                d[k] = block_feats(noisy[k][0], noisy[k][1], valid, s)
        out.append(d)
    return out


def tf(f):
    g = f.copy()
    g[LOG_FEATS] = np.log(np.maximum(g[LOG_FEATS], 1e-6))
    return g


def main(data_dir, out_dir, pattern="*"):
    os.makedirs(out_dir, exist_ok=True)
    recs = {}
    for npz in sorted(glob.glob(os.path.join(data_dir, pattern, "*.npz"))):
        js = npz[:-4] + ".json"
        if os.path.exists(js):
            rid = os.path.basename(npz)[:-4]
            recs[rid] = dict(blocks=rec_blocks(npz, js), **parse_id(rid))
    print(f"{len(recs)} recordings", flush=True)

    # ---- good blocks (as analysis1), degraded blocks, jittered blocks
    GOOD = {}      # rid -> {b: (t, sf)}
    DEG = defaultdict(list)    # pid -> [(t, sf)]  (all recordings of that person)
    JIT = {k: {} for k in JITTER}   # k -> rid -> {b: (t, sf)}
    for rid, r in recs.items():
        GOOD[rid] = {}
        for k in JITTER:
            JIT[k][rid] = {}
        for d in r["blocks"]:
            if d["f"] is None:
                continue
            if d["vf"] >= MIN_VALID:
                GOOD[rid][d["b"]] = (tf(d["f"]), d["sf"])
                for k in JITTER:
                    JIT[k][rid][d["b"]] = (tf(d[k]), d["sf"])
            else:
                DEG[r["participant"]].append((rid, tf(d["f"]), d["sf"]))
    pid_of = {rid: r["participant"] for rid, r in recs.items()}

    # LOPO scaling on good blocks (identical to analysis1)
    allb = [(pid_of[rid], t) for rid in GOOD for (t, _) in GOOD[rid].values()]
    BPID = np.array([p for p, _ in allb])
    TB = np.array([t for _, t in allb])
    SCALE = {p: (TB[BPID != p].mean(0), TB[BPID != p].std(0) + 1e-9) for p in sorted(set(BPID))}
    MU_ALL, SD_ALL = TB.mean(0), TB.std(0) + 1e-9

    # ---- query windows (two adjacent good blocks, non-overlapping tiling)
    Q = []
    for rid, r in recs.items():
        n = len(r["blocks"])
        for w in range(n // 2):
            if 2 * w in GOOD[rid] and 2 * w + 1 in GOOD[rid]:
                t = (GOOD[rid][2 * w][0] + GOOD[rid][2 * w + 1][0]) / 2
                Q.append(dict(rid=rid, pid=pid_of[rid], w=w, t=t))

    # ---- structured sampler: n clips of L blocks from one recording, gap >= GAP blocks
    def sample_clips(rid, n, L, rng, forbid=frozenset(), tries=200):
        g = GOOD[rid]
        starts = [b for b in g if all((b + i) in g and (b + i) not in forbid for i in range(L))]
        if len(starts) < n:
            return None
        for _ in range(tries):
            ch = sorted(rng.choice(starts, n, replace=False))
            if all(ch[i + 1] - (ch[i] + L) >= GAP for i in range(n - 1)):
                return [b + i for b in ch for i in range(L)]
        return None

    STRUCT = {  # name -> list of (n_clips, clip_blocks) per distinct recording
        "1x20s_1rec": [(1, 4)], "2x10s_1rec": [(2, 2)], "2x10s_2rec": [(1, 2), (1, 2)],
        "4x5s_1rec": [(4, 1)], "4x5s_2rec": [(2, 1), (2, 1)],
        "4x5s_3rec": [(2, 1), (1, 1), (1, 1)],
        "1x5s_1rec": [(1, 1)], "1x10s_1rec": [(1, 2)],
        "2x5s_2rec": [(1, 1), (1, 1)], "2x20s_2rec": [(1, 4), (1, 4)],
    }
    OWN = {"own_1x20s": (1, 4), "own_2x10s": (2, 2), "own_4x5s": (4, 1)}

    def rec_ok(rid):
        rng = np.random.default_rng(1)
        return all(sample_clips(rid, n, L, rng) is not None for (n, L) in [(1, 4), (2, 2), (4, 1)])

    elig = {rid for rid in recs if rec_ok(rid)}
    recs_of = defaultdict(list)
    for rid in sorted(elig):
        recs_of[pid_of[rid]].append(rid)
    all_pids = sorted({pid_of[r] for r in recs})
    main_p = [p for p in all_pids if len(recs_of[p]) >= MIN_RECS]
    outside_p = [p for p in all_pids if p not in main_p]   # never queries -> wrong-person pool
    excl = {p: dict(n_recordings=len({r for r in recs if pid_of[r] == p}),
                    n_eligible_recordings=len(recs_of[p])) for p in outside_p}
    print("main subset", len(main_p), "excluded", len(outside_p), flush=True)

    def mk(blocks_t_sf):
        t = np.mean([x[0] for x in blocks_t_sf], 0)
        sf = float(np.mean([x[1] for x in blocks_t_sf]))
        return t, sf

    def draw_struct(p, name, rng, exclude_rid=None):
        srcs = [r for r in recs_of[p] if r != exclude_rid]
        parts = STRUCT[name]
        if len(srcs) < len(parts):
            return None
        chosen = rng.choice(len(srcs), len(parts), replace=False)
        out = []
        for (n, L), ci in zip(parts, chosen):
            rid = srcs[ci]
            bl = sample_clips(rid, n, L, rng)
            if bl is None:
                return None
            out += [GOOD[rid][b] for b in bl]
        return mk(out)

    # ---- block pools for pool-based conditions (C, D, E)
    def pool(p, exclude_rid=None):
        return [(rid, b) for rid in recs_of[p] if rid != exclude_rid for b in GOOD[rid]]

    def cov_vec(t):
        z = (t - MU_ALL) / SD_ALL
        return np.array([z[d].mean() for d in COV_DIMS])

    def pick(pl, k, policy, rng):
        if len(pl) < k:
            return None
        if policy == "random":
            return [pl[i] for i in rng.choice(len(pl), k, replace=False)]
        C = np.array([cov_vec(GOOD[rid][b][0]) for rid, b in pl])
        idx = [int(rng.integers(len(pl)))]
        dmin = np.linalg.norm(C - C[idx[0]], axis=1)
        for _ in range(k - 1):
            j = int(np.argmax(dmin))
            idx.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(C - C[j], axis=1))
        return [pl[i] for i in idx]

    WP = [(rid, b) for rid in GOOD if pid_of[rid] in outside_p for b in GOOD[rid]]
    LOWQ = ["good", "jit_lo", "jit_hi", "degraded", "wrong_person"]

    def draw_pool(p, cond, rng, exclude_rid=None):
        kind = cond[0]
        pl = pool(p, exclude_rid)
        if kind in ("rand", "cov"):
            sel = pick(pl, cond[1], "random" if kind == "rand" else "coverage", rng)
            return None if sel is None else mk([GOOD[r][b] for r, b in sel])
        # low quality: 4 good random + m extra of type typ
        _, typ, m = cond
        base = pick(pl, 4 + (m if typ == "good" else 0), "random", rng)
        if base is None:
            return None
        items = [GOOD[r][b] for r, b in base]
        if typ == "good" or m == 0:
            return mk(items)
        if typ in JITTER:
            rest = [x for x in pl if x not in base]
            if len(rest) < m:
                return None
            items += [JIT[typ][r][b] for r, b in (rest[i] for i in rng.choice(len(rest), m, replace=False))]
        elif typ == "degraded":
            dg = [x for x in DEG[p] if x[0] != exclude_rid]
            if len(dg) < m:
                return None
            items += [(dg[i][1], dg[i][2]) for i in rng.choice(len(dg), m, replace=False)]
        elif typ == "wrong_person":
            items += [GOOD[r][b] for r, b in (WP[i] for i in rng.choice(len(WP), m, replace=False))]
        return mk(items)

    CONDS = {n: ("struct", n) for n in STRUCT}
    for k in (1, 2, 4, 8, 16):
        CONDS[f"rand_k{k}"] = ("pool", ("rand", k))
    for k in (2, 4, 8):
        CONDS[f"cov_k{k}"] = ("pool", ("cov", k))
    CONDS["lq_base4"] = ("pool", ("lq", "good", 0))
    for m in (2, 4):
        for typ in LOWQ:
            CONDS[f"lq_{typ}_m{m}"] = ("pool", ("lq", typ, m))

    def draw(p, cname, rng, exclude_rid=None):
        kind, c = CONDS[cname]
        return draw_struct(p, c, rng, exclude_rid) if kind == "struct" else draw_pool(p, c, rng, exclude_rid)

    # ---- negative banks: N_NEG sets per main-subset person (all their recordings)
    BANK = {}
    for cname in CONDS:
        T_, S_ = np.full((len(main_p), N_NEG, 7), np.nan), np.full((len(main_p), N_NEG), np.nan)
        for i, p in enumerate(main_p):
            rng = np.random.default_rng(1000 + i)
            for j in range(N_NEG):
                s = draw(p, cname, rng)
                if s is not None:
                    T_[i, j], S_[i, j] = s
        BANK[cname] = (T_, S_)
    # own-recording conditions use single-recording negatives of matching structure
    OWN_NEG = {"own_1x20s": "1x20s_1rec", "own_2x10s": "2x10s_1rec", "own_4x5s": "4x5s_1rec"}
    print("banks built", flush=True)

    MAINIDX = {p: i for i, p in enumerate(main_p)}

    def score(q, pos, cneg):
        """pos: list of (t, sf). Returns (mean Delta, mean winrate) or None."""
        p = q["pid"]
        mu, sd = SCALE[p]
        dq = (q["t"] - mu) / sd
        T_, S_ = BANK[cneg]
        keep = np.arange(len(main_p)) != MAINIDX[p]
        Tn, Sn = T_[keep], S_[keep]
        dist = np.linalg.norm((Tn - mu) / sd - dq, axis=-1)                    # (P,B)
        pt = np.array([x[0] for x in pos])
        psf = np.array([x[1] for x in pos])
        dpos = np.linalg.norm((pt - mu) / sd - dq, axis=-1)                    # (D,)
        M = (np.abs(Sn[None] - psf[:, None, None]) <= TOL) & ~np.isnan(Sn)[None]   # (D,P,B)
        cnt = M.sum(-1)
        with np.errstate(invalid="ignore"):
            pm = (M * np.nan_to_num(dist)[None]).sum(-1) / cnt                # (D,P)
            dneg = np.nanmean(np.where(cnt > 0, pm, np.nan), 1)
            win = (M & (np.nan_to_num(dist)[None] > dpos[:, None, None])).sum((1, 2)) / cnt.sum(1)
        ok = ~np.isnan(dneg)
        if not ok.any():
            return None
        return float(np.mean(dneg[ok] - dpos[ok])), float(np.mean(win[ok]))

    per = {c: defaultdict(list) for c in list(CONDS) + list(OWN)}
    perw = {c: defaultdict(list) for c in list(CONDS) + list(OWN)}
    POS = {}
    QM = [q for q in Q if q["pid"] in MAINIDX]
    print("queries", len(QM), flush=True)
    for qi, q in enumerate(QM):
        p = q["pid"]
        for cname in CONDS:
            key = (p, q["rid"], cname)
            if key not in POS:
                rng = np.random.default_rng(zlib.crc32(repr(key).encode()))
                POS[key] = [s for s in (draw(p, cname, rng, q["rid"]) for _ in range(N_POS)) if s is not None]
            if POS[key]:
                r = score(q, POS[key], cname)
                if r:
                    per[cname][p].append(r[0])
                    perw[cname][p].append(r[1])
        # same-recording upper bound
        rid, w = q["rid"], q["w"]
        forbid = frozenset(range(2 * w - GAP, 2 * w + 2 + GAP))
        rng = np.random.default_rng(qi)
        for oname, (n, L) in OWN.items():
            pos = []
            for _ in range(N_POS_OWN):
                bl = sample_clips(rid, n, L, rng, forbid)
                if bl is not None:
                    pos.append(mk([GOOD[rid][b] for b in bl]))
            if pos:
                r = score(q, pos, OWN_NEG[oname])
                if r:
                    per[oname][p].append(r[0])
                    perw[oname][p].append(r[1])
        if qi % 200 == 0:
            print("query", qi, flush=True)

    # ---- summaries
    res = dict(conditions={}, paired={})
    pmean = {c: {p: float(np.mean(v)) for p, v in per[c].items()} for c in per}
    wmean = {c: {p: float(np.mean(v)) for p, v in perw[c].items()} for c in perw}
    for c in per:
        ps = sorted(pmean[c])
        if len(ps) < 2:
            continue
        res["conditions"][c] = dict(participants=ps, delta=[pmean[c][p] for p in ps],
                                    win=[wmean[c][p] for p in ps],
                                    delta_ci=boot_ci([pmean[c][p] for p in ps]),
                                    win_ci=boot_ci([wmean[c][p] - 0.5 for p in ps]),
                                    n_queries=sum(len(per[c][p]) for p in ps))

    def paired(a, b):
        ps = sorted(set(pmean[a]) & set(pmean[b]))
        if len(ps) < 2:
            return None
        return dict(delta=boot_ci([pmean[a][p] - pmean[b][p] for p in ps]),
                    win=boot_ci([wmean[a][p] - wmean[b][p] for p in ps]))

    PAIRS = [("2x10s_1rec", "1x20s_1rec"), ("4x5s_1rec", "1x20s_1rec"),
             ("2x10s_2rec", "2x10s_1rec"), ("2x10s_2rec", "1x20s_1rec"),
             ("4x5s_2rec", "4x5s_1rec"), ("4x5s_2rec", "1x20s_1rec"), ("4x5s_3rec", "4x5s_1rec"),
             ("4x5s_3rec", "4x5s_2rec"), ("4x5s_3rec", "1x20s_1rec"),
             ("1x10s_1rec", "1x5s_1rec"), ("1x20s_1rec", "1x10s_1rec"),
             ("2x10s_2rec", "2x5s_2rec"), ("2x20s_2rec", "2x10s_2rec"),
             ("2x5s_2rec", "1x10s_1rec"), ("2x20s_2rec", "4x5s_3rec"),
             ("cov_k2", "rand_k2"), ("cov_k4", "rand_k4"), ("cov_k8", "rand_k8"),
             ("rand_k2", "rand_k1"), ("rand_k4", "rand_k2"), ("rand_k8", "rand_k4"),
             ("rand_k16", "rand_k8"),
             ("own_2x10s", "own_1x20s"), ("own_4x5s", "own_1x20s"),
             ("own_1x20s", "1x20s_1rec"), ("own_4x5s", "4x5s_3rec")]
    for m in (2, 4):
        for typ in LOWQ:
            PAIRS.append((f"lq_{typ}_m{m}", "lq_base4"))
            if typ != "good":
                PAIRS.append((f"lq_{typ}_m{m}", f"lq_good_m{m}"))
    for a, b in PAIRS:
        r = paired(a, b)
        if r:
            res["paired"][f"{a} - {b}"] = r
    for c, v in res["conditions"].items():
        print(f"{c:22s} delta {np.round(v['delta_ci'][:3], 3)} n={v['delta_ci'][4]} "
              f"win-0.5 {np.round(v['win_ci'][:3], 3)}", flush=True)
    for k, v in res["paired"].items():
        print(f"{k:40s} {np.round(v['delta'][:3], 3)} p<=0 {v['delta'][3]:.3f}", flush=True)

    # coverage / data description
    jit_raw = {k: [] for k in JITTER}
    clean_raw = []
    for rid in GOOD:
        for b in GOOD[rid]:
            clean_raw.append(GOOD[rid][b][0])
            for k in JITTER:
                jit_raw[k].append(JIT[k][rid][b][0])
    res["data"] = dict(
        n_recordings=len(recs), n_participants=len(all_pids),
        n_eligible_recordings=len(elig), main_subset=main_p,
        recordings_per_main={p: len(recs_of[p]) for p in main_p},
        excluded=excl, n_query_windows=len(QM),
        n_degraded_blocks={p: len(DEG[p]) for p in main_p},
        jitter_median_transformed={k: np.median(np.array(v) - np.array(clean_raw), 0).tolist()
                                   for k, v in jit_raw.items()},
        settings=dict(block_s=BLOCK / FPS, gap_s=GAP * BLOCK / FPS, tol=TOL, n_pos=N_POS,
                      n_pos_own=N_POS_OWN, n_neg=N_NEG, min_recs=MIN_RECS, jitter=JITTER,
                      deg_min=DEG_MIN, n_boot=N_BOOT, feats=FEATS,
                      cov_dims=[[FEATS[i] for i in d] for d in COV_DIMS],
                      structures=STRUCT, own=OWN))
    json.dump(res, open(os.path.join(out_dir, "analysis2_results.json"), "w"), indent=1)
    print("saved", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ["*"]))
