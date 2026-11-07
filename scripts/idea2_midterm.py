"""Idea 2 midterm: reference encoders evaluated on held-out participants.

Representations of a reference set (all scored with the same protocol):
  desc_mean / desc_median   handcrafted descriptors (analysis2), mean or median pooling
  si_latent / si_expr       frozen Seamless Interaction movement features (frame latent;
                            expression + FAU), mean pooling
  si_hyper                  frozen hypernet features (appearance / identity control)
  learned_mean              Transformer block encoder, InfoNCE across recordings, mean pooling
  learned_mean_aug          same, trained with contaminated reference sets
  learned_gated             same + quality-gated attention pooling (updated Idea 2)
  posture_static            block-mean upper-body posture, no training (shortcut control)
  learned_dyn_mean/_gated   encoder whose input has the block-mean posture removed, so it
                            sees motion dynamics only (shortcut control)

Protocol. Participants of the evaluation archives (0000-0009) are split into 4 folds.
For fold f, encoders are trained on all other participants (other folds + training-only
archives) and evaluated on fold f. Queries are 10 s windows of fold-f participants with
>= MIN_RECS eligible recordings; references never come from the query recording (except the
same-recording upper bound). Negatives are reference sets of the other fold-f participants,
built with the same structure and matched within TOL speaking fraction.
Per query: A = fraction of matched negative sets farther than the positive set,
Delta_rel = (mean negative distance - positive distance) / mean negative distance.
Participant means; bootstrap over participants.

Usage: python idea2_midterm.py CACHE_DIR DATA_DIR OUT_DIR [steps]
"""
import glob
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis1 import MIN_VALID, TOL, parse_id  # noqa: E402
from analysis2 import boot_ci  # noqa: E402

EVAL_GLOB = "0000_000[0-9]"
NFOLD = 4
MIN_RECS = 4
GAP = 2
N_POS, N_NEG = 20, 50
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

STRUCT = {"1x20s_1rec": [(1, 4)], "2x10s_1rec": [(2, 2)], "4x5s_1rec": [(4, 1)],
          "2x10s_2rec": [(1, 2), (1, 2)], "4x5s_2rec": [(2, 1), (2, 1)],
          "4x5s_3rec": [(2, 1), (1, 1), (1, 1)]}
LQ = {"lq_base4": ("none", 0), "lq_jit_m2": ("jit", 2), "lq_jit_m4": ("jit", 4),
      "lq_wp_m2": ("wp", 2), "lq_wp_m4": ("wp", 4)}
CONDS = list(STRUCT) + list(LQ) + ["own_1x20s"]
REPS = ["desc_mean", "desc_median", "si_latent", "si_expr", "si_hyper", "posture_static",
        "learned_mean", "learned_mean_aug", "learned_gated", "learned_dyn_mean", "learned_dyn_gated"]
MODES = ["mean", "mean_aug", "gated", "dyn_mean", "dyn_gated"]
NO_JIT = {"si_latent", "si_expr", "si_hyper", "posture_static"}     # video features: pose jitter does not apply


# ----------------------------------------------------------------------------- data
def load(cache, data):
    ev = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data, EVAL_GLOB, "*.npz"))}
    R = {}
    for f in sorted(glob.glob(os.path.join(cache, "*.npz"))):
        rid = os.path.basename(f)[:-4]
        z = np.load(f)
        if z["desc"].ndim != 2 or len(z["desc"]) == 0:      # recording shorter than one block
            continue
        good = (z["vf"] >= MIN_VALID) & (z["vm"] >= MIN_VALID) & ~np.isnan(z["desc"]).any(1) \
            & ~np.isnan(z["si_latent"]).any(1) & ~np.isnan(z["si_hyper"].astype(np.float32)).any(1)
        R[rid] = dict(pid=parse_id(rid)["participant"], ev=rid in ev, good=set(np.flatnonzero(good)),
                      **{k: z[k] for k in z.files})
        pm = z["pmask"][..., None]
        R[rid]["posture"] = (z["pose"].astype(np.float32) * pm).sum(1) / np.maximum(pm.sum(1), 1)
    return R


# ----------------------------------------------------------------------------- model
class Enc(nn.Module):
    def __init__(self, din=168, d=128, T=50, dyn=False):
        super().__init__()
        self.dyn = dyn
        self.inp = nn.Linear(din, d)
        self.pos = nn.Parameter(torch.zeros(T, d))
        layer = nn.TransformerEncoderLayer(d, 4, 256, 0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.gate = nn.Sequential(nn.Linear(d + 1, 64), nn.GELU(), nn.Linear(64, 1))

    def block(self, x, m):                       # x (N,T,84) standardized, m (N,T)
        if self.dyn:                             # remove the block's static posture
            mf = m[..., None].float()
            x = (x - (x * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)) * mf
        v = torch.cat([x, torch.diff(x, dim=1, prepend=x[:, :1])], -1)
        h = self.tr(self.inp(v) + self.pos, src_key_padding_mask=~m)
        h = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.out(h)

    def pool(self, h, sm, gated):                # h (S,K,d), sm (S,K) set mask
        if not gated:
            a = sm.float() / sm.sum(1, keepdim=True)
        else:
            hn = Fn.normalize(h, dim=-1)
            tot = (hn * sm[..., None]).sum(1, keepdim=True)
            loo = (tot - hn) / (sm.sum(1, keepdim=True)[..., None] - 1).clamp(min=1)
            c = (hn * Fn.normalize(loo, dim=-1)).sum(-1, keepdim=True)
            lg = self.gate(torch.cat([h, c], -1))[..., 0].masked_fill(~sm, -1e9)
            a = torch.softmax(lg, 1)
        return Fn.normalize((a[..., None] * h).sum(1), dim=-1), a


def train_model(R, train_pids, mode, steps, log):
    """mode: mean | mean_aug | gated | dyn_mean | dyn_gated."""
    recs = defaultdict(list)
    for rid, r in R.items():
        if r["pid"] in train_pids and len(r["good"]) > 0:
            recs[r["pid"]].append(rid)
    pids = [p for p in recs if len(recs[p]) >= 2]
    allrid = [rid for p in recs for rid in recs[p]]
    X = np.concatenate([R[rid]["pose"][sorted(R[rid]["good"])] for rid in allrid]).astype(np.float32)
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-4
    # flat GPU tensors of all training blocks, clean and jittered
    idx, off = {}, 0
    for rid in allrid:
        g = sorted(R[rid]["good"])
        idx[rid] = np.arange(off, off + len(g))
        off += len(g)

    def stack(key):
        a = np.concatenate([R[rid][key][sorted(R[rid]["good"])] for rid in allrid]).astype(np.float32)
        return torch.tensor((a - mu) / sd, device=DEV, dtype=torch.float16)
    XC, XJ = stack("pose"), stack("pose_jit")
    MK = torch.tensor(np.concatenate([R[rid]["pmask"][sorted(R[rid]["good"])] for rid in allrid]), device=DEV)
    model = Enc(dyn=mode.startswith("dyn")).to(DEV)
    gated = mode.endswith("gated")
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    P, tau = 48, 0.1
    for step in range(steps):
        bp = rng.choice(len(pids), min(P, len(pids)), replace=False)
        items, setid, corrupt = [], [], []     # items: (flat index, jittered?)
        for si in range(2 * len(bp)):
            p = pids[bp[si // 2]]
            rr = list(recs[p])
            rng.shuffle(rr)
            half = max(1, len(rr) // 2)
            grp = rr[:half] if si % 2 == 0 else rr[half:]
            k = int(rng.choice([1, 2, 4]))
            for _ in range(k):
                rid = grp[rng.integers(len(grp))]
                items.append((int(rng.choice(idx[rid])), False)); setid.append(si); corrupt.append(False)
            if mode in ("mean_aug", "gated", "dyn_gated") and rng.random() < 0.5:
                kind = "jit" if rng.random() < 0.5 else "wp"
                for _ in range(int(rng.integers(1, 3))):
                    if kind == "jit":
                        rid = grp[rng.integers(len(grp))]
                        items.append((int(rng.choice(idx[rid])), True))
                    else:
                        q = pids[bp[(si // 2 + 1 + rng.integers(len(bp) - 1)) % len(bp)]]
                        rid = recs[q][rng.integers(len(recs[q]))]
                        items.append((int(rng.choice(idx[rid])), False))
                    setid.append(si); corrupt.append(True)
        fi = torch.tensor([i for i, _ in items], device=DEV)
        jj = torch.tensor([j for _, j in items], device=DEV)
        x = torch.where(jj[:, None, None], XJ[fi], XC[fi]).float()
        m = MK[fi]
        h = model.block(x, m)
        S = 2 * len(bp)
        sid = np.array(setid)
        slot = np.zeros(len(sid), int)
        cnt = np.zeros(S, int)
        for n, s_ in enumerate(sid):
            slot[n] = cnt[s_]; cnt[s_] += 1
        H = torch.zeros(S, cnt.max(), h.shape[-1], device=DEV)
        SM = torch.zeros(S, cnt.max(), dtype=torch.bool, device=DEV)
        CM = torch.zeros(S, cnt.max(), device=DEV)
        ti, tk = torch.tensor(sid, device=DEV), torch.tensor(slot, device=DEV)
        H = H.index_put((ti, tk), h)
        SM[ti, tk] = True
        CM[ti, tk] = torch.tensor(corrupt, device=DEV, dtype=torch.float)
        z, a = model.pool(H, SM, gated)
        za, zb = z[0::2], z[1::2]
        lg = za @ zb.T / tau
        lab = torch.arange(len(bp), device=DEV)
        loss = (Fn.cross_entropy(lg, lab) + Fn.cross_entropy(lg.T, lab)) / 2
        if gated:
            loss = loss + (a * CM).sum(1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            acc = (lg.argmax(1) == lab).float().mean().item()
            log.append(dict(step=step, loss=loss.item(), batch_acc=acc))
            print(f"  [{mode}] step {step} loss {loss.item():.3f} acc {acc:.2f}", flush=True)
    model.eval()
    return model, mu, sd, len(pids)


# ----------------------------------------------------------------------------- sets
def make_sampler(R, recs_of):
    def clips(rid, n, L, rng, forbid=frozenset(), tries=200):
        g = R[rid]["good"]
        starts = [b for b in g if all((b + i) in g and (b + i) not in forbid for i in range(L))]
        if len(starts) < n:
            return None
        for _ in range(tries):
            ch = sorted(rng.choice(starts, n, replace=False))
            if all(ch[i + 1] - (ch[i] + L) >= GAP for i in range(n - 1)):
                return [b + i for b in ch for i in range(L)]
        return None

    def draw(p, cond, rng, excl=None, intruders=()):
        """Returns list of (rid, b, kind, corrupt) or None. kind: 'c' clean, 'j' jittered."""
        srcs = [r for r in recs_of[p] if r != excl]
        if cond in STRUCT:
            parts = STRUCT[cond]
            if len(srcs) < len(parts):
                return None
            out = []
            for (n, L), ci in zip(parts, rng.choice(len(srcs), len(parts), replace=False)):
                bl = clips(srcs[ci], n, L, rng)
                if bl is None:
                    return None
                out += [(srcs[ci], b, "c", False) for b in bl]
            return out
        typ, m = LQ[cond]
        pl = [(r, b) for r in srcs for b in R[r]["good"]]
        if len(pl) < 4 + m:
            return None
        ch = [pl[i] for i in rng.choice(len(pl), 4 + m, replace=False)]
        out = [(r, b, "c", False) for r, b in ch[:4]]
        if typ == "jit":
            out += [(r, b, "j", True) for r, b in ch[4:]]
        elif typ == "wp":
            if len(intruders) < m:
                return None
            out += [(*intruders[i], "c", True) for i in rng.choice(len(intruders), m, replace=False)]
        return out
    return clips, draw


# ----------------------------------------------------------------------------- main
def main(cache, data, out, steps=4000):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    R = load(cache, data)
    print(f"{len(R)} recordings loaded ({time.time() - t0:.0f}s)", flush=True)
    ev_pids = sorted({r["pid"] for r in R.values() if r["ev"]})
    rng0 = np.random.default_rng(1)

    def rec_ok(rid, clips):
        rng = np.random.default_rng(1)
        return all(clips(rid, n, L, rng) is not None for n, L in [(1, 4), (2, 2), (4, 1)])

    recs_of = defaultdict(list)
    clips, draw = make_sampler(R, recs_of)
    for rid in sorted(R):
        if R[rid]["ev"] and rec_ok(rid, clips):
            recs_of[R[rid]["pid"]].append(rid)
    main_p = [p for p in ev_pids if len(recs_of[p]) >= MIN_RECS]
    other_p = [p for p in ev_pids if p not in main_p]
    rng0.shuffle(main_p); rng0.shuffle(other_p)
    fold = {p: i % NFOLD for i, p in enumerate(main_p)}
    fold.update({p: i % NFOLD for i, p in enumerate(other_p)})
    extra = sorted({r["pid"] for r in R.values() if not r["ev"]})
    print(f"eval participants {len(ev_pids)} (main {len(main_p)}), extra-only "
          f"{len(set(extra) - set(ev_pids))}, overlap {len(set(extra) & set(ev_pids))}", flush=True)

    per = {rep: {c: {} for c in CONDS} for rep in REPS}          # rep -> cond -> pid -> [A, Drel, D]
    attn = defaultdict(list)                                       # cond -> [mean alpha corrupt, clean]
    trainlog = {}
    per_query = defaultdict(list)

    for f in range(NFOLD):
        test = [p for p in ev_pids if fold[p] == f]
        train_pids = set(R[r]["pid"] for r in R) - set(test)
        qp = [p for p in test if p in main_p]
        print(f"fold {f}: test {len(test)} ({len(qp)} query), train participants {len(train_pids)}", flush=True)

        # ---- reference sets (shared by every representation)
        intr = {p: [(r, b) for q in test if q != p for r in recs_of[q] for b in R[r]["good"]] for p in test}
        NEG = {}
        for c in CONDS:
            cc = "1x20s_1rec" if c == "own_1x20s" else c
            for p in test:
                rng = np.random.default_rng(zlib.crc32(f"neg{p}{c}".encode()))
                NEG[(p, c)] = [s for s in (draw(p, cc, rng, intruders=intr[p]) for _ in range(N_NEG)) if s]
        Q, POS = [], {}
        for p in qp:
            for rid in recs_of[p]:
                g = R[rid]["good"]
                for w in range(len(R[rid]["sf"]) // 2):
                    if 2 * w in g and 2 * w + 1 in g:
                        Q.append((p, rid, w))
        for p, rid, w in Q:
            for c in CONDS:
                key = (p, rid, c) if c != "own_1x20s" else (p, rid, c, w)
                if key in POS:
                    continue
                rng = np.random.default_rng(zlib.crc32(repr(key).encode()))
                if c == "own_1x20s":
                    forbid = frozenset(range(2 * w - GAP, 2 * w + 2 + GAP))
                    POS[key] = [[(rid, b, "c", False) for b in bl] for bl in
                                (clips(rid, 1, 4, rng, forbid) for _ in range(N_POS)) if bl]
                else:
                    POS[key] = [s for s in (draw(p, c, rng, rid, intr[p]) for _ in range(N_POS)) if s]
        print(f"  {len(Q)} queries, sets built", flush=True)

        # ---- block vectors per representation
        frecs = sorted({r for p in test for r in recs_of[p]})
        trrecs = [r for r in R if R[r]["pid"] in train_pids and R[r]["good"]]

        def zscore(key):
            A = np.concatenate([R[r][key][sorted(R[r]["good"])].astype(np.float32) for r in trrecs])
            mu, sd = A.mean(0), A.std(0) + 1e-6
            V = {}
            for r in frecs:
                V[(r, "c")] = (R[r][key].astype(np.float32) - mu) / sd
                if key == "desc":
                    V[(r, "j")] = (R[r]["desc_jit"].astype(np.float32) - mu) / sd
            return V
        VEC = {"desc": zscore("desc"), "si_latent": zscore("si_latent"),
               "si_expr": zscore("si_expr"), "si_hyper": zscore("si_hyper"),
               "posture_static": zscore("posture")}
        models = {}
        for mode in MODES:
            log = []
            model, mu, sd, npid = train_model(R, train_pids, mode, steps, log)
            trainlog[f"fold{f}_{mode}"] = dict(log=log, n_train_participants=npid)
            V = {}
            with torch.no_grad():
                for r in frecs:
                    for kind, key in (("c", "pose"), ("j", "pose_jit")):
                        x = torch.tensor((R[r][key].astype(np.float32) - mu) / sd, device=DEV)
                        m = torch.tensor(R[r]["pmask"], device=DEV)
                        V[(r, kind)] = model.block(x, m).float().cpu().numpy()
            models[mode] = (model, V)
        print(f"  models trained ({time.time() - t0:.0f}s)", flush=True)

        def embed(rep, sets):
            """sets: list of item lists -> (S, d) array (normalized for learned), alphas."""
            if rep.startswith("learned"):
                mode = rep[len("learned_"):]
                model, V = models[mode]
                K = max(len(s) for s in sets)
                H = np.zeros((len(sets), K, 128), np.float32)
                SM = np.zeros((len(sets), K), bool)
                for i, s in enumerate(sets):
                    for k, (r, b, kind, _) in enumerate(s):
                        H[i, k] = V[(r, kind)][b]
                    SM[i, :len(s)] = True
                with torch.no_grad():
                    z, a = model.pool(torch.tensor(H, device=DEV), torch.tensor(SM, device=DEV),
                                      mode.endswith("gated"))
                return z.cpu().numpy(), a.cpu().numpy()
            key = "desc" if rep.startswith("desc") else rep
            V = VEC[key]
            rows = [np.stack([V[(r, kind)][b] for r, b, kind, _ in s]) for s in sets]
            return np.stack([np.median(x, 0) if rep == "desc_median" else x.mean(0) for x in rows]), None

        def sfs(sets):
            return np.array([np.mean([R[r]["sf"][b] for r, b, _, _ in s]) for s in sets])

        for rep in REPS:
            conds = [c for c in CONDS if not (rep in NO_JIT and c.startswith("lq_jit"))]
            negE = {}
            for c in conds:
                for p in test:
                    s = NEG[(p, c)]
                    negE[(p, c)] = (embed(rep, s)[0], sfs(s)) if s else None
            # queries: mean of two clean blocks (normalized for learned)
            qsets = [[(rid, 2 * w, "c", False), (rid, 2 * w + 1, "c", False)] for p, rid, w in Q]
            if rep.startswith("learned"):
                _, V = models[rep[len("learned_"):]]
                qe = np.stack([(V[(r, "c")][b1] + V[(r, "c")][b2]) / 2
                               for (r, b1, _, _), (_, b2, _, _) in qsets])
                qe /= np.linalg.norm(qe, axis=1, keepdims=True)
            else:
                qe = embed("desc_mean" if rep.startswith("desc") else rep, qsets)[0]
            posE = {}
            for key, s in POS.items():
                if s and key[2] in conds:
                    e, a = embed(rep, s)
                    posE[key] = (e, sfs(s))
                    if rep.endswith("gated") and key[2] in LQ and a is not None:
                        for i, st in enumerate(s):
                            cm = np.array([it[3] for it in st] + [False] * (a.shape[1] - len(st)))
                            vm = np.zeros(a.shape[1], bool); vm[:len(st)] = True
                            if cm.any():
                                attn[f"{rep}|{key[2]}"].append([a[i][cm].mean(), a[i][vm & ~cm].mean()])
            acc = {c: defaultdict(list) for c in conds}
            for qi, (p, rid, w) in enumerate(Q):
                for c in conds:
                    key = (p, rid, c) if c != "own_1x20s" else (p, rid, c, w)
                    if key not in posE:
                        continue
                    pe, psf = posE[key]
                    dpos = np.linalg.norm(pe - qe[qi], axis=1)
                    As, Ds, Dr = [], [], []
                    for p2 in test:
                        if p2 == p or negE.get((p2, c)) is None:
                            continue
                        ne, nsf = negE[(p2, c)]
                        dn = np.linalg.norm(ne - qe[qi], axis=1)
                        M = np.abs(nsf[None] - psf[:, None]) <= TOL              # (pos, neg)
                        cnt = M.sum(1)
                        okk = cnt > 0
                        if not okk.any():
                            continue
                        mdn = (M * dn[None]).sum(1)[okk] / cnt[okk]
                        As.append(((M & (dn[None] > dpos[:, None])).sum(1)[okk] / cnt[okk]))
                        Ds.append(mdn - dpos[okk]); Dr.append((mdn - dpos[okk]) / mdn)
                    if As:
                        # average over negative participants, then over positive draws
                        A = np.mean([a.mean() for a in As]); D = np.mean([d.mean() for d in Ds])
                        Drl = np.mean([d.mean() for d in Dr])
                        acc[c][p].append((A, Drl, D))
                        if c in ("1x20s_1rec", "4x5s_3rec"):
                            per_query[f"{rep}|{c}"].append(dict(p=p, rid=rid, w=w, A=float(A)))
            for c in conds:
                for p, v in acc[c].items():
                    per[rep][c][p] = np.mean(v, 0).tolist()
            a1 = per[rep]["1x20s_1rec"]
            print(f"  {rep:18s} A(1x20s) " + " ".join(f"{p}:{a1[p][0]:.2f}" for p in qp if p in a1), flush=True)

    # ----------------------------------------------------------------------------- summary
    res = dict(conditions={}, paired={}, attention={}, train=trainlog,
               data=dict(eval_participants=ev_pids, main=sorted(main_p), fold=fold,
                         n_recordings=len(R), n_eval_recordings=sum(r["ev"] for r in R.values()),
                         extra_participants=len(set(extra) - set(ev_pids)),
                         settings=dict(nfold=NFOLD, n_pos=N_POS, n_neg=N_NEG, tol=TOL, steps=steps,
                                       min_recs=MIN_RECS)))
    for rep in REPS:
        for c in CONDS:
            d = per[rep][c]
            if len(d) < 2:
                continue
            ps = sorted(d)
            res["conditions"][f"{rep}|{c}"] = dict(
                participants=ps, A=[d[p][0] for p in ps], Drel=[d[p][1] for p in ps], D=[d[p][2] for p in ps],
                A_ci=boot_ci([d[p][0] for p in ps]), Drel_ci=boot_ci([d[p][1] for p in ps]),
                D_ci=boot_ci([d[p][2] for p in ps]))

    def paired(ra, ca, rb, cb):
        a, b = per[ra][ca], per[rb][cb]
        ps = sorted(set(a) & set(b))
        if len(ps) < 2:
            return None
        return dict(A=boot_ci([a[p][0] - b[p][0] for p in ps]), Drel=boot_ci([a[p][1] - b[p][1] for p in ps]),
                    D=boot_ci([a[p][2] - b[p][2] for p in ps]))
    for rep in REPS:
        for ca, cb in [("2x10s_1rec", "1x20s_1rec"), ("4x5s_1rec", "1x20s_1rec"),
                       ("2x10s_2rec", "1x20s_1rec"), ("4x5s_2rec", "1x20s_1rec"),
                       ("4x5s_3rec", "1x20s_1rec"), ("4x5s_3rec", "4x5s_1rec"),
                       ("own_1x20s", "1x20s_1rec")] + [(c, "lq_base4") for c in LQ if c != "lq_base4"]:
            r = paired(rep, ca, rep, cb)
            if r:
                res["paired"][f"{rep}|{ca} - {cb}"] = r
    for c in CONDS:
        for ra, rb in [("learned_gated", "learned_mean"), ("learned_gated", "learned_mean_aug"),
                       ("learned_mean_aug", "learned_mean"), ("desc_median", "desc_mean"),
                       ("learned_mean", "desc_mean"), ("si_latent", "desc_mean"),
                       ("learned_dyn_gated", "learned_dyn_mean"), ("learned_dyn_mean", "desc_mean"),
                       ("learned_mean", "learned_dyn_mean")]:
            r = paired(ra, c, rb, c)
            if r:
                res["paired"][f"{ra} - {rb}|{c}"] = r
    for c, v in attn.items():
        v = np.array(v)
        res["attention"][c] = dict(alpha_corrupt=float(v[:, 0].mean()), alpha_clean=float(v[:, 1].mean()), n=len(v))
    res["per_query"] = per_query
    json.dump(res, open(os.path.join(out, "idea2_midterm_results.json"), "w"), indent=1, default=float)
    for k, v in res["conditions"].items():
        print(f"{k:34s} A {np.round(v['A_ci'][:3], 3)} Drel {np.round(v['Drel_ci'][:3], 3)} n={v['A_ci'][4]}")
    for k, v in res["paired"].items():
        print(f"{k:52s} dA {np.round(v['A'][:3], 3)} dDrel {np.round(v['Drel'][:3], 3)}")
    print("attention", res["attention"])
    print(f"saved ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4000)
