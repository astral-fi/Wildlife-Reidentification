"""Leopard SIFT rescue: for leopard queries ranked `new_individual` first, verify the query with SIFT
against up to CAP gallery images (most MiewID-similar first) of each of its top candidate identities.
If the best identity has >= T moving-similarity RANSAC inliers and leads the runner-up identity by
>= M, call it known: that identity goes first, `new_individual` second.

  python src_v3/leopard_sift_rescue.py cv                 # SIFT on all CV new-first leopards (cached)
  python src_v3/leopard_sift_rescue.py grid               # net MAP@5 change per (T, M), honest split
  python src_v3/leopard_sift_rescue.py test SUB.csv OUT.csv T M

CV uses v3's outer folds with default stage-1 bundles + light stage-2 (anatomy_default_v3.py).
Test uses the submission's own known guesses plus the default stage-1 top-20 candidates.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # must precede numpy import: workers fork with these BLAS settings
import sys, time, pickle
sys.path.insert(0, "src_v3"); sys.path.insert(0, "src_v2")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize

import anatomy_default_v3 as A
import build_sift_v3_and_fv as V

CAP = 24
NEW = "new_individual"
CV_CACHE = "cache_v3/leopard_rescue_cv.pkl"


def per_identity_inliers(queries, cand_lists, gallery_rows, ids_all, X):
    """queries: global rows; cand_lists: list of identity arrays; returns list of {identity: max inliers}."""
    gal_ids = ids_all[gallery_rows]
    by_id = {c: gallery_rows[gal_ids == c] for c in np.unique(gal_ids)}
    jobs, owner = [], []
    for qi, (q, cands) in enumerate(zip(queries, cand_lists)):
        for c in cands:
            g = by_id[c]
            g = g[np.argsort(-(X[g] @ X[q]))][:CAP]
            jobs.extend((q, gg) for gg in g)
            owner.extend((qi, c) for _ in g)
    J = np.array(jobs, dtype=np.int32)
    t0 = time.time()
    with ProcessPoolExecutor(16, initializer=V._init_worker) as ex:
        F = np.vstack(list(ex.map(V._compute_v3_pair_batch, np.array_split(J, max(16, len(J) // 4000)))))
    print(f"  {len(J):,} pairs in {time.time() - t0:.0f}s", flush=True)
    out = [dict() for _ in queries]
    for (qi, c), v in zip(owner, F[:, 5]):
        if v > out[qi].get(c, -1):
            out[qi][c] = float(v)
    return out


def fires(inl, T, M):
    if not inl:
        return None
    s = sorted(inl.items(), key=lambda kv: -kv[1])
    s1, s2 = s[0][1], (s[1][1] if len(s) > 1 else 0.0)
    return s[0][0] if (s1 >= T and s1 - s2 >= M) else None


def rescue(pred, c):
    """pred: ranked list with NEW first. Put c first, NEW second, then the rest."""
    rest = [p for p in pred if p not in (c, NEW)]
    return ([c, NEW] + rest)[:5]


def load_common():
    tr = pd.read_csv("reid_student_data_main/train.csv")
    X = normalize(np.load("reid_student_data_main/miewid_features.npy"))
    return tr, X


def run_cv():
    tr, X = load_common()
    ids_all = np.asarray(tr.identity.values, dtype=object)
    B = pickle.load(open("cache_v3/v3_outer_default_leopard.pkl", "rb"))
    S2 = A.stage2_lite(B)
    folds = []
    for k, (b, (s, pn)) in enumerate(zip(B, S2)):
        pred = A.decode(b, s, pn)
        nf = np.where(pred[:, 0] == NEW)[0]
        print(f"fold {k}: {len(nf)} new-first leopard queries", flush=True)
        inl = per_identity_inliers(b["v_rows"][nf], [b["cand_names"][i] for i in nf], b["g_rows"], ids_all, X)
        folds.append(dict(pred=pred, truth=b["y_true"], nf=nf, inl=inl))
    pickle.dump(folds, open(CV_CACHE, "wb"))


def run_grid():
    folds = pickle.load(open(CV_CACHE, "rb"))
    grid = [(T, M) for T in (10, 12, 15, 18, 22) for M in (4, 6, 8, 12)]

    def delta(fs, T, M):
        d, n, fired, good, bad_new = 0.0, 0, 0, 0, 0
        for f in fs:
            n += len(f["truth"])
            base = A.ap_of(f["pred"], f["truth"])
            for i, inl in zip(f["nf"], f["inl"]):
                c = fires(inl, T, M)
                if c is None:
                    continue
                newp = np.array([rescue(list(f["pred"][i]), c)], dtype=object)
                d += A.ap_of(newp, f["truth"][i:i + 1])[0] - base[i]
                fired += 1; good += c == f["truth"][i]; bad_new += f["truth"][i] == NEW
        return d / n, fired, good, bad_new

    print("Net leopard MAP@5 change on CV (fold-native ~54% new). honest = choose on folds 0-2, score 3-4 (and swap)")
    print(f"{'T':>3} {'M':>3} {'all folds':>10} {'fired':>6} {'right id':>8} {'on new':>6} | {'f0-2':>8} {'f3-4':>8}")
    res = {}
    for T, M in grid:
        a = delta(folds, T, M); x = delta(folds[:3], T, M); y = delta(folds[3:], T, M)
        res[(T, M)] = (x[0], y[0])
        print(f"{T:3d} {M:3d} {a[0]:+10.5f} {a[1]:6d} {a[2]:8d} {a[3]:6d} | {x[0]:+8.5f} {y[0]:+8.5f}")
    bx = max(res, key=lambda k: res[k][0]); by = max(res, key=lambda k: res[k][1])
    print(f"chosen on folds 0-2: {bx} -> folds 3-4 score {res[bx][1]:+.5f};  chosen on 3-4: {by} -> folds 0-2 score {res[by][0]:+.5f}")


def run_test(sub_path, out_path, T, M):
    tr, X = load_common()
    te = pd.read_csv("reid_student_data_main/test.csv")
    ids_all = np.asarray(list(tr.identity.values) + ["TEST"] * len(te), dtype=object)
    raw = A.load_v3_raw_data()
    from features_v3 import build_v3_feature_bundle, DEFAULT_STAGE1_V3_CFG
    _, _, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr = raw
    g = np.where(tr.species.values == "leopard")[0]
    tm = np.where(te.species.values == "leopard")[0]
    b = build_v3_feature_bundle(g, len(tr) + tm, "leopard", all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all,
                                kp_counts, sift_csr, top_k_cands=20, stage1_cfg=DEFAULT_STAGE1_V3_CFG["leopard"])
    sub = pd.read_csv(sub_path).set_index("id").loc[te.id]
    preds = sub.predictions.str.split().tolist()
    rows = [j for j, ti in enumerate(tm) if preds[ti][0] == NEW]
    cands = []
    for j in rows:
        own = [p for p in preds[tm[j]] if p != NEW]
        cands.append(list(dict.fromkeys(own + list(b["cand_names"][j]))))
    print(f"test: {len(rows)} leopard rows with new first", flush=True)
    inl = per_identity_inliers(len(tr) + tm[rows], cands, g, ids_all, X)
    changed = 0
    for j, d in zip(rows, inl):
        c = fires(d, T, M)
        if c is not None:
            preds[tm[j]] = rescue(preds[tm[j]], c); changed += 1
    pickle.dump(dict(rows=tm[rows], inl=inl), open("cache_v3/leopard_rescue_test.pkl", "wb"))
    pd.DataFrame({"id": te.id, "predictions": [" ".join(p) for p in preds]}).to_csv(out_path, index=False)
    print(f"wrote {out_path}: {changed} leopard rows changed (T={T}, M={M})")


# ---------------------------------------------------------------------------------------------
# Learned re-check (the fixed T/M rule above fails honest validation; this one passes):
# a logistic model on new-first rows predicts "the best-SIFT identity is the true one" from
# SIFT best/margin, light stage-2 P(new), and that identity's ranker/MiewID/stage-1 evidence.
# ---------------------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax


def recheck_features(inl, cand_names, scores, pn, X_cand):
    it = sorted(inl.items(), key=lambda kv: -kv[1])
    s1, s2 = it[0][1], (it[1][1] if len(it) > 1 else 0.0)
    c = it[0][0]; j = list(cand_names).index(c)
    return c, [np.log1p(s1), np.log1p(s1) - np.log1p(s2), s1 - s2, np.log(pn / (1 - pn + 1e-9) + 1e-9),
               scores[j] - scores.max(), float((scores > scores[j]).sum() == 0), X_cand[j, 6], X_cand[j, 0]]


def fit_light_stage2(B):
    """Same models as anatomy_default_v3.stage2_lite, trained on all folds (for test)."""
    Xc = np.vstack([b["X_cand"][~b["is_new"]].reshape(-1, b["X_cand"].shape[2]) for b in B])
    yc = np.concatenate([(b["cand_names"][~b["is_new"]] == b["y_true"][~b["is_new"], None]).ravel() for b in B])
    rk = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=400)).fit(Xc, yc)
    def cs(b):
        n, K, F = b["X_cand"].shape
        return rk.decision_function(b["X_cand"].reshape(-1, F)).reshape(n, K)
    def ox(b, s):
        ss = -np.sort(-s, 1)
        return np.c_[b["X_open_sr"], ss[:, 0], ss[:, 0] - ss[:, 1], ss[:, 0] - ss[:, 4]]
    Xo = np.vstack([ox(b, cs(b)) for b in B]); yo = np.concatenate([b["is_new"] for b in B])
    op = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=600)).fit(Xo, yo)
    return lambda b: (cs(b), op.predict_proba(ox(b, cs(b)))[:, 1])


def fit_recheck():
    folds = pickle.load(open(CV_CACHE, "rb"))
    B = pickle.load(open("cache_v3/v3_outer_default_leopard.pkl", "rb"))
    S2 = A.stage2_lite(B)
    X, y = [], []
    for f, b, (s, pn) in zip(folds, B, S2):
        for i, inl in zip(f["nf"], f["inl"]):
            c, x = recheck_features(inl, b["cand_names"][i], s[i], pn[i], b["X_cand"][i])
            X.append(x); y.append(c == f["truth"][i])
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(np.array(X, float), np.array(y)), B


def run_test_lr(sub_path, out_path, thresholds):
    tr, X = load_common()
    te = pd.read_csv("reid_student_data_main/test.csv")
    ids_all = np.asarray(list(tr.identity.values) + ["TEST"] * len(te), dtype=object)
    model, B = fit_recheck()
    light = fit_light_stage2(B)
    raw = A.load_v3_raw_data()
    from features_v3 import build_v3_feature_bundle, DEFAULT_STAGE1_V3_CFG
    _, _, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr = raw
    g = np.where(tr.species.values == "leopard")[0]
    tm = np.where(te.species.values == "leopard")[0]
    b = build_v3_feature_bundle(g, len(tr) + tm, "leopard", all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all,
                                kp_counts, sift_csr, top_k_cands=20, stage1_cfg=DEFAULT_STAGE1_V3_CFG["leopard"])
    s, pn = light(b)
    sub = pd.read_csv(sub_path).set_index("id").loc[te.id]
    base = sub.predictions.str.split().tolist()
    rows = [j for j, ti in enumerate(tm) if base[ti][0] == NEW]
    print(f"test: {len(rows)} leopard rows with new first in {sub_path}", flush=True)
    cache = "cache_v3/leopard_rescue_test_inl.pkl"
    if os.path.exists(cache) and pickle.load(open(cache, "rb"))["rows"] == rows:
        inl = pickle.load(open(cache, "rb"))["inl"]
    else:
        inl = per_identity_inliers(len(tr) + tm[rows], [b["cand_names"][j] for j in rows], g, ids_all, X)
        pickle.dump({"rows": rows, "inl": inl}, open(cache, "wb"))
    feats = [recheck_features(d, b["cand_names"][j], s[j], pn[j], b["X_cand"][j]) for j, d in zip(rows, inl)]
    p = model.predict_proba(np.array([f[1] for f in feats], float))[:, 1]
    for thr, out in zip(thresholds, out_path):
        preds = [list(x) for x in base]; changed = 0
        for j, (c, _), pj in zip(rows, feats, p):
            if pj >= thr:
                preds[tm[j]] = rescue(preds[tm[j]], c); changed += 1
        pd.DataFrame({"id": te.id, "predictions": [" ".join(x) for x in preds]}).to_csv(out, index=False)
        print(f"wrote {out}: {changed} leopard rows rescued (threshold {thr})")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cv":
        run_cv()
    elif mode == "grid":
        run_grid()
    elif mode == "test":
        run_test(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
    elif mode == "test_lr":   # test_lr SUB.csv OUT_PREFIX thr1,thr2,...
        thrs = [float(t) for t in sys.argv[4].split(",")]
        run_test_lr(sys.argv[2], [f"{sys.argv[3]}_p{int(t * 100)}.csv" for t in thrs], thrs)
