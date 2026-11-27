"""Where does v3 lose MAP@5? Rebuilds v3's outer-fold validation bundles with the DEFAULT stage-1
config (no Optuna), fits light stage-2 models (logistic ranker + logistic novelty, fold-out), and
breaks the validation MAP@5 down by query type.

  python src_v3/anatomy_default_v3.py            # build bundles (cached) + report
Outputs cache_v3/v3_outer_default_{species}.pkl.
"""
import os, sys, time, pickle
sys.path.insert(0, "src_v3"); sys.path.insert(0, "src_v2")
import numpy as np, pandas as pd
from scipy.special import softmax, expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cv_v3 import build_or_load_cv_v3
from features_v3 import build_v3_feature_bundle, DEFAULT_STAGE1_V3_CFG
from smoke_test_v3 import load_v3_raw_data

LEFT = {"left", "frontleft", "backleft"}; RIGHT = {"right", "frontright", "backright"}
def flank(v): return "L" if v in LEFT else "R" if v in RIGHT else "O"

def build(sp, raw, folds):
    path = f"cache_v3/v3_outer_default_{sp}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr = raw
    out = []
    for fd in folds:
        t0 = time.time()
        gi, vi = fd["gallery_idx"], fd["val_idx"]
        g_rows = gi[train_df["species"].values[gi] == sp]
        vm = train_df["species"].values[vi] == sp
        b = build_v3_feature_bundle(g_rows, vi[vm], sp, all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all,
                                    kp_counts, sift_csr, top_k_cands=20, stage1_cfg=DEFAULT_STAGE1_V3_CFG[sp])
        keep = {k: b[k] for k in ["cand_names", "stage1_top_scores", "X_cand", "X_open_sr", "eff_vp_q"]}
        keep.update(g_rows=g_rows, v_rows=vi[vm], y_true=fd["val_labels"][vm], is_new=fd["val_is_new"][vm])
        out.append(keep)
        print(f"[{sp}] fold {fd['fold']} bundle in {time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return out

def ap_of(names, truth):
    hit = names[:, :5] == truth[:, None]
    return np.where(hit.any(1), 1.0 / (hit.argmax(1) + 1), 0.0)

def stage2_lite(B):
    """Fold-out logistic ranker on X_cand and logistic novelty on X_open_sr + ranker margins."""
    res = []
    for k in range(len(B)):
        tr = [b for j, b in enumerate(B) if j != k]
        Xc = np.vstack([b["X_cand"][~b["is_new"]].reshape(-1, b["X_cand"].shape[2]) for b in tr])
        yc = np.concatenate([(b["cand_names"][~b["is_new"]] == b["y_true"][~b["is_new"], None]).ravel() for b in tr])
        rk = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=400)).fit(Xc, yc)
        def cand_scores(b):
            n, K, F = b["X_cand"].shape
            return rk.decision_function(b["X_cand"].reshape(-1, F)).reshape(n, K)
        def open_x(b, s):
            ss = -np.sort(-s, 1)
            return np.c_[b["X_open_sr"], ss[:, 0], ss[:, 0] - ss[:, 1], ss[:, 0] - ss[:, 4]]
        Xo = np.vstack([open_x(b, cand_scores(b)) for b in tr]); yo = np.concatenate([b["is_new"] for b in tr])
        op = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=600)).fit(Xo, yo)
        b = B[k]; s = cand_scores(b)
        pn = op.predict_proba(open_x(b, s))[:, 1]                    # trained at fold-native new rate
        res.append((s, pn))
    return res

def decode(b, s, pn, temp=1.0):
    pk = softmax(s / temp, axis=1) * (1 - pn[:, None])
    names = np.c_[b["cand_names"], np.full(len(pn), "new_individual")]
    P = np.c_[pk, pn]
    return np.take_along_axis(names, np.argsort(-P, 1)[:, :5], 1)

if __name__ == "__main__":
    raw = load_v3_raw_data(); folds = build_or_load_cv_v3()
    train_df = raw[0]; vp_all = train_df["viewpoint"].fillna("MISSING").values; ids_all = train_df["identity"].values
    rows = []
    for sp in ["hyena", "leopard"]:
        B = build(sp, raw, folds)
        S2 = stage2_lite(B)
        for b, (s, pn) in zip(B, S2):
            gc = pd.Series(ids_all[b["g_rows"]]).value_counts()
            gal_flanks = {}
            for r in b["g_rows"]:
                gal_flanks.setdefault(ids_all[r], set()).add(flank(vp_all[r]))
            ap_st1 = ap_of(b["cand_names"], b["y_true"])
            pred = decode(b, s, pn)
            ap = ap_of(pred, b["y_true"])
            rank_known = np.argsort(-s, 1)
            top1_known = b["cand_names"][np.arange(len(s)), rank_known[:, 0]] == b["y_true"]
            in20 = (b["cand_names"] == b["y_true"][:, None]).any(1)
            for i in range(len(ap)):
                yt = b["y_true"][i]; new = b["is_new"][i]
                fq = flank(vp_all[b["v_rows"][i]])
                if new: cat = "new"
                else:
                    n = gc.get(yt, 0)
                    xv = fq in ("L", "R") and fq not in gal_flanks.get(yt, set())
                    cat = "known_xview" if xv else ("known_rare(1-2)" if n <= 2 else "known_same")
                rows.append(dict(sp=sp, cat=cat, ap=ap[i], ap_stage1_noNew=ap_st1[i], top1_known=top1_known[i],
                                 in_top20=in20[i], new_first=pred[i, 0] == "new_individual", p_new=pn[i]))
    df = pd.DataFrame(rows)
    df.to_csv("cache_v3/anatomy_default_rows.csv", index=False)
    pd.set_option("display.width", 200)
    for sp in ["hyena", "leopard"]:
        d = df[df.sp == sp]
        g = d.groupby("cat").agg(n=("ap", "size"), share=("ap", lambda x: len(x) / len(d)), MAP5=("ap", "mean"),
                                 known_top1=("top1_known", "mean"), in_top20=("in_top20", "mean"), new_first=("new_first", "mean"))
        g["points_lost"] = g["share"] * (1 - g["MAP5"])
        print(f"\n=== {sp}: overall MAP@5 {d.ap.mean():.4f} (fold-native new rate {(d.cat=='new').mean():.3f})")
        print(g.round(4).to_string())
