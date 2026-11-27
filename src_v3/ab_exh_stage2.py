"""A/B on v3's 5 outer CV folds (leopard): stage-2 with v3 features (A) vs. v3 + exhaustive-SIFT
features (B). Same fixed LightGBM settings for both (amigo v1 'safe' params), leave-one-fold-out,
novelty model trained on nested out-of-fold candidate scores.

  python src_v3/ab_exh_stage2.py
"""
import os, sys, pickle
sys.path.insert(0, "src_v3"); sys.path.insert(0, "src_v2")
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.special import softmax, expit, logit
from exh_sift_features import add_exh_features

P_BIN = dict(n_estimators=240, learning_rate=0.03, num_leaves=15, max_depth=5, min_child_samples=30,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
             n_jobs=8, verbose=-1, random_state=7)
P_RANK = dict(P_BIN, objective="lambdarank")
P_OPEN = dict(n_estimators=220, learning_rate=0.03, num_leaves=12, max_depth=4, min_child_samples=25,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=6.0,
              n_jobs=8, verbose=-1, random_state=7)
NEW = "new_individual"
GAMMA = 1.5


def z(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)


def fit_cand(parts):
    """parts: list of dicts with Xc (n,K,F), names (n,K), y (n,), new (n,)."""
    Xb, yb, Xr, yr, gr = [], [], [], [], []
    for p in parts:
        n, K, F = p["Xc"].shape
        hit = p["names"] == p["y"][:, None]
        keep_b = ~p["new"] | (np.arange(n) % 4 == 0)
        Xb.append(p["Xc"][keep_b].reshape(-1, F)); yb.append(hit[keep_b].ravel())
        keep_r = ~p["new"] & hit.any(1)
        Xr.append(p["Xc"][keep_r].reshape(-1, F)); yr.append(hit[keep_r].ravel()); gr += [K] * int(keep_r.sum())
    mb = lgb.LGBMClassifier(**P_BIN).fit(np.vstack(Xb), np.concatenate(yb))
    mr = lgb.LGBMRanker(**P_RANK).fit(np.vstack(Xr), np.concatenate(yr).astype(int), group=gr)

    def score(p):
        n, K, F = p["Xc"].shape
        flat = p["Xc"].reshape(-1, F)
        pb = np.clip(mb.predict_proba(flat)[:, 1], 1e-6, 1 - 1e-6)
        return 0.5 * z(logit(pb).reshape(n, K)) + 0.5 * z(mr.predict(flat).reshape(n, K))
    return score


def open_x(p, s):
    ss = -np.sort(-s, 1); pr = softmax(GAMMA * s, 1)
    ent = -(pr * np.log(pr + 1e-12)).sum(1)
    return np.c_[p["Xo"], ss[:, 0], ss[:, 0] - ss[:, 1], ss[:, 0] - ss[:, 4], pr.max(1), ent]


def predict_fold(parts, k):
    tr = [j for j in range(len(parts)) if j != k]
    # nested out-of-fold candidate scores for the novelty model's training rows
    Xo, yo = [], []
    for j in tr:
        sc = fit_cand([parts[i] for i in tr if i != j])
        Xo.append(open_x(parts[j], sc(parts[j]))); yo.append(parts[j]["new"])
    mo = lgb.LGBMClassifier(**P_OPEN).fit(np.vstack(Xo), np.concatenate(yo))
    sc = fit_cand([parts[i] for i in tr])
    s = sc(parts[k])
    pn = np.clip(mo.predict_proba(open_x(parts[k], s))[:, 1], 1e-6, 1 - 1e-6)
    return s, pn, float(np.concatenate(yo).mean())


def decode(names, s, pn):
    P = np.c_[softmax(GAMMA * s, 1) * (1 - pn[:, None]), pn]
    lab = np.c_[names, np.full(len(pn), NEW)]
    return np.take_along_axis(lab, np.argsort(-P, 1)[:, :5], 1)


def ap(pred, y):
    hit = pred == y[:, None]
    return np.where(hit.any(1), 1.0 / (hit.argmax(1) + 1), 0.0)


def evaluate(parts, preds, target_pi=0.44):
    rows = []
    for p, (s, pn, pi_tr) in zip(parts, preds):
        a_native = ap(decode(p["names"], s, pn), p["y"])
        pn_t = expit(logit(pn) + logit(target_pi) - logit(pi_tr))       # prior shift to the test-like rate
        a_t = ap(decode(p["names"], s, pn_t), p["y"])
        new = p["new"]
        rows.append(dict(native=a_native.mean(), known=a_t[~new].mean(), new=a_t[new].mean(),
                         at_target=target_pi * a_t[new].mean() + (1 - target_pi) * a_t[~new].mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    B = pickle.load(open("cache_v3/v3_outer_default_leopard.pkl", "rb"))
    E = pickle.load(open("cache_v3/exh_feats_cv_leopard.pkl", "rb"))
    base = [dict(Xc=b["X_cand"], Xo=b["X_open_sr"], names=b["cand_names"], y=b["y_true"], new=b["is_new"]) for b in B]
    plus = []
    for b, e in zip(B, E):
        Xc, Xo = add_exh_features(b, e)
        plus.append(dict(Xc=Xc, Xo=Xo, names=b["cand_names"], y=b["y_true"], new=b["is_new"]))
    out = {}
    for name, parts in [("A_v3", base), ("B_v3+exhSIFT", plus)]:
        preds = [predict_fold(parts, k) for k in range(len(parts))]
        out[name] = evaluate(parts, preds)
        pickle.dump(preds, open(f"cache_v3/ab_preds_{name}.pkl", "wb"))
        print(f"\n{name}  (per fold; at_target = reweighted to 44% new with prior-shifted P(new))")
        print(out[name].round(4).to_string()); print("mean", out[name].mean().round(4).to_dict(), flush=True)
    d = out["B_v3+exhSIFT"] - out["A_v3"]
    print("\nB - A per fold:"); print(d.round(4).to_string())
    print("mean delta", d.mean().round(4).to_dict(), "| folds improved (at_target):", int((d.at_target > 0).sum()), "/ 5")
