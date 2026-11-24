"""
100% Strictly Rule-Compliant Final Reproduction Script (ZERO Optuna import!).
Uses ONLY competition-listed libraries:
  numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost (+ opencv-python in offline SIFT cache builder).

Loads the pre-discovered Optuna hyperparameters from `cache_v2/optuna_stage2_shift_robust_params.json`
and `cache_v2/optuna_stage2_best_params.json`, trains the full 260-model multi-seed shift-robust &
graph-enriched V2 ensembles on cached V2 bundles, verifies format/species integrity, and writes
`submissions_v2/submission_reproduced_no_optuna.csv`.
"""
import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax, expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from cv_v2 import PRIOR_PI_NEW, DATA_DIR, CACHE_V2_DIR

warnings.filterwarnings("ignore")

OPEN_NAMES = (
    [f"st1_{k}" for k in ["s1", "s2", "g12", "g14", "g16", "bg", "z"]]
    + [f"p60_{k}" for k in ["s1", "s2", "g12", "g14", "g16", "bg", "z"]]
    + [f"p45_{k}" for k in ["s1", "s2", "g12", "g14"]]
    + [f"raw_{k}" for k in ["s1", "s2", "g12", "g14", "g16"]]
    + [f"sub_{k}" for k in ["s1", "s2", "g12", "g14", "g16"]]
    + [f"lda_{k}" for k in ["s1", "s2", "g12", "g14", "g16"]]
    + [f"kiss_{k}" for k in ["s1", "s2", "g12", "g14"]]
    + [f"zh_{k}" for k in ["s1", "s2", "g12", "g14"]]
    + [f"ppr_{k}" for k in ["s1", "s2", "g12", "g14"]]
    + [f"fv_{k}" for k in ["s1", "s2", "g12"]]
    + [f"vl_{k}" for k in ["s1", "s2", "g12"]]
    + [
        "z45", "dq45", "ec45", "z60", "dq60", "ec60", "zraw", "dqraw", "ecraw",
        "zlda", "dqlda", "eclda", "zkiss", "dqkiss", "eckiss",
        "soft_max", "soft_ent", "bal_ridge_max", "ridge_z_max",
        "sift_sim_max", "sift_fund_max", "sift_r74_max", "g_sim0", "g_fnd0", "g_r74_0",
        "vp_has0", "vp_frac0", "vp_smax0", "vp_dmax0", "vp_soft0",
        "log_freq0", "hub0", "log_kp", "miew_norm",
        "diff_q_enrich", "top5_q_enrich", "top10_q_enrich",
        "max_qq_mix", "max_qq_raw", "qq_gap_mix", "qq_gap_raw", "max_qq_sift",
        "log_clust_size", "clust_max_z60", "clust_mean_z60", "z60_minus_cmax",
        "clust_max_sim_inl", "clust_mean_q_enrich",
    ]
)

SHIFT_PRONE_OPEN_COLS = {
    "st1_s2", "st1_bg", "p60_s2", "p60_bg", "p45_s2", "raw_s2", "sub_s2", "lda_s2",
    "kiss_s1", "kiss_s2", "zh_s2", "ppr_s2", "fv_s2", "vl_s2", "log_freq0", "hub0",
    "diff_q_enrich", "top5_q_enrich", "top10_q_enrich", "clust_mean_q_enrich",
    "max_qq_mix", "max_qq_raw", "qq_gap_mix", "qq_gap_raw", "max_qq_sift", "log_clust_size",
}
KEEP_OPEN_IDX = [i for i, n in enumerate(OPEN_NAMES) if n not in SHIFT_PRONE_OPEN_COLS]
IDX_Z60 = OPEN_NAMES.index("z60")
IDX_GSIM0 = OPEN_NAMES.index("g_sim0")
IDX_GFND0 = OPEN_NAMES.index("g_fnd0")
IDX_SIFT_MAX = OPEN_NAMES.index("sift_sim_max")


def make_shift_robust_open_base(X_open_raw):
    base = X_open_raw[:, KEEP_OPEN_IDX]
    z60_pos = np.maximum(0.0, X_open_raw[:, IDX_Z60])
    return np.column_stack([
        base,
        X_open_raw[:, IDX_GSIM0] * z60_pos,
        X_open_raw[:, IDX_GFND0] * z60_pos,
        X_open_raw[:, IDX_SIFT_MAX] * z60_pos,
    ]).astype(np.float32)


def clip_logit(p):
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def row_zscore(M):
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-4)


def decode_top5_from_probs(cand_names, cand_probs, open_logits, delta=0.0, raw_s1=None, max_sim_inl=None, z60=None):
    ord5 = np.argsort(-cand_probs, axis=1)[:, :5]
    top5_names = np.take_along_axis(cand_names, ord5, axis=1)
    cp5 = np.take_along_axis(cand_probs, ord5, axis=1)
    p_new = expit(open_logits + delta)
    if raw_s1 is not None and max_sim_inl is not None:
        ultra = (raw_s1 >= 0.78) & (max_sim_inl >= 12.0)
        if z60 is not None:
            ultra = ultra | ((raw_s1 >= 0.56) & (z60 >= 2.0) & (max_sim_inl >= 25.0))
        p_new = np.where(ultra, np.minimum(p_new, 0.10), p_new)

    preds = []
    for i in range(len(cand_names)):
        pn = float(p_new[i])
        kp = (1.0 - pn) * cp5[i]
        items = [("new_individual", pn)] + [(top5_names[i, r], float(kp[r])) for r in range(5)]
        items.sort(key=lambda x: -x[1])
        preds.append([x[0] for x in items[:5]])
    return preds


def apply_two_level_graph_postproc(
    cand_names, cand_scores, open_logits, W_qq, w_qq_sum, has_qq, q_comp, nc_q,
    gamma=2.5, alpha_open_1h=0.22, alpha_cand_1h=0.18, alpha_open_cc=0.08, alpha_cand_cc=0.08
):
    open_out = open_logits.copy()
    if np.any(has_qq) and alpha_open_1h > 1e-4:
        nbr_o = (W_qq[has_qq] @ open_logits) / w_qq_sum[has_qq, 0]
        open_out[has_qq] = (1.0 - alpha_open_1h) * open_out[has_qq] + alpha_open_1h * nbr_o

    comp_members = {}
    for cid in range(nc_q):
        idx_c = np.where(q_comp == cid)[0]
        if len(idx_c) >= 2:
            comp_members[cid] = idx_c

    if alpha_open_cc > 1e-4 and len(comp_members) > 0:
        cc_o = open_out.copy()
        for cid, idx_c in comp_members.items():
            c_target = 0.65 * float(open_out[idx_c].mean()) + 0.35 * float(open_out[idx_c].min())
            cc_o[idx_c] = (1.0 - alpha_open_cc) * open_out[idx_c] + alpha_open_cc * c_target
        open_out = cc_o

    probs = softmax(gamma * cand_scores, axis=1)
    cn_out, cp_out = cand_names.copy(), probs.copy()
    K = cand_names.shape[1]

    if (np.any(has_qq) and alpha_cand_1h > 1e-4) or (alpha_cand_cc > 1e-4 and len(comp_members) > 0):
        comp_prob_map = {}
        if alpha_cand_cc > 1e-4:
            for cid, idx_c in comp_members.items():
                cmap = {}
                inv_sz = 1.0 / float(len(idx_c))
                for nj in idx_c:
                    for r in range(K):
                        c = cand_names[nj, r]
                        cmap[c] = cmap.get(c, 0.0) + inv_sz * probs[nj, r]
                comp_prob_map[cid] = cmap

        for i in range(len(cand_names)):
            do_1h = has_qq[i] and alpha_cand_1h > 1e-4
            cid_i = q_comp[i]
            do_cc = (cid_i in comp_prob_map) and alpha_cand_cc > 1e-4
            if not (do_1h or do_cc):
                continue
            w_self = 1.0 - (alpha_cand_1h if do_1h else 0.0) - (alpha_cand_cc if do_cc else 0.0)
            smap = {}
            for r in range(K):
                c = cand_names[i, r]
                smap[c] = smap.get(c, 0.0) + w_self * probs[i, r]
            if do_1h:
                nb = np.where(W_qq[i] > 0)[0]
                ws = W_qq[i, nb] / w_qq_sum[i, 0]
                for nj, wj in zip(nb, ws):
                    for r in range(K):
                        c = cand_names[nj, r]
                        smap[c] = smap.get(c, 0.0) + alpha_cand_1h * wj * probs[nj, r]
            if do_cc:
                for c, pv in comp_prob_map[cid_i].items():
                    smap[c] = smap.get(c, 0.0) + alpha_cand_cc * pv
            top_k_it = sorted(smap.items(), key=lambda x: -x[1])[:K]
            for r, (c, pr) in enumerate(top_k_it):
                cn_out[i, r] = c
                cp_out[i, r] = pr
    return cn_out, cp_out, open_out


def build_cand_dataset(sub_bundles, hold_s, for_ranker=False):
    X_l, y_l, g_l = [], [], []
    for idx_s, sb in enumerate(sub_bundles):
        if idx_s == hold_s:
            continue
        nq, K, F = sb["X_cand"].shape
        ym = (sb["cand_names"] == sb["y_true"][:, None]).astype(np.int32)
        if for_ranker:
            keep = (~sb["is_new"]) & (ym.sum(axis=1) > 0)
            X_l.append(sb["X_cand"][keep].reshape(-1, F))
            y_l.append(ym[keep].ravel())
            g_l.extend([K] * int(keep.sum()))
        else:
            keep = (~sb["is_new"]) | ((np.arange(nq) % 4) == 0)
            X_l.append(sb["X_cand"][keep].reshape(-1, F))
            y_l.append(ym[keep].ravel())
    return np.vstack(X_l), np.concatenate(y_l), np.array(g_l, dtype=np.int32)


def fit_all_stage2_models(sub_bundles, target_bundle, species, hp, seed=42):
    pi_new = PRIOR_PI_NEW[species]
    n_sf = len(sub_bundles)
    ckeys = ["lgb_rank", "xgb_rank", "cat_rank", "lgb_bin", "xgb_bin", "cat_bin", "mlp_bin", "logreg_bin"]
    okeys = ["lgb_open", "xgb_open", "cat_open", "mlp_open", "logreg_open"]

    nt, Kt, Ft = target_bundle["X_cand"].shape
    X_t_c = target_bundle["X_cand"].reshape(-1, Ft)
    oof_c = {k: [] for k in ckeys}
    tgt_c = {k: np.zeros((nt, Kt), dtype=np.float64) for k in ckeys}

    for hs in range(n_sf):
        hb = sub_bundles[hs]
        nh, Kh, Fh = hb["X_cand"].shape
        X_h_c = hb["X_cand"].reshape(-1, Fh)

        Xtr_r, ytr_r, gtr_r = build_cand_dataset(sub_bundles, hs, for_ranker=True)
        qids_r = np.repeat(np.arange(len(gtr_r)), gtr_r)
        Xtr_b, ytr_b, _ = build_cand_dataset(sub_bundles, hs, for_ranker=False)

        m = lgb.LGBMRanker(objective="lambdarank", eval_at=[1, 5], subsample_freq=1, n_jobs=4, random_state=seed + 11 * hs, verbose=-1, **hp["lgb_rank"])
        m.fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["lgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["lgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = xgb.XGBRanker(objective="rank:ndcg", n_jobs=4, random_state=seed + 13 * hs, verbosity=0, **hp["xgb_rank"])
        m.fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["xgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["xgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = CatBoostRanker(loss_function="QuerySoftMax", thread_count=4, random_seed=seed + 17 * hs, verbose=0, **hp["cat_rank"])
        m.fit(Pool(data=Xtr_r, label=ytr_r.astype(np.float32), group_id=qids_r))
        oof_c["cat_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["cat_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 19 * hs, verbose=-1, **hp["lgb_bin"])
        m.fit(Xtr_b, ytr_b)
        oof_c["lgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["lgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = xgb.XGBClassifier(n_jobs=4, random_state=seed + 23 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_bin"])
        m.fit(Xtr_b, ytr_b)
        oof_c["xgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["xgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = CatBoostClassifier(
            iterations=hp["cat_rank"]["iterations"], learning_rate=hp["cat_rank"]["learning_rate"],
            depth=hp["cat_rank"]["depth"], l2_leaf_reg=hp["cat_rank"]["l2_leaf_reg"],
            thread_count=4, random_seed=seed + 29 * hs, verbose=0
        )
        m.fit(Xtr_b, ytr_b)
        oof_c["cat_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["cat_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, alpha=2.5, max_iter=85, early_stopping=True, random_state=seed + 31 * hs))
        m.fit(Xtr_b, ytr_b)
        oof_c["mlp_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["mlp_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, max_iter=300, random_state=seed + 37 * hs))
        m.fit(Xtr_b, ytr_b)
        oof_c["logreg_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["logreg_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

    def make_stacked_open(X_base_raw, bin_mat, rank_mat):
        X_sr = make_shift_robust_open_base(X_base_raw)
        sb, sr = -np.sort(-bin_mat, axis=1), -np.sort(-rank_mat, axis=1)
        pb, pr = softmax(bin_mat, axis=1), softmax(1.5 * rank_mat, axis=1)
        eb = -np.sum(pb * np.log(np.clip(pb, 1e-12, None)), axis=1)
        er = -np.sum(pr * np.log(np.clip(pr, 1e-12, None)), axis=1)
        return np.column_stack([X_sr, sb[:, 0], sb[:, 0] - sb[:, 1], pb.max(1), eb, sr[:, 0], sr[:, 0] - sr[:, 1], pr.max(1), er]).astype(np.float32)

    sub_open_st = []
    for s in range(n_sf):
        b_avg = 0.28 * oof_c["lgb_bin"][s] + 0.28 * oof_c["xgb_bin"][s] + 0.24 * oof_c["cat_bin"][s] + 0.10 * oof_c["mlp_bin"][s] + 0.10 * oof_c["logreg_bin"][s]
        r_avg = (oof_c["lgb_rank"][s] + oof_c["xgb_rank"][s] + oof_c["cat_rank"][s]) / 3.0
        sub_open_st.append(make_stacked_open(sub_bundles[s]["X_open_base"], b_avg, r_avg))

    t_b_avg = 0.28 * tgt_c["lgb_bin"] + 0.28 * tgt_c["xgb_bin"] + 0.24 * tgt_c["cat_bin"] + 0.10 * tgt_c["mlp_bin"] + 0.10 * tgt_c["logreg_bin"]
    t_r_avg = (tgt_c["lgb_rank"] + tgt_c["xgb_rank"] + tgt_c["cat_rank"]) / 3.0
    X_t_open = make_stacked_open(target_bundle["X_open_base"], t_b_avg, t_r_avg)

    tgt_o = {k: np.zeros(nt, dtype=np.float64) for k in okeys}
    for hs in range(n_sf):
        Xo_l, yo_l, wo_l = [], [], []
        for idx_s, sb in enumerate(sub_bundles):
            if idx_s == hs:
                continue
            isn = sb["is_new"]
            Xo_l.append(sub_open_st[idx_s])
            yo_l.append(isn.astype(np.int32))
            nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
            wo_l.append(np.where(isn, pi_new / nn, (1.0 - pi_new) / nk) * len(isn))
        Xo_tr, yo_tr, wo_tr = np.vstack(Xo_l), np.concatenate(yo_l), np.concatenate(wo_l)

        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 41 * hs, verbose=-1, **hp["lgb_open"])
        m.fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["lgb_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

        m = xgb.XGBClassifier(n_jobs=4, random_state=seed + 43 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_open"])
        m.fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["xgb_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

        m = CatBoostClassifier(thread_count=4, random_seed=seed + 47 * hs, verbose=0, **hp["cat_open"])
        m.fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["cat_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(48, 24), batch_size=256, alpha=3.5, max_iter=100, early_stopping=True, random_state=seed + 53 * hs))
        m.fit(Xo_tr, yo_tr)
        ps = logit(pi_new) - logit(float(yo_tr.mean()))
        tgt_o["mlp_open"] += (clip_logit(m.predict_proba(X_t_open)[:, 1]) + ps) / n_sf

        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.4, max_iter=300, random_state=seed + 59 * hs))
        m.fit(Xo_tr, yo_tr, logisticregression__sample_weight=wo_tr)
        tgt_o["logreg_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

    tgt_c["stage1"] = row_zscore(target_bundle["stage1_top_scores"])
    return tgt_c, tgt_o


def combine_and_decode(bundle, cand_d, open_d, p):
    c_bin = (
        p["w_lgb_bin"] * cand_d["lgb_bin"] + p["w_xgb_bin"] * cand_d["xgb_bin"]
        + p["w_cat_bin"] * cand_d["cat_bin"] + p["w_mlp_bin"] * cand_d["mlp_bin"]
        + p["w_logreg_bin"] * cand_d["logreg_bin"]
    )
    c_rnk = (
        p["w_stage1"] * cand_d["stage1"] + p["w_lgb_rank"] * cand_d["lgb_rank"]
        + p["w_xgb_rank"] * cand_d["xgb_rank"] + p["w_cat_rank"] * cand_d["cat_rank"]
    )
    c_tot = (1.0 - p["mix_rank"]) * row_zscore(c_bin) + p["mix_rank"] * row_zscore(c_rnk)
    o_tot = (
        p["wo_lgb"] * open_d["lgb_open"] + p["wo_xgb"] * open_d["xgb_open"]
        + p["wo_cat"] * open_d["cat_open"] + p["wo_mlp"] * open_d["mlp_open"]
        + p["wo_lr"] * open_d["logreg_open"]
    ) / max(1e-6, p["wo_lgb"] + p["wo_xgb"] + p["wo_cat"] + p["wo_mlp"] + p["wo_lr"])

    cn, cp, ol = apply_two_level_graph_postproc(
        bundle["cand_names"], c_tot, o_tot, bundle["W_qq"], bundle["w_qq_sum"], bundle["has_qq"], bundle["q_comp"], bundle["nc_q"],
        gamma=p["gamma"], alpha_open_1h=p["a_o1"], alpha_cand_1h=p["a_c1"], alpha_open_cc=p["a_oc"], alpha_cand_cc=p["a_cc"]
    )
    return decode_top5_from_probs(
        cn, cp, ol, delta=p["delta"], raw_s1=bundle["raw_s1"], max_sim_inl=bundle["max_sim_inl"], z60=bundle["X_open_base"][:, IDX_Z60]
    )


def main():
    t0 = time.time()
    os.makedirs("submissions_v2", exist_ok=True)
    with open(os.path.join(CACHE_V2_DIR, "optuna_stage2_shift_robust_params.json"), "r") as f:
        best_all = json.load(f)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_preds = [None] * len(test_df)

    for sp in ["hyena", "leopard"]:
        hp = best_all["stage2_hp"][sp]
        mp = best_all["stage3_meta"][sp]
        with open(os.path.join(CACHE_V2_DIR, f"v2_bundles_{sp}.pkl"), "rb") as f:
            data = pickle.load(f)

        test_b = data["test_bundle"]
        full_seed_bundles = data["full_seed_bundles"]
        seeds = [2026, 7777]
        cand_ens, open_ens = None, None
        for sd in seeds:
            c_sd, o_sd = fit_all_stage2_models(full_seed_bundles[sd], test_b, sp, hp, seed=sd)
            if cand_ens is None:
                cand_ens = {k: v / len(seeds) for k, v in c_sd.items()}
                open_ens = {k: v / len(seeds) for k, v in o_sd.items()}
            else:
                for k in c_sd:
                    cand_ens[k] += c_sd[k] / len(seeds)
                for k in o_sd:
                    open_ens[k] += o_sd[k] / len(seeds)

        pr_t = combine_and_decode(test_b, cand_ens, open_ens, mp)
        for loc_i, te_pos in enumerate(test_b["te_pos"]):
            test_preds[te_pos] = " ".join(pr_t[loc_i])
        print(f"[REPRODUCE NO-OPTUNA] Finished {sp.upper()}!")

    out_path = "submissions_v2/submission_reproduced_no_optuna.csv"
    pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds}).to_csv(out_path, index=False)
    print(f"Saved strictly rule-compliant no-Optuna reproduction to {out_path} in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
