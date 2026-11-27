"""
100% Strictly Rule-Compliant V3 Final Reproduction Script (ZERO Optuna import!).
Uses ONLY competition-listed libraries:
  numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost (+ opencv-python in offline SIFT cache builder).

Loads the pre-discovered V3 hyperparameters from `cache_v3/optuna_stage2_3_v3_best_params.json`
and `cache_v3/v3_final_polished_stage3_params.json`, runs the 3-seed (360-model) clean ensemble
(3 Listwise Rankers + 5 Boosted/ET Binary Rerankers + 4 Platt-Calibrated Boosted Open-Set Detectors)
+ Symmetric Verified SIFT+MiewID Cluster-Consensus Solver, verifies format/species integrity, and writes
`submissions_v3/submission_v3_reproduced_no_optuna.csv`.
"""
import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from cv_v3 import PRIOR_PI_NEW_V3, DATA_DIR, CACHE_V3_DIR

warnings.filterwarnings("ignore")

TOP4_OPEN = ["cat_open", "lgb_open", "xgb_open", "hgb_open"]


def clip_logit(p):
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def row_zscore(M):
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-4)


def _apply_v3_ultra_guard(p_new, raw_s1, max_mov_inl, z60):
    if raw_s1 is None or max_mov_inl is None:
        return p_new
    ultra = (raw_s1 >= 0.76) & (max_mov_inl >= 10.0)
    if z60 is not None:
        ultra = ultra | ((raw_s1 >= 0.50) & (z60 >= 1.6) & (max_mov_inl >= 20.0))
        ultra = ultra | ((raw_s1 >= 0.40) & (z60 >= 0.8) & (max_mov_inl >= 40.0))
    return np.where(ultra, np.minimum(p_new, 0.10), p_new)


def decode_top5(cand_names, cand_probs, open_logits, delta=0.0, raw_s1=None, max_mov_inl=None, z60=None):
    ord5 = np.argsort(-cand_probs, axis=1)[:, :5]
    top5_names = np.take_along_axis(cand_names, ord5, axis=1)
    cp5 = np.take_along_axis(cand_probs, ord5, axis=1)
    p_new = _apply_v3_ultra_guard(expit(open_logits + delta), raw_s1, max_mov_inl, z60)
    preds = []
    for i in range(len(cand_names)):
        pn = float(p_new[i])
        kp = (1.0 - pn) * cp5[i]
        items = [("new_individual", pn)] + [(top5_names[i, r], float(kp[r])) for r in range(5)]
        items.sort(key=lambda x: -x[1])
        preds.append([x[0] for x in items[:5]])
    return preds


def apply_symmetric_cluster_consensus(bundle, cand_scores, open_logits, p):
    W_csr, V_csr = bundle["W_qq_csr"], bundle["V_edge_csr"]
    w_sum, has_qq = bundle["w_qq_sum"][:, 0], bundle["has_qq"]
    v_comp, nc_v, cand_names = bundle["v_comp"], bundle["nc_v"], bundle["cand_names"]

    open_out = open_logits.copy()
    if np.any(has_qq) and p["a_o1"] > 1e-4:
        nbr_o = (W_csr @ open_out) / np.maximum(1e-8, w_sum)
        open_out[has_qq] = (1.0 - p["a_o1"]) * open_out[has_qq] + p["a_o1"] * nbr_o[has_qq]

    has_ve = np.asarray(V_csr.sum(axis=1)).ravel() > 1e-6
    if np.any(has_ve) and p["a_bp_o"] > 1e-4:
        for _ in range(2):
            node_w = (1.0 + p["anchor_w"] * np.minimum(4.0, np.abs(open_out))).astype(np.float64)
            num, den = V_csr @ (node_w * open_out), V_csr @ node_w
            m_ok = has_ve & (den > 1e-6)
            bp_msg = np.where(m_ok, num / np.maximum(1e-8, den), open_out)
            open_out[m_ok] = (1.0 - 0.50 * p["a_bp_o"]) * open_out[m_ok] + (0.50 * p["a_bp_o"]) * bp_msg[m_ok]

    v_members = {cid: np.where(v_comp == cid)[0] for cid in range(nc_v) if (v_comp == cid).sum() >= 2}
    if len(v_members) > 0 and p["a_bp_o"] > 1e-4:
        cc_o = open_out.copy()
        a_cc = 0.55 * p["a_bp_o"]
        for cid, idx_c in v_members.items():
            sub_o = open_out[idx_c]
            nw = 1.0 + p["anchor_w"] * np.minimum(4.0, np.abs(sub_o))
            c_target = float(np.sum(nw * sub_o) / np.sum(nw))
            cc_o[idx_c] = (1.0 - a_cc) * sub_o + a_cc * c_target
        open_out = cc_o

    probs = softmax(p["gamma"] * cand_scores, axis=1)
    cn_out, cp_out = cand_names.copy(), probs.copy()
    K = cand_names.shape[1]
    a_c1 = 0.8 * p["a_o1"]

    if (np.any(has_qq) and a_c1 > 1e-4) or (len(v_members) > 0 and p["a_vc_c"] > 1e-4):
        comp_prob_map = {}
        if p["a_vc_c"] > 1e-4:
            for cid, idx_c in v_members.items():
                fw = 0.40 + p["anchor_w"] * (1.0 - expit(open_out[idx_c]))
                fw /= float(fw.sum())
                cmap = {}
                for nj, wj in zip(idx_c, fw):
                    for r in range(min(10, K)):
                        c = cand_names[nj, r]
                        cmap[c] = cmap.get(c, 0.0) + wj * probs[nj, r]
                comp_prob_map[cid] = cmap

        for i in range(len(cand_names)):
            do_1h = has_qq[i] and a_c1 > 1e-4
            cid_i = v_comp[i]
            do_vc = (cid_i in comp_prob_map) and p["a_vc_c"] > 1e-4
            if not (do_1h or do_vc):
                continue
            w1 = a_c1 if do_1h else 0.0
            wv = p["a_vc_c"] if do_vc else 0.0
            if w1 + wv > 0.72:
                sc = 0.72 / (w1 + wv)
                w1 *= sc; wv *= sc
            w_self = 1.0 - w1 - wv
            smap = {}
            for r in range(K):
                c = cand_names[i, r]
                smap[c] = smap.get(c, 0.0) + w_self * probs[i, r]
            if do_1h:
                s, e = W_csr.indptr[i], W_csr.indptr[i + 1]
                for nj, wd in zip(W_csr.indices[s:e], W_csr.data[s:e]):
                    aw = w1 * (wd / max(1e-8, w_sum[i]))
                    for r in range(min(10, K)):
                        c = cand_names[nj, r]
                        smap[c] = smap.get(c, 0.0) + aw * probs[nj, r]
            if do_vc:
                for c, pv in comp_prob_map[cid_i].items():
                    smap[c] = smap.get(c, 0.0) + wv * pv
            for r, (c, pr) in enumerate(sorted(smap.items(), key=lambda x: -x[1])[:K]):
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


def get_open_base(bundle, species):
    if species == "hyena":
        return np.hstack([bundle["X_open_sr"], bundle["X_open_graph"]]).astype(np.float32)
    return bundle["X_open_sr"]


def fit_clean_stage2(sub_bundles, target_bundle, species, hp, seed=42):
    pi_new = PRIOR_PI_NEW_V3[species]
    n_sf = len(sub_bundles)
    ckeys = ["lgb_rank", "xgb_rank", "cat_rank", "lgb_bin", "xgb_bin", "cat_bin", "hgb_bin", "et_bin"]
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

        m = lgb.LGBMRanker(objective="lambdarank", eval_at=[1, 5], subsample_freq=1, n_jobs=4, random_state=seed + 11 * hs, verbose=-1, **hp["lgb_rank"]).fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["lgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["lgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = xgb.XGBRanker(objective="rank:ndcg", n_jobs=4, random_state=seed + 13 * hs, verbosity=0, **hp["xgb_rank"]).fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["xgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["xgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = CatBoostRanker(loss_function="QuerySoftMax", thread_count=4, random_seed=seed + 17 * hs, verbose=0, **hp["cat_rank"]).fit(Pool(data=Xtr_r, label=ytr_r.astype(np.float32), group_id=qids_r))
        oof_c["cat_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["cat_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 19 * hs, verbose=-1, **hp["lgb_bin"]).fit(Xtr_b, ytr_b)
        oof_c["lgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["lgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = xgb.XGBClassifier(n_jobs=4, random_state=seed + 23 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_bin"]).fit(Xtr_b, ytr_b)
        oof_c["xgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["xgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = CatBoostClassifier(iterations=hp["cat_rank"]["iterations"], learning_rate=hp["cat_rank"]["learning_rate"], depth=hp["cat_rank"]["depth"], l2_leaf_reg=hp["cat_rank"]["l2_leaf_reg"], thread_count=4, random_seed=seed + 29 * hs, verbose=0).fit(Xtr_b, ytr_b)
        oof_c["cat_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["cat_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = HistGradientBoostingClassifier(random_state=seed + 31 * hs, **hp["hgb_bin"]).fit(Xtr_b, ytr_b)
        oof_c["hgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["hgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = ExtraTreesClassifier(n_estimators=180, max_depth=14, min_samples_leaf=8, max_features="sqrt", n_jobs=4, random_state=seed + 33 * hs).fit(Xtr_b, ytr_b)
        oof_c["et_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["et_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

    def make_stacked(b_i, bin_m, rnk_m):
        X_ob = get_open_base(b_i, species)
        sb, sr = -np.sort(-bin_m, axis=1), -np.sort(-rnk_m, axis=1)
        pb, pr = softmax(bin_m, axis=1), softmax(1.5 * rnk_m, axis=1)
        eb = -np.sum(pb * np.log(np.clip(pb, 1e-12, None)), axis=1)
        er = -np.sum(pr * np.log(np.clip(pr, 1e-12, None)), axis=1)
        return np.column_stack([X_ob, sb[:, 0], sb[:, 0] - sb[:, 1], pb.max(1), eb, sr[:, 0], sr[:, 0] - sr[:, 1], pr.max(1), er]).astype(np.float32)

    sub_open = [
        make_stacked(
            sub_bundles[s],
            0.27 * oof_c["hgb_bin"][s] + 0.25 * oof_c["cat_bin"][s] + 0.24 * oof_c["lgb_bin"][s] + 0.19 * oof_c["xgb_bin"][s] + 0.05 * oof_c["et_bin"][s],
            (oof_c["cat_rank"][s] + oof_c["lgb_rank"][s] + oof_c["xgb_rank"][s]) / 3.0,
        )
        for s in range(n_sf)
    ]
    X_t_open = make_stacked(
        target_bundle,
        0.27 * tgt_c["hgb_bin"] + 0.25 * tgt_c["cat_bin"] + 0.24 * tgt_c["lgb_bin"] + 0.19 * tgt_c["xgb_bin"] + 0.05 * tgt_c["et_bin"],
        (tgt_c["cat_rank"] + tgt_c["lgb_rank"] + tgt_c["xgb_rank"]) / 3.0,
    )

    tgt_o = {k: np.zeros(nt, dtype=np.float64) for k in TOP4_OPEN}
    for hs in range(n_sf):
        Xo_l, yo_l, wo_l = [], [], []
        for idx_s, sb in enumerate(sub_bundles):
            if idx_s == hs:
                continue
            isn = sb["is_new"]
            Xo_l.append(sub_open[idx_s])
            yo_l.append(isn.astype(np.int32))
            nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
            wo_l.append(np.where(isn, pi_new / nn, (1.0 - pi_new) / nk) * len(isn))
        Xo_tr, yo_tr, wo_tr = np.vstack(Xo_l), np.concatenate(yo_l), np.concatenate(wo_l)

        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 43 * hs, verbose=-1, **hp["lgb_open"]).fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["lgb_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf
        m = xgb.XGBClassifier(n_jobs=4, random_state=seed + 47 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_open"]).fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["xgb_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf
        m = CatBoostClassifier(thread_count=4, random_seed=seed + 53 * hs, verbose=0, **hp["cat_open"]).fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["cat_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf
        m = HistGradientBoostingClassifier(random_state=seed + 59 * hs, **hp["hgb_open"]).fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["hgb_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

    return tgt_c, tgt_o


def main():
    t0 = time.time()
    # Opt-in: V3_BUNDLE_SUFFIX=_exh reads cache_v3/v3_bundles_{sp}_exh.pkl (exhaustive-SIFT features added by
    # src_v3/add_exh_to_bundles.py) and writes ..._reproduced_no_optuna_exh.csv. Unset = original behaviour.
    suffix = os.environ.get("V3_BUNDLE_SUFFIX", "")
    os.makedirs("submissions_v3", exist_ok=True)
    with open(os.path.join(CACHE_V3_DIR, "optuna_stage2_3_v3_best_params.json"), "r") as f:
        hp_all = json.load(f)["stage2_hp"]
    with open(os.path.join(CACHE_V3_DIR, "v3_final_polished_stage3_params.json"), "r") as f:
        s3_all = json.load(f)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_preds = [None] * len(test_df)

    for sp in ["hyena", "leopard"]:
        hp = hp_all[sp]
        s3 = s3_all[sp]
        p, cal_p, ow = s3["stage3"], s3["platt"], s3["open_weights"]
        with open(os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}{suffix}.pkl") if os.path.exists(os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}{suffix}.pkl")) else os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}.pkl"), "rb") as f:
            data = pickle.load(f)
        test_b, full_seed_bundles = data["test_bundle"], data["full_seed_bundles"]
        run_specs = [(2026, 2026), (7777, 7777), (2026, 9999)]
        cand_ens, open_ens = None, None
        for split_sd, model_sd in run_specs:
            c_sd, o_sd = fit_clean_stage2(full_seed_bundles[split_sd], test_b, sp, hp, seed=model_sd)
            ws = 1.0 / len(run_specs)
            if cand_ens is None:
                cand_ens = {k: v * ws for k, v in c_sd.items()}
                open_ens = {k: v * ws for k, v in o_sd.items()}
            else:
                for k in c_sd:
                    cand_ens[k] += c_sd[k] * ws
                for k in o_sd:
                    open_ens[k] += o_sd[k] * ws

        c_rnk = 0.44 * cand_ens["cat_rank"] + 0.28 * cand_ens["lgb_rank"] + 0.28 * cand_ens["xgb_rank"]
        c_bin = (
            0.27 * row_zscore(cand_ens["hgb_bin"]) + 0.25 * row_zscore(cand_ens["cat_bin"])
            + 0.24 * row_zscore(cand_ens["lgb_bin"]) + 0.19 * row_zscore(cand_ens["xgb_bin"])
            + 0.05 * row_zscore(cand_ens["et_bin"])
        )
        c_tot = (1.0 - p["mix_rank"]) * row_zscore(c_bin) + p["mix_rank"] * row_zscore(c_rnk)
        o_tot = np.zeros(len(c_tot), dtype=np.float64)
        for ok in TOP4_OPEN:
            a, b = cal_p[ok]
            o_tot += ow[ok] * (a * open_ens[ok] + b)

        cn, cp, ol = apply_symmetric_cluster_consensus(test_b, c_tot, o_tot, p)
        pr_t = decode_top5(cn, cp, ol, delta=p["delta"], raw_s1=test_b["raw_s1"], max_mov_inl=test_b["max_mov_inl"], z60=test_b["z60"])
        for loc_i, te_pos in enumerate(test_b["te_pos"]):
            test_preds[te_pos] = " ".join(pr_t[loc_i])
        print(f"[REPRODUCE V3 NO-OPTUNA] Finished {sp.upper()}!")

    out_path = f"submissions_v3/submission_v3_reproduced_no_optuna{suffix}.csv"
    pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds}).to_csv(out_path, index=False)
    print(f"Saved strictly rule-compliant V3 reproduction to {out_path} in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
