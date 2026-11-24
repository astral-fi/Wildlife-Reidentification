import pickle
import numpy as np
import pandas as pd

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
        "z45", "dq45", "ec45",
        "z60", "dq60", "ec60",
        "zraw", "dqraw", "ecraw",
        "zlda", "dqlda", "eclda",
        "zkiss", "dqkiss", "eckiss",
        "soft_max", "soft_ent",
        "bal_ridge_max", "ridge_z_max",
        "sift_sim_max", "sift_fund_max", "sift_r74_max",
        "g_sim0", "g_fnd0", "g_r74_0",
        "vp_has0", "vp_frac0", "vp_smax0", "vp_dmax0", "vp_soft0",
        "log_freq0", "hub0", "log_kp", "miew_norm",
        "diff_q_enrich", "top5_q_enrich", "top10_q_enrich",
        "max_qq_mix", "max_qq_raw", "qq_gap_mix", "qq_gap_raw", "max_qq_sift",
        "log_clust_size",
        "clust_max_z60", "clust_mean_z60", "z60_minus_cmax",
        "clust_max_sim_inl", "clust_mean_q_enrich",
    ]
)

train_df = pd.read_csv("reid_student_data_main/train.csv")
test_df = pd.read_csv("reid_student_data_main/test.csv")
sub_v1 = pd.read_csv("submissions/submission_best.csv")
sub_v2 = pd.read_csv("submissions_v2/submission_v2_optuna_champion.csv")

print("=== AGREEMENT V2 vs V1 (0.82895) ===")
for sp in ["hyena", "leopard"]:
    m = (test_df["species"].values == sp)
    r1_1 = sub_v1["predictions"].str.split(" ").str[0].values[m]
    r1_2 = sub_v2["predictions"].str.split(" ").str[0].values[m]
    n1 = (r1_1 == "new_individual")
    n2 = (r1_2 == "new_individual")
    print(f"[{sp:7s}] Rank1 exact agree={(r1_1==r1_2).mean():.3f} | V1=new:{n1.mean():.3f} | V2=new:{n2.mean():.3f}")
    print(f"   V1=new & V2=known: {(n1 & (~n2)).sum()} | V1=known & V2=new: {((~n1) & n2).sum()} | Both known diff ID: {((~n1) & (~n2) & (r1_1!=r1_2)).sum()}")

print("\n=== CONNECTED COMPONENT SIZES & FEATURE SHIFTS (CV vs TEST) ===")
for sp in ["hyena", "leopard"]:
    with open(f"cache_v2/v2_bundles_{sp}.pkl", "rb") as f:
        d = pickle.load(f)
    vb = d["outer_data"][0]["val_bundle"]
    tb = d["test_bundle"]
    assert len(OPEN_NAMES) == tb["X_open_base"].shape[1], f"Len mismatch: {len(OPEN_NAMES)} vs {tb['X_open_base'].shape[1]}"
    for name, b in [("CV_F0", vb), ("TEST ", tb)]:
        _, cnts = np.unique(b["q_comp"], return_counts=True)
        cnts = np.sort(cnts)[::-1]
        print(f"[{sp:7s} {name}] n_q={len(b['q_comp'])} | top-10 cc sizes={cnts[:10].tolist()} | pct in cc>=2: {(cnts[cnts>=2].sum()/len(b['q_comp'])):.3f}")

    # Compare Subfold (training dist of Open-Set model!) vs Outer CV vs TEST!
    Xsf = np.vstack([sb["X_open_base"] for sb in d["full_seed_bundles"][2026]])
    Xcv = np.vstack([od["val_bundle"]["X_open_base"] for od in d["outer_data"]])
    Xte = tb["X_open_base"]
    mu_sf, sd_sf = Xsf.mean(0), Xsf.std(0) + 1e-6
    z_shift = (Xte.mean(0) - mu_sf) / sd_sf
    top_sh = np.argsort(-np.abs(z_shift))[:15]
    print(f"[{sp:7s}] Top-15 X_open_base shifts (TEST - FullSubfold_Train) / sd:")
    for idx in top_sh:
        print(f"   {idx:2d} {OPEN_NAMES[idx]:20s}: sub_mu={mu_sf[idx]:+.3f}, cv_mu={Xcv.mean(0)[idx]:+.3f}, te_mu={Xte.mean(0)[idx]:+.3f}, z_shift={z_shift[idx]:+.3f}")

    # Inspect rows where V1=new & V2=known on test.csv!
    m = (test_df["species"].values == sp)
    te_idx = np.where(m)[0]
    r1_1 = sub_v1["predictions"].str.split(" ").str[0].values[m]
    r1_2 = sub_v2["predictions"].str.split(" ").str[0].values[m]
    sw_kn = np.where((r1_1 == "new_individual") & (r1_2 != "new_individual"))[0]
    sw_nw = np.where((r1_1 != "new_individual") & (r1_2 == "new_individual"))[0]
    print(f"[{sp:7s}] Disagreement profile (V1=new->V2=known, n={len(sw_kn)}): med raw_s1={np.median(tb['raw_s1'][sw_kn]):.3f}, med sim_inl={np.median(tb['max_sim_inl'][sw_kn]):.1f}, pct sim>=6={(tb['max_sim_inl'][sw_kn]>=6).mean():.2f}")
    print(f"[{sp:7s}] Disagreement profile (V1=known->V2=new, n={len(sw_nw)}): med raw_s1={np.median(tb['raw_s1'][sw_nw]):.3f}, med sim_inl={np.median(tb['max_sim_inl'][sw_nw]):.1f}, pct sim>=6={(tb['max_sim_inl'][sw_nw]>=6).mean():.2f}")

print("\n=== DIAGNOSTIC ROWS ===")
for tid in ["te_00022", "te_00176", "te_00205", "te_00213", "te_00216"]:
    idx = test_df.index[test_df["id"] == tid][0]
    print(f"{tid} ({test_df.loc[idx, 'species']:7s}, vp={test_df.loc[idx, 'viewpoint']})")
    print(f"   V1: {sub_v1.loc[idx, 'predictions']}")
    print(f"   V2: {sub_v2.loc[idx, 'predictions']}")

import json
import lightgbm as lgb

with open("cache_v2/optuna_stage2_best_params.json") as f:
    hp_all = json.load(f)["stage2_hp"]

for sp in ["hyena", "leopard"]:
    with open(f"cache_v2/v2_bundles_{sp}.pkl", "rb") as f:
        d = pickle.load(f)
    Xsf = np.vstack([sb["X_open_base"] for sb in d["full_seed_bundles"][2026]])
    ysf = np.concatenate([sb["is_new"].astype(int) for sb in d["full_seed_bundles"][2026]])
    Xte = d["test_bundle"]["X_open_base"]
    mu_sf, sd_sf = Xsf.mean(0), Xsf.std(0) + 1e-6
    z_sh = (Xte.mean(0) - mu_sf) / sd_sf

    print(f"\n=== [{sp.upper()}] LOIO-CALIBRATED vs GRAPH SHIFTS ===")
    for fn in ["z45", "z60", "zraw", "zlda", "zkiss", "ec60", "ecraw", "eclda", "p60_z", "st1_z", "p60_g12", "raw_g12", "sift_sim_max", "sift_fund_max", "diff_q_enrich", "top5_q_enrich", "log_clust_size", "clust_max_z60", "qq_gap_mix"]:
        i = OPEN_NAMES.index(fn)
        print(f"  {fn:18s}: sub_mu={mu_sf[i]:+.3f}, te_mu={Xte.mean(0)[i]:+.3f}, z_shift={z_sh[i]:+.3f}")

    m = lgb.LGBMClassifier(n_jobs=4, random_state=42, verbose=-1, **hp_all[sp]["lgb_open"]).fit(Xsf, ysf)
    imp = m.booster_.feature_importance(importance_type="gain")
    topi = np.argsort(-imp)[:18]
    print(f"[{sp.upper()}] Top-18 Open-Set LGBM Gain Features (with test z_shift!):")
    for idx in topi:
        print(f"  {OPEN_NAMES[idx]:18s}: gain={imp[idx]:8.1f} | z_shift={z_sh[idx]:+.3f}")

    if sp == "leopard":
        tb = d["test_bundle"]
        ms = (test_df["species"].values == "leopard")
        te_ids = test_df["id"].values[ms]
        r1_1 = sub_v1["predictions"].str.split(" ").str[0].values[ms]
        r1_2 = sub_v2["predictions"].str.split(" ").str[0].values[ms]
        sw_nw = np.where((r1_1 != "new_individual") & (r1_2 == "new_individual"))[0]
        print("\n[LEOPARD] ALL 13 rows where V1=known -> V2=new:")
        for li in sw_nw:
            t2 = sub_v2.loc[tb["te_pos"][li], "predictions"].split()[:2]
            print(f"  {te_ids[li]} | V1={r1_1[li]:8s} | V2_top2={t2} | raw_s1={tb['raw_s1'][li]:.3f} | sim_inl={tb['max_sim_inl'][li]:3.0f} | z60={Xte[li, OPEN_NAMES.index('z60')]:+.2f} | diff_q={Xte[li, OPEN_NAMES.index('diff_q_enrich')]:+.2f} | vp_frac0={Xte[li, OPEN_NAMES.index('vp_frac0')]:.2f}")
        sw_kn = np.where((r1_1 == "new_individual") & (r1_2 != "new_individual"))[0]
        print("\n[LEOPARD] First 15 rows where V1=new -> V2=known:")
        for li in sw_kn[:15]:
            print(f"  {te_ids[li]} | V2={r1_2[li]:8s} | raw_s1={tb['raw_s1'][li]:.3f} | sim_inl={tb['max_sim_inl'][li]:3.0f} | z60={Xte[li, OPEN_NAMES.index('z60')]:+.2f} | diff_q={Xte[li, OPEN_NAMES.index('diff_q_enrich')]:+.2f} | vp_frac0={Xte[li, OPEN_NAMES.index('vp_frac0')]:.2f} | log_csz={Xte[li, OPEN_NAMES.index('log_clust_size')]:.2f}")
