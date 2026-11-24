import hashlib
import pickle
import numpy as np
import pandas as pd

test_df = pd.read_csv("reid_student_data_main/test.csv")
train_df = pd.read_csv("reid_student_data_main/train.csv")
sub_v1 = pd.read_csv("submissions/submission_best.csv")
sub_gr = pd.read_csv("submissions_v2/submission_v2_optuna_champion.csv")
sub_sr = pd.read_csv("submissions_v2/submission_v2_shift_robust_champion.csv")
sub_hb = pd.read_csv("submissions_v2/submission_v2_hybrid_grandmaster.csv")

def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

print("Backup submissions/submission_best.csv SHA256:", sha256("submissions/submission_best.csv"))

with open("cache_v2/v2_bundles_hyena.pkl", "rb") as f:
    dh = pickle.load(f)
tbh = dh["test_bundle"]
mh = (test_df["species"].values == "hyena")
r1_v1 = sub_v1["predictions"].str.split(" ").str[0].values[mh]
r1_gr = sub_gr["predictions"].str.split(" ").str[0].values[mh]
r1_sr = sub_sr["predictions"].str.split(" ").str[0].values[mh]

print(f"\n[HYENA] GR vs SR Rank1 agree: {(r1_gr == r1_sr).mean():.4f}")
for lbl, a, b in [
    ("GR=new  & SR=known", r1_gr == "new_individual", r1_sr != "new_individual"),
    ("GR=known & SR=new ", r1_gr != "new_individual", r1_sr == "new_individual"),
    ("V1=new  & SR=known", r1_v1 == "new_individual", r1_sr != "new_individual"),
    ("V1=known & SR=new ", r1_v1 != "new_individual", r1_sr == "new_individual"),
]:
    idx = np.where(a & b)[0]
    if len(idx) > 0:
        print(f"  {lbl} (n={len(idx):2d}): med raw_s1={np.median(tbh['raw_s1'][idx]):.3f}, med sim_inl={np.median(tbh['max_sim_inl'][idx]):.1f}, pct sim>=6={(tbh['max_sim_inl'][idx]>=6).mean():.2f}")

with open("cache_v2/v2_bundles_leopard.pkl", "rb") as f:
    dl = pickle.load(f)
tbl = dl["test_bundle"]
ml = (test_df["species"].values == "leopard")
r1_lv1 = sub_v1["predictions"].str.split(" ").str[0].values[ml]
r1_lsr = sub_sr["predictions"].str.split(" ").str[0].values[ml]
r1_lhb = sub_hb["predictions"].str.split(" ").str[0].values[ml]

print(f"\n[LEOPARD] V1 vs SR Rank1 agree: {(r1_lv1 == r1_lsr).mean():.4f}")
for lbl, a, b in [
    ("V1=new  & SR=known", r1_lv1 == "new_individual", r1_lsr != "new_individual"),
    ("V1=known & SR=new ", r1_lv1 != "new_individual", r1_lsr == "new_individual"),
    ("V1=new  & HB=known", r1_lv1 == "new_individual", r1_lhb != "new_individual"),
    ("V1=known & HB=new ", r1_lv1 != "new_individual", r1_lhb == "new_individual"),
]:
    idx = np.where(a & b)[0]
    if len(idx) > 0:
        print(f"  {lbl} (n={len(idx):2d}): med raw_s1={np.median(tbl['raw_s1'][idx]):.3f}, med sim_inl={np.median(tbl['max_sim_inl'][idx]):.1f}, pct sim>=6={(tbl['max_sim_inl'][idx]>=6).mean():.2f}")

# Build the optimal species-adaptive Grandmaster Champion:
#   Hyena:   50% V2_Graph_Enriched (0.91445 Nested CV, zero shift!) + 40% V2_Shift_Robust + 10% V1
#   Leopard: 85% V2_Shift_Robust (0.93548 Nested CV, zero shift, Top2=96.9%!) + 15% V1
rw = [1.0, 0.46, 0.28, 0.19, 0.14]
final_preds = []
for i in range(len(test_df)):
    sp = test_df["species"].iat[i]
    sc = {}
    if sp == "hyena":
        parts = [(0.50, sub_gr["predictions"].iat[i]), (0.40, sub_sr["predictions"].iat[i]), (0.10, sub_v1["predictions"].iat[i])]
    else:
        parts = [(0.85, sub_sr["predictions"].iat[i]), (0.15, sub_v1["predictions"].iat[i])]
    for w_m, rstr in parts:
        for r, tok in enumerate(rstr.split(" ")[:5]):
            sc[tok] = sc.get(tok, 0.0) + w_m * rw[r]
    final_preds.append(" ".join([k for k, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]]))

out_gm = "submissions_v2/submission_v2_final_grandmaster.csv"
pd.DataFrame({"id": test_df["id"].values, "predictions": final_preds}).to_csv(out_gm, index=False)
pd.DataFrame({"id": test_df["id"].values, "predictions": final_preds}).to_csv("submission.csv", index=False)

# Strict verification of final submission.csv
valid_by_sp = {
    sp: set(train_df[train_df["species"] == sp]["identity"].unique()) | {"new_individual"}
    for sp in ["hyena", "leopard"]
}
sdf = pd.read_csv("submission.csv")
assert len(sdf) == 2791 and list(sdf.columns) == ["id", "predictions"]
assert (sdf["id"].values == test_df["id"].values).all() and sdf["predictions"].isna().sum() == 0
for i, (ps, sp) in enumerate(zip(sdf["predictions"].values, test_df["species"].values)):
    toks = ps.strip().split(" ")
    assert len(toks) == 5 and len(set(toks)) == 5
    for t in toks:
        assert t in valid_by_sp[sp]

r1 = sdf["predictions"].str.split(" ").str[0].values
r2 = sdf["predictions"].str.split(" ").str[1].values
print("\n=== FINAL CHAMPION (submission.csv == submission_v2_final_grandmaster.csv) ===")
for sp in ["hyena", "leopard"]:
    m = (test_df["species"].values == sp)
    r1_base = sub_v1["predictions"].str.split(" ").str[0].values[m]
    print(f"  [{sp:7s}] R1=new:{(r1[m]=='new_individual').mean():.3f} | R2=new:{(r2[m]=='new_individual').mean():.3f} | Top2=new:{((r1[m]=='new_individual')|(r2[m]=='new_individual')).mean():.3f} | agree_V1={(r1[m]==r1_base).mean():.3f}")

print("Final submission.csv SHA256:", sha256("submission.csv"))
print("[ALL CHECKS PASSED 100%!]")
