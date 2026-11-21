import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components

ART_DIR = r"C:\Users\ankit\.gemini\antigravity\brain\1b35347c-d2d2-4f03-9c8a-66a4dcb52159"
os.makedirs(ART_DIR, exist_ok=True)

train_df = pd.read_csv("reid_student_data_main/train.csv")
test_df = pd.read_csv("reid_student_data_main/test.csv")
miew = np.load("reid_student_data_main/miewid_features.npy")
X_norm = normalize(miew, axis=1)
X_tr, X_te = X_norm[:len(train_df)], X_norm[len(train_df):]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=150)

# Panel 1: Long-tail class frequency distribution per species
ax = axes[0, 0]
for sp, color in [("hyena", "#d95f02"), ("leopard", "#1b9e77")]:
    vc = train_df[train_df["species"] == sp]["identity"].value_counts().values
    ax.plot(np.arange(1, len(vc) + 1), vc, lw=2.2, color=color, label=f"{sp.capitalize()} ({len(vc)} IDs, max={vc.max()})")
ax.set_yscale("log")
ax.set_xlabel("Individual Rank (sorted by sightings)")
ax.set_ylabel("Sightings in Train (log scale)")
ax.set_title("A. Heavy Long-Tail Class Imbalance Across 550 Individuals")
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: Viewpoint distribution shift (Train vs Test)
ax = axes[0, 1]
vps_main = ["left", "right", "front", "back", "other/missing"]
def bucket_vp(s):
    s = s.fillna("MISSING")
    return np.where(s.isin(["left", "right", "front", "back"]), s, "other/missing")

tr_b = pd.Series(bucket_vp(train_df["viewpoint"])).value_counts(normalize=True).reindex(vps_main).fillna(0) * 100
te_b = pd.Series(bucket_vp(test_df["viewpoint"])).value_counts(normalize=True).reindex(vps_main).fillna(0) * 100
x = np.arange(len(vps_main))
ax.bar(x - 0.18, tr_b.values, width=0.35, color="#4c72b0", label=f"Train (n={len(train_df)})")
ax.bar(x + 0.18, te_b.values, width=0.35, color="#dd8452", label=f"Test (n={len(test_df)})")
ax.set_xticks(x)
ax.set_xticklabels(vps_main)
ax.set_ylabel("Percentage of Sightings (%)")
ax.set_title("B. Viewpoint Distribution (Train vs Test)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()

# Panel 3 & 4: Open-Set LOBO Known vs Unseen vs Test Max-Gallery Similarity for Hyena & Leopard
for idx_sp, (sp, ax_s) in enumerate([("hyena", axes[1, 0]), ("leopard", axes[1, 1])]):
    tr_i = np.where(train_df["species"].values == sp)[0]
    te_i = np.where(test_df["species"].values == sp)[0]
    Xt, Xq = X_tr[tr_i], X_te[te_i]
    ids_t = train_df["identity"].values[tr_i]
    S_tt = Xt @ Xt.T
    np.fill_diagonal(S_tt, -2.0)
    same_id = (ids_t[:, None] == ids_t[None, :])
    n_c, comp = connected_components((S_tt >= 0.85) & same_id, directed=False)
    id_b = pd.DataFrame({"id": ids_t, "b": comp}).groupby("id")["b"].nunique()

    lobo_k, loio_n = [], []
    for i in range(len(tr_i)):
        loio_n.append(S_tt[i, ids_t != ids_t[i]].max())
        if id_b[ids_t[i]] >= 2:
            lobo_k.append(S_tt[i, comp != comp[i]].max())
    test_m = (Xq @ Xt.T).max(axis=1)

    bins = np.linspace(0.28, 0.96, 30)
    ax_s.hist(loio_n, bins=bins, density=True, alpha=0.45, color="#e41a1c", label="Simulated `new_individual` (LOIO)")
    ax_s.hist(lobo_k, bins=bins, density=True, alpha=0.45, color="#377eb8", label="Simulated Known ID (Encounter-Split)")
    ax_s.hist(test_m, bins=bins, density=True, histtype="step", lw=2.4, color="#111111", label="Actual Test Sightings")
    pi_txt = "39.5%" if sp == "hyena" else "60.5%"
    ax_s.set_title(f"{'C' if idx_sp==0 else 'D'}. {sp.capitalize()} Open-Set Max-Sim Mixture (Est. new_individual ≈ {pi_txt})")
    ax_s.set_xlabel("Max Cosine Similarity to Training Gallery")
    ax_s.set_ylabel("Density")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(ART_DIR, "eda_open_set_diagnostics.png"), dpi=150)
plt.close()
print("Saved eda_open_set_diagnostics.png")
