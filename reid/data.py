"""Data loading. Paths follow D-004 (data in projectwild3/ at the repo root)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "projectwild3"
MAIN = DATA / "reid_student_data_main"
SIFT_DIRS = {"tr": DATA / "sift_features_train.npz", "te": DATA / "sift_features_test.npz"}
NEW = "new_individual"


@lru_cache(maxsize=1)
def load():
    """Return (train_df, test_df, X_train, X_test). X rows align with the dataframes.

    Missing viewpoints become the string "missing". Descriptors are raw (not normalised).
    """
    train = pd.read_csv(MAIN / "train.csv")
    test = pd.read_csv(MAIN / "test.csv")
    for d in (train, test):
        d["viewpoint"] = d["viewpoint"].fillna("missing")
    ids = pd.read_csv(MAIN / "miewid_ids.csv")["id"].values
    X = np.load(MAIN / "miewid_features.npy")
    row = {k: i for i, k in enumerate(ids)}
    Xtr = X[[row[i] for i in train["id"]]]
    Xte = X[[row[i] for i in test["id"]]]
    return train, test, Xtr, Xte


def load_sift(img_id: str):
    """Return (xy (n,2) float32, desc (n,128) uint8) for a train or test id.

    Handles both the extracted-directory layout and an original .npz file.
    """
    p = SIFT_DIRS[img_id[:2]]
    if p.is_dir():
        return np.load(p / f"{img_id}_xy.npy"), np.load(p / f"{img_id}_desc.npy")
    z = _npz(str(p))
    return z[f"{img_id}_xy"], z[f"{img_id}_desc"]


@lru_cache(maxsize=2)
def _npz(path):
    return np.load(path)


def l2n(A: np.ndarray) -> np.ndarray:
    return A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)


def preprocess(Xg: np.ndarray, *others: np.ndarray):
    """D-007 default: centre on the gallery mean, then L2-normalise every row."""
    mu = Xg.mean(0)
    return tuple(l2n(A - mu) for A in (Xg, *others))
