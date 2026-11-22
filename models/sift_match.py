"""SIFT pair verification on the PROVIDED keypoints only (no images are read).

For a pair (a, b) of image ids, compute several match statistics at once:
  n_a, n_b          keypoint counts
  ratio             RootSIFT ratio-test (0.8) matches, mutual-NN filtered
  ratio7            same with ratio 0.7
  aff               inliers of cv2.estimateAffinePartial2D (RANSAC, 8 px) on `ratio` matches
  fund              inliers of cv2.findFundamentalMat (USAC_MAGSAC, 3 px)
  homo              inliers of cv2.findHomography (RANSAC, 20 px)
Results are cached in cache/sift_pairs.pkl (dict (a,b) -> tuple).
"""
from __future__ import annotations

import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from reid.data import ROOT, load_sift

CACHE = ROOT / "cache" / "sift_pairs.pkl"
FIELDS = ("n_a", "n_b", "ratio", "ratio7", "aff", "fund", "homo")
MAXKP = 1500
_feat = {}


def rootsift(img_id):
    if img_id in _feat:
        return _feat[img_id]
    xy, d = load_sift(img_id)
    d = d.astype(np.float32)
    if len(d) > MAXKP:  # keep a deterministic subset
        idx = np.linspace(0, len(d) - 1, MAXKP).astype(int)
        xy, d = xy[idx], d[idx]
    d /= np.clip(d.sum(1, keepdims=True), 1e-7, None)
    d = np.sqrt(d)
    if len(_feat) > 400:
        _feat.clear()
    _feat[img_id] = (xy.astype(np.float32), d)
    return _feat[img_id]


def _inl(mask):
    return 0 if mask is None else int(mask.sum())


def pair_stats(pair):
    a, b = pair
    xa, da = rootsift(a)
    xb, db = rootsift(b)
    na, nb = len(da), len(db)
    if na < 2 or nb < 2:
        return pair, (na, nb, 0, 0, 0, 0, 0)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m_ab = bf.knnMatch(da, db, k=2)
    m_ba = bf.match(db, da)
    back = np.full(nb, -1)
    for m in m_ba:
        back[m.queryIdx] = m.trainIdx
    sel, sel7 = [], []
    for mm in m_ab:
        if len(mm) < 2:
            continue
        m, n = mm
        if back[m.trainIdx] != m.queryIdx:
            continue
        r = m.distance / max(n.distance, 1e-7)
        if r < 0.8:
            sel.append((m.queryIdx, m.trainIdx))
            if r < 0.7:
                sel7.append(1)
    k = len(sel)
    aff = fund = homo = 0
    if k >= 3:
        s = np.array(sel)
        pa, pb = xa[s[:, 0]], xb[s[:, 1]]
        _, mk = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=8.0,
                                            maxIters=2000)
        aff = _inl(mk)
        if k >= 8:
            try:
                _, mk = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 3.0, 0.999, 2000)
                fund = _inl(mk)
            except cv2.error:
                pass
        if k >= 4:
            _, mk = cv2.findHomography(pa, pb, cv2.RANSAC, 20.0)
            homo = _inl(mk)
    return pair, (na, nb, k, len(sel7), aff, fund, homo)


def load_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def compute(pairs, workers=6):
    """Return dict pair -> stats for all pairs, using / updating the disk cache."""
    cache = load_cache()
    todo = sorted({p for p in pairs if p not in cache})
    if todo:
        cv2.setNumThreads(1)
        print(f"matching {len(todo)} new pairs", flush=True)
        # sort by first id so the per-process feature cache hits
        with Pool(workers) as pool:
            for i, (p, s) in enumerate(pool.imap_unordered(pair_stats, todo, chunksize=64)):
                cache[p] = s
        CACHE.parent.mkdir(exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)
    return {p: cache[p] for p in pairs}
