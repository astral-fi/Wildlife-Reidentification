"""b1 teammate pipeline (branch submissions/b1, LB 0.83484), unchanged except: paths from env
vars, and the run block moved under __main__ so it can be imported. Run: python -m models.b1_pipeline
(writes submissions/b1/*.csv; set B1_VALIDATE=1 for b1's own validation)."""
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

RUN_VALIDATION = os.environ.get("B1_VALIDATE") == "1"
ON_KAGGLE = Path("/kaggle/input").is_dir()
DATA_ROOT = Path(os.environ.get("B1_DATA", "projectwild3"))
OUTPUT = Path(os.environ.get("B1_OUT", "submissions/b1"))
OUTPUT.mkdir(parents=True, exist_ok=True)

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, fields
import hashlib
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace

import cv2
from lightgbm import LGBMRanker
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


NEW = "new_individual"
SPECIES = ("hyena", "leopard")
ANGLES = {
    "front": 0, "frontright": 1, "right": 2, "backright": 3,
    "back": 4, "backleft": 5, "left": 6, "frontleft": 7,
}
SHORTLIST = 14
SIFT_IDENTITIES = int(os.environ.get("B1_SIFT_IDS", 9))
SIFT_REFS = int(os.environ.get("B1_SIFT_REFS", 2))        # our change (D-016): more refs per id
TAIL_REFS = int(os.environ.get("B1_TAIL_REFS", 1))        # refs for shortlist positions >= 4
SIFT_TOP2 = os.environ.get("B1_SIFT_TOP2", "0") == "1"   # add sum of top-2 inliers feature


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def read_table(path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def locate_data(root, need_sift=True):
    root = Path(root)
    required = ("train.csv", "test.csv", "miewid_ids.csv", "miewid_features.npy",
                "sample_submission.csv")
    tables = [p for p in [root, root / "reid_student_data_main"]
              if all((p / name).is_file() for name in required)]
    if not tables:
        tables = [p.parent for p in sorted(root.rglob("miewid_ids.csv"))
                  if all((p.parent / name).is_file() for name in required)]
    if len(tables) != 1:
        raise ValueError(f"Expected one competition table directory under {root}; found {tables}")
    table = tables[0]
    archives = []
    filenames = ("sift_features_train.npz", "sift_features_test.npz") if need_sift else ()
    for filename in filenames:
        candidates = [table / filename, table.parent / filename, root / filename]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            matches = sorted(root.rglob(filename))
            if len(matches) != 1:
                raise ValueError(f"Expected one supplied {filename}; found {matches}")
            found = matches[0]
        archives.append(found)
    return table, archives


class Data:
    def __init__(self, root, need_sift=True):
        table, self.archives = locate_data(root, need_sift=need_sift)
        self.train = read_table(table / "train.csv")
        self.test = read_table(table / "test.csv")
        self.sample_ids = [row["id"] for row in read_table(table / "sample_submission.csv")]
        meta = self.train + self.test
        feature_ids = [row["id"] for row in read_table(table / "miewid_ids.csv")]
        mapping = {identifier: i for i, identifier in enumerate(feature_ids)}
        ids = [row["id"] for row in meta]
        raw = np.load(table / "miewid_features.npy", allow_pickle=False)
        if (len(mapping) != len(feature_ids) or len(set(ids)) != len(ids)
                or set(feature_ids) != set(ids)
                or raw.shape != (len(meta), 2152)
                or not np.isfinite(raw).all()):
            raise ValueError("Unexpected global descriptor shape, IDs, or values")
        if set(row["species"] for row in meta) != set(SPECIES) or any(
                not row["identity"] or row["identity"] == NEW for row in self.train):
            raise ValueError("Unexpected species or empty training identities")
        self.label_species = {row["identity"]: row["species"] for row in self.train}
        if any(self.label_species[row["identity"]] != row["species"] for row in self.train):
            raise ValueError("A training identity appears in both species")
        if (len(set(self.sample_ids)) != len(self.sample_ids)
                or set(self.sample_ids) != {row["id"] for row in self.test}):
            raise ValueError("Sample IDs do not match test IDs")
        positions = np.array([mapping[identifier] for identifier in ids])
        self.x = unit(np.ascontiguousarray(raw[positions], dtype=np.float32))
        self.id = np.asarray(ids)
        self.species = np.asarray([row["species"] for row in meta])
        self.view = np.asarray([row["viewpoint"] for row in meta])
        self.angle = np.asarray([ANGLES.get(v, -1) for v in self.view])
        self.label = np.asarray([row.get("identity", "") for row in meta])
        self.train_rows = np.arange(len(self.train))
        self.test_rows = np.arange(len(self.train), len(meta))
        self.duplicates = np.asarray([
            hashlib.blake2b(v, digest_size=12).hexdigest() for v in raw[positions[:len(self.train)]]
        ])


def split(data, rows, seed, novelty=0.17, known=0.20, cross_view=False):
    """Withhold complete identities for novelty and image groups for known matches."""
    rng = np.random.default_rng(seed)
    gallery, queries = [], []
    for species in SPECIES:
        species_rows = rows[data.species[rows] == species]
        labels = data.label[species_rows]
        identities, counts = np.unique(labels, return_counts=True)
        held = set()
        buckets = np.digitize(counts, [2, 4, 8, 16, 32])
        for bucket in np.unique(buckets):
            group = identities[buckets == bucket]
            n = min(len(group) - 1, round(novelty * len(group)))
            held.update(rng.permutation(group)[:n])
        by_identity = {
            identity: species_rows[labels == identity] for identity in identities
        }
        for identity, members in by_identity.items():
            if identity in held:
                queries.extend(members)
                continue
            if len(members) < 2 or (len(members) < 5 and rng.random() > 0.65):
                gallery.extend(members)
                continue
            chosen = np.array([], dtype=int)
            if cross_view:
                sides = data.view[members]
                side_choices = [side for side in ("left", "right")
                                if np.any(sides == side) and np.any(sides != side)
                                and (sides == side).mean() <= 0.70]
                if side_choices and rng.random() < 0.70:
                    chosen = members[sides == rng.choice(side_choices)]
            if len(chosen) == 0:
                n = min(len(members) - 1, max(1, round(known * len(members))))
                chosen = rng.permutation(members)[:n]
            # Keep exact duplicate descriptors on the same side of the split.
            mask = np.isin(data.duplicates[members], data.duplicates[chosen])
            if mask.all():
                mask[:] = False
            queries.extend(members[mask])
            gallery.extend(members[~mask])
    return np.asarray(sorted(gallery)), np.asarray(sorted(queries))


class SiftStore:
    def __init__(self, data, workers):
        self.ids = data.id
        self.n_train = len(data.train)
        self.archives = [p if p.is_dir() else np.load(p, allow_pickle=False)
                         for p in data.archives]
        self.images = OrderedDict()
        self.pair_cache = {}
        self.lock = Lock()
        self.workers = workers
        cv2.setNumThreads(1)

    def close(self):
        for archive in self.archives:
            if not isinstance(archive, Path):
                archive.close()

    def load(self, row):
        with self.lock:
            if row in self.images:
                self.images.move_to_end(row)
                return self.images[row]
        archive = self.archives[int(row >= self.n_train)]
        keys = [f"{self.ids[row]}_{suffix}" for suffix in ("xy", "desc")]
        if isinstance(archive, Path):
            xy, desc = [np.load(archive / f"{key}.npy", allow_pickle=False) for key in keys]
        else:
            xy, desc = [archive[key] for key in keys]
        if xy.shape != (len(desc), 2) or desc.ndim != 2 or desc.shape[1] != 128:
            raise ValueError(f"Unexpected SIFT arrays for {self.ids[row]}")
        desc = desc.astype(np.float32)
        norm = desc.sum(axis=1)
        valid = norm > 0
        desc = desc[valid] / norm[valid, None]
        # Hellinger distance on the provided descriptors, with no image processing.
        record = (xy[valid].astype(np.float32), np.sqrt(desc, out=desc))
        with self.lock:
            self.images[row] = record
            if len(self.images) > 1400:
                self.images.popitem(last=False)
        return record

    def compare(self, pair):
        xy1, desc1 = self.load(pair[0])
        xy2, desc2 = self.load(pair[1])
        out = np.zeros(6, dtype=np.float32)
        if min(len(desc1), len(desc2)) < 4:
            return out
        out[5] = 1
        # RootSIFT has unit L2 norm, so 2(1 - dot product) is its squared distance.
        similarity = desc1 @ desc2.T
        rows = np.arange(len(desc1))
        nearest = similarity.argmax(axis=1)
        reciprocal = similarity.argmax(axis=0)[nearest] == rows
        d1 = np.maximum(1 - similarity[rows, nearest], 0)
        similarity[rows, nearest] = -np.inf
        d2 = np.maximum(1 - similarity.max(axis=1), 0)
        keep = reciprocal & (d2 > 1e-7) & (d1 < 0.84 ** 2 * d2)
        if keep.sum() < 4:
            return out
        a, b = xy1[keep], xy2[nearest[keep]]
        ratios = np.sqrt(d1[keep] / d2[keep])
        out[0] = np.log1p(len(a))
        scale = max(float(np.ptp(b, axis=0).max()), 128)
        limit = float(np.clip(0.014 * scale, 3.0, 6.0))
        for fit in (cv2.estimateAffinePartial2D, cv2.estimateAffine2D):
            transform, mask = fit(a, b, method=cv2.RANSAC, ransacReprojThreshold=limit,
                                  maxIters=700, confidence=0.995, refineIters=5)
            if transform is None or not np.isfinite(transform).all():
                continue
            stretch = np.linalg.svd(transform[:, :2], compute_uv=False)
            if stretch[1] < 0.18 or stretch[0] > 5 or stretch[0] / stretch[1] > 3.0:
                continue
            inlier = mask.ravel().astype(bool)
            n = int(inlier.sum())
            if n < 4 or np.log1p(n) <= out[1] + 1e-6:
                continue
            extent_a = np.linalg.norm(np.ptp(a[inlier], axis=0))
            extent_b = np.linalg.norm(np.ptp(b[inlier], axis=0))
            out[1:5] = (
                np.log1p(n), n / len(a), 1 - float(ratios[inlier].mean()),
                min(extent_a, extent_b) / 512,
            )
        return out

    def features(self, queries, refs):
        q, candidates, count = refs.shape
        result = np.zeros((q, candidates, count, 6), dtype=np.float32)
        requests = {}
        for i, j, k in np.argwhere(refs >= 0):
            key = tuple(sorted((int(queries[i]), int(refs[i, j, k]))))
            requests.setdefault(key, []).append((i, j, k))
        missing = sorted(requests.keys() - self.pair_cache.keys())
        if missing:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                self.pair_cache.update(zip(missing, pool.map(self.compare, missing)))
        for key, positions in requests.items():
            for position in positions:
                result[position] = self.pair_cache[key]
        best = result[..., 1].argmax(axis=2)
        return np.concatenate([
            np.take_along_axis(result, best[:, :, None, None], axis=2).squeeze(2),
            result[..., 1].mean(axis=2, keepdims=True),
            result[..., 5].max(axis=2, keepdims=True),
        ] + ([np.sort(result[..., 1], axis=2)[..., -2:].sum(axis=2, keepdims=True)]
             if SIFT_TOP2 and result.shape[2] >= 2 else []), axis=2)


def strongest(matrix, labels, n_labels):
    # Gallery columns are ordered by label, so reduceat computes all class maxima.
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    if len(starts) != n_labels:
        raise ValueError("Noncontiguous or missing gallery identity")
    return np.maximum.reduceat(matrix, starts, axis=1)


def label_groups(labels):
    return np.split(np.arange(len(labels)), np.flatnonzero(np.diff(labels)) + 1)


def view_affinity(data, rows, cols):
    a, b = data.angle[rows, None], data.angle[cols][None, :]
    gap = np.minimum(abs(a - b), 8 - abs(a - b))
    valid = (a >= 0) & (b >= 0)
    close = np.where(valid, 1 - gap * 0.25,
                     (data.view[rows, None] == data.view[cols][None, :])
                     & (data.view[rows, None] != ""))
    return close.astype(np.float32)


def metric_embeddings(x, labels):
    """Regularized within-class whitening and a discriminative linear projection."""
    n = len(labels)
    size = min(144, n - 1, x.shape[1])
    pca = PCA(n_components=size, svd_solver="randomized", random_state=391)
    pca.fit(x[:n])
    projected = pca.transform(x)
    groups = label_groups(labels)
    centers = np.stack([projected[idx].mean(axis=0) for idx in groups])
    residual = projected[:n] - centers[labels]
    within = residual.T @ residual / n
    centered = centers - centers.mean(axis=0)
    between = centered.T @ centered / len(groups)
    within.flat[::size + 1] += 0.9 * np.trace(within) / size + 1e-5
    values, basis = eigh(between, within, check_finite=False)
    weights = np.maximum(values, 0) ** 0.35
    metric = (projected @ basis) * weights
    return unit(metric), unit(projected / np.maximum(pca.singular_values_, 1e-5) ** 0.35)


def top(matrix, n):
    n = min(n, matrix.shape[1])
    part = np.argpartition(matrix, -n, axis=1)[:, -n:]
    values = np.take_along_axis(matrix, part, axis=1)
    order = np.argsort(-values, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def neighborhood(similarity, k=18):
    cols, vals = [], []
    for offset in range(0, len(similarity), 128):
        scores = similarity[offset:offset + 128].copy()
        rows = np.arange(len(scores))
        scores[rows, rows + offset] = -np.inf
        chosen = top(scores, k)
        cols.append(chosen)
        vals.append(np.take_along_axis(scores, chosen, axis=1))
    cols = np.vstack(cols)
    vals = np.vstack(vals)
    weight = np.exp(np.clip((vals - 0.75) / 0.10, -15, 4)).astype(np.float32)
    directed = sparse.csr_matrix((weight.ravel(), (np.repeat(np.arange(len(cols)), cols.shape[1]),
                                                  cols.ravel())), shape=similarity.shape)
    symmetric = 0.65 * directed.minimum(directed.T) + 0.35 * directed.maximum(directed.T)
    return symmetric.tocsr(), cols, vals


def sparse_rowsum(matrix):
    total = np.asarray(matrix.sum(axis=1)).ravel()
    return matrix.multiply(1 / np.maximum(total, 1e-12)[:, None]).tocsr()


def pooled_features(scores, gallery_class, candidates, side, refs, priority):
    out = np.empty((*candidates.shape, 5), dtype=np.float32)
    groups = label_groups(gallery_class)
    for label, gallery_columns in enumerate(groups):
        qi, ci = np.where(candidates == label)
        if len(qi) == 0:
            continue
        sub = np.ix_(qi, gallery_columns)
        scores_for_label = scores[sub]
        best = top(scores_for_label, 3)
        values = np.take_along_axis(scores_for_label, best, axis=1)
        first = best[:, 0]
        out[qi, ci, 0] = values[:, 0]
        out[qi, ci, 1] = values.mean(axis=1)
        out[qi, ci, 2] = (values[:, 0] - values[:, -1])
        out[qi, ci, 3] = side[qi, gallery_columns[first]]
        out[qi, ci, 4] = np.log1p(len(gallery_columns))
        active = ci < SIFT_IDENTITIES
        picked = top(priority[sub][active], SIFT_REFS)
        refs[qi[active], ci[active], :picked.shape[1]] = gallery_columns[picked]
    # The tail of the shortlist gets one geometric check. For well-separated
    # global leaders, verify the nearest few alternatives rather than every tail.
    refs[:, 4:, TAIL_REFS:] = -1
    secure = (out[:, 0, 0] > 0.67) & (
        out[:, 0, 0] - np.max(out[:, 1:, 0], axis=1) > 0.18)
    refs[secure, 4:] = -1
    return out


@dataclass
class Episode:
    rows: np.ndarray
    labels: np.ndarray
    species: np.ndarray
    base_labels: np.ndarray
    base_scores: np.ndarray
    candidates: np.ndarray
    global_features: np.ndarray
    query_features: np.ndarray
    sift_features: np.ndarray | None
    gallery_count: np.ndarray
    gallery_has_view: np.ndarray

    @property
    def novel(self):
        return self.labels == NEW


def episode_for_species(data, gallery, query, store=None, transductive=True):
    classes, gal_label = np.unique(data.label[gallery], return_inverse=True)
    order = np.argsort(gal_label, kind="stable")
    gallery, gal_label = gallery[order], gal_label[order]
    n, nq = len(gallery), len(query)
    count = len(classes)
    rows = np.r_[gallery, query]
    x = data.x[rows]
    raw = x[n:] @ x[:n].T
    base_values = strongest(raw, gal_label, count)
    base = top(base_values, SHORTLIST)
    centered = unit(x - x[:n].mean(axis=0))
    by_view = centered.copy()
    for view in np.unique(data.view[gallery]):
        use = data.view[gallery] == view
        if view and use.sum() >= 15:
            center = centered[:n][use].mean(axis=0)
            by_view[data.view[rows] == view] -= 0.8 * use.sum() / (use.sum() + 75) * center
    by_view = unit(by_view)
    metric, pca = metric_embeddings(centered, gal_label)
    broad = 0.50 * centered + 0.25 * by_view + 0.25 * x
    # Concatenated weighted embeddings produce the average of component similarities.
    weights = np.sqrt(np.array([0.54, 0.20, 0.16, 0.10], dtype=np.float32))
    retrieval = np.concatenate([weight * vectors for weight, vectors
                                in zip(weights, (centered, by_view, x, pca))], axis=1)
    similarity = retrieval @ retrieval.T
    side = view_affinity(data, query, gallery)
    direct = similarity[n:, :n].copy()
    supervised = metric[n:] @ metric[:n].T
    graph, neighbor_cols, neighbor_vals = neighborhood(similarity)
    del similarity, retrieval
    membership = sparse.csr_matrix((np.ones(n, dtype=np.float32),
                                   (np.arange(n), gal_label)), shape=(n, count))
    if transductive:
        propagation = sparse_rowsum(graph[n:])
        q_to_g = propagation[:, :n]
        q_to_q = propagation[:, n:]
    else:
        q_to_g = sparse_rowsum(graph[n:, :n])
        q_to_q = sparse.csr_matrix((nq, nq), dtype=np.float32)
    direct_flow = (q_to_g @ membership).toarray()
    flow = direct_flow.copy()
    for _ in range(5):
        flow = direct_flow + 0.83 * (q_to_q @ flow)
    affinity = sparse_rowsum(graph)
    profiles = affinity[n:] @ affinity[:n].T
    shared = profiles.toarray()
    del profiles
    shared /= np.sqrt(
        np.maximum(np.asarray(affinity[n:].power(2).sum(axis=1)), 1e-9)
        * np.maximum(np.asarray(affinity[:n].power(2).sum(axis=1)).T, 1e-9)
    )
    pair = 0.86 * direct + 0.14 * supervised + 0.02 * side
    pair_class = strongest(pair, gal_label, count)
    shared_class = strongest(shared, gal_label, count)
    flow_cond = flow / np.maximum(flow.sum(axis=1, keepdims=True), 1e-8)
    combined = pair_class + 0.18 * np.sqrt(flow_cond) + 0.07 * shared_class
    # Admit diverse candidates from the learned metric, direct match and graph.
    shortlist = np.empty((nq, min(SHORTLIST, count)), dtype=np.int32)
    other = [top(pair_class, 5), top(flow_cond, 4),
             top(strongest(supervised, gal_label, count), 4), base[:, :4]]
    lead = top(combined, shortlist.shape[1])
    for i in range(nq):
        stream = [*lead[i, :8], *(label for ranks in other for label in ranks[i]), *lead[i, 8:]]
        shortlist[i] = list(dict.fromkeys(stream))[:shortlist.shape[1]]
    row_ids = np.arange(nq)[:, None]
    refs = np.full((*shortlist.shape, SIFT_REFS), -1, dtype=np.int32)
    pooled = pooled_features(pair, gal_label, shortlist, side, refs,
                             pair + 0.06 * side + 0.035 * shared)
    center = np.stack([broad[idx].mean(axis=0) for idx in label_groups(gal_label)])
    prototype = unit(broad[n:]) @ unit(center).T
    global_features = np.stack([
        base_values[row_ids, shortlist],
        pair_class[row_ids, shortlist],
        strongest(direct, gal_label, count)[row_ids, shortlist],
        strongest(supervised, gal_label, count)[row_ids, shortlist],
        strongest(pca[n:] @ pca[:n].T, gal_label, count)[row_ids, shortlist],
        shared_class[row_ids, shortlist],
        np.sqrt(flow_cond[row_ids, shortlist]),
        flow[row_ids, shortlist],
        prototype[row_ids, shortlist],
        combined[row_ids, shortlist],
    ], axis=2)
    global_features = np.concatenate([global_features, pooled], axis=2)
    global_features = np.concatenate([
        global_features, global_features[:, :, :10] - global_features[:, :, :10].max(
            axis=1, keepdims=True),
    ], axis=2)
    query_features = np.column_stack([
        neighbor_vals[n:, 0], neighbor_vals[n:, -1],
        (neighbor_cols[n:] >= n).mean(axis=1),
        flow.sum(axis=1), direct_flow.sum(axis=1),
        (data.view[query] == "").astype(float), (data.species[query] == "hyena").astype(float),
        pair_class.max(axis=1), base_values.max(axis=1),
        supervised.max(axis=1), shared_class.max(axis=1),
    ]).astype(np.float32)
    counts = np.bincount(gal_label)
    lookup = {label: counts[i] for i, label in enumerate(classes)}
    gallery_count = np.array([lookup.get(label, 0) for label in data.label[query]])
    gallery_views = set(zip(data.label[gallery], data.view[gallery]))
    gallery_has_view = np.array([(label, view) in gallery_views and bool(view)
                                 for label, view in zip(data.label[query], data.view[query])])
    truth = np.where(gallery_count > 0, data.label[query], NEW)
    if store is not None:
        refs = np.where(refs >= 0, gallery[refs], -1)
        sift_features = store.features(query, refs)
    else:
        sift_features = None
    return Episode(query, truth, data.species[query], classes[base],
                   base_values[row_ids, base], classes[shortlist],
                   global_features, query_features, sift_features, gallery_count,
                   gallery_has_view)

def stack(pieces):
    return Episode(**{
        name: (None if getattr(pieces[0], name) is None
               else np.concatenate([getattr(ep, name) for ep in pieces], axis=0))
        for name in Episode.__dataclass_fields__
    })


def candidate_inputs(ep, version):
    if version == 1:
        return ep.global_features
    sift = ep.sift_features
    relative = sift[:, :, :2] - sift[:, :, :2].max(axis=1, keepdims=True)
    return np.concatenate([ep.global_features, sift, relative], axis=2)


def classifier(kind, target="candidate"):
    if kind == "linear":
        c = 0.08 if target == "candidate" else 0.05
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500))
    return HistGradientBoostingClassifier(
        max_iter=105, learning_rate=0.085, max_leaf_nodes=15,
        min_samples_leaf=100 if target == "candidate" else 55,
        l2_regularization=6.0, early_stopping=False, random_state=392,
    )


def model_fit(kind, features, labels, weights, target):
    model = classifier(kind, target)
    if kind == "linear":
        model.fit(features, labels, logisticregression__sample_weight=weights)
    else:
        model.fit(features, labels, sample_weight=weights)
    return model


def log_odds(model, features):
    return logit(np.clip(model.predict_proba(features)[:, 1], 1e-5, 1 - 1e-5))


def fit_rankers(episode, version, mode):
    if version == 0 or mode == "distance":
        return []
    x = candidate_inputs(episode, version)
    y = episode.candidates == episode.labels[:, None]
    # The identity distribution is conditional on a catalogued individual.
    has_match = y.any(axis=1)
    train_x = x[has_match].reshape(-1, x.shape[2])
    train_y = y[has_match].ravel()
    weights = np.where(train_y, 6.0, 1.0)
    kinds = ["linear"] if mode == "linear" else ["linear", "tree"]
    return [model_fit(kind, train_x, train_y, weights, "candidate") for kind in kinds]


def rank(episode, version, models):
    if version == 0:
        return episode.base_labels, episode.base_scores
    if not models:
        return episode.candidates, episode.global_features[:, :, 9]
    x = candidate_inputs(episode, version)
    flat = x.reshape(-1, x.shape[2])
    scores = log_odds(models[0], flat)
    if len(models) == 2:
        scores = 0.45 * scores + 0.55 * log_odds(models[1], flat)
    return episode.candidates, scores.reshape(x.shape[:2])


def novelty_inputs(episode, version, scores):
    best = top(scores, 4)
    ordered = np.take_along_axis(scores, best, axis=1)
    first = ordered[:, 0]
    fourth = ordered[:, -1]
    baseline = episode.base_scores
    basic = np.column_stack([
        baseline[:, 0], baseline[:, 0] - baseline[:, 1], baseline[:, 0] - baseline[:, 3],
        (episode.species == "hyena").astype(float),
    ])
    if version == 0:
        return basic
    rows = np.arange(len(scores))
    selected = episode.global_features[rows, best[:, 0]]
    best_signals = episode.global_features[:, :, [1, 3, 5, 6, 7]].max(axis=1)
    features = [basic, episode.query_features, selected[:, [1, 3, 5, 7, 10, 11, 13, 14]],
                best_signals, np.column_stack([first, first - ordered[:, 1], first - fourth])]
    if version == 2:
        sift = episode.sift_features
        features.extend([sift[rows, best[:, 0]], sift[:, :, :5].max(axis=1)])
    return np.column_stack(features)


def fit_novelty(x, y, species, nonlinear):
    weights = np.ones(len(y))
    for name in SPECIES:
        group = species == name
        for value in (False, True):
            use = group & (y == value)
            weights[use] = 0.5 * group.sum() / max(use.sum(), 1)
    kinds = ["linear", "tree"] if nonlinear else ["linear"]
    return [model_fit(kind, x, y, weights, "novelty") for kind in kinds]


def temperature(scores, classes, labels):
    match = classes == labels[:, None]
    keep = match.any(axis=1)
    s, correct = scores[keep], match[keep].argmax(axis=1)
    scale = float(np.median(np.std(s, axis=1)))
    search = np.maximum(scale, 1e-5) * np.geomspace(0.06, 1.30, 25)
    rows = np.arange(len(correct))
    nll = [-np.log(np.maximum(softmax(s / t, axis=1)[rows, correct], 1e-12)).mean()
           for t in search]
    return float(search[np.argmin(nll)])


@dataclass
class Trained:
    version: int
    mode: str
    rankers: list
    novelty: list
    temperature: float


def fit_variant(episodes, version, mode="linear", nonlinear=False):
    scores, classes, features = [], [], []
    for i, episode in enumerate(episodes):
        neighbors = stack([other for j, other in enumerate(episodes) if i != j])
        rankers = fit_rankers(neighbors, version, mode)
        cls, value = rank(episode, version, rankers)
        scores.append(value)
        classes.append(cls)
        features.append(novelty_inputs(episode, version, value))
    data = stack(episodes)
    novelty = fit_novelty(np.concatenate(features), data.novel, data.species, nonlinear)
    temp = temperature(np.concatenate(scores), np.concatenate(classes), data.labels)
    return Trained(version, mode, fit_rankers(data, version, mode), novelty, temp)


def evidence(model, episode):
    labels, values = rank(episode, model.version, model.rankers)
    known = softmax(values / model.temperature, axis=1)
    values_for_novelty = novelty_inputs(episode, model.version, values)
    odds = log_odds(model.novelty[0], values_for_novelty)
    if len(model.novelty) == 2:
        odds = 0.50 * odds + 0.50 * log_odds(model.novelty[1], values_for_novelty)
    return labels, known, odds


def estimate_prior(odds, center=0.30, shrink=0.65):
    # The novelty models were fitted with a 50:50 prior within each species.
    prior = center
    for _ in range(30):
        update = float(expit(odds + logit(prior)).mean())
        if abs(update - prior) < 0.0001:
            break
        prior = float(np.clip(update, 0.08, 0.70))
    return float(np.clip(center + shrink * (prior - center), 0.12, 0.62))


def predictions(parts, species, prior=0.30, adapt=False):
    classes, known, odds = parts
    prior_by_species = {sp: estimate_prior(odds[species == sp], prior) if adapt else prior
                        for sp in SPECIES}
    novel = expit(odds + logit(np.array([prior_by_species[sp] for sp in species])))
    labels = np.column_stack([classes, np.full(len(classes), NEW)])
    probs = np.column_stack([known * (1 - novel[:, None]), novel])
    return np.take_along_axis(labels, top(probs, 5), axis=1), prior_by_species


def reciprocal_rank(labels, predictions):
    scores = (predictions == labels[:, None]) / np.arange(1, predictions.shape[1] + 1)
    return scores.max(axis=1)


def score_details(ep, pred, data):
    rr = reciprocal_rank(ep.labels, pred)
    slices = {
        "known": ~ep.novel, "unseen": ep.novel,
        "known_with_1_or_2_references": (ep.gallery_count > 0) & (ep.gallery_count <= 2),
        "known_without_matching_view": (~ep.novel & ~ep.gallery_has_view
                                         & (data.view[ep.rows] != "")),
        "known_with_matching_view": ~ep.novel & ep.gallery_has_view,
        "left": data.view[ep.rows] == "left", "right": data.view[ep.rows] == "right",
        "missing_view": data.view[ep.rows] == "",
        "hyena": ep.species == "hyena", "leopard": ep.species == "leopard",
    }
    details = {"rows": len(rr), "map5": float(rr.mean()),
               "top1": float((pred[:, 0] == ep.labels).mean()),
               "top5": float((rr > 0).mean())}
    for name, mask in slices.items():
        if mask.any():
            details[name] = {"rows": int(mask.sum()), "map5": float(rr[mask].mean())}
    return details, rr


def write_submission(data, ep, pred, path):
    by_row = dict(zip(data.id[ep.rows], pred))
    output_species = {row["id"]: row["species"] for row in data.test}
    if len(by_row) != len(ep.rows) or set(by_row) != set(data.sample_ids):
        raise ValueError("Missing or repeated output IDs")
    for image, labels in by_row.items():
        if not 1 <= len(labels) <= 5 or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid number of distinct guesses at {image}")
        if any(label != NEW and data.label_species.get(label) != output_species[image]
               for label in labels):
            raise ValueError(f"Out-of-species prediction at {image}")
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("id", "predictions"))
        writer.writerows((image, " ".join(by_row[image])) for image in data.sample_ids)
    temp.replace(path)


r = SimpleNamespace(**{k: v for k, v in globals().copy().items() if not k.startswith('_')})

def calibration_splits(data, gallery):
    return [r.split(data, gallery, seed, cross_view=(i == 1))
            for i, seed in enumerate((119, 222, 325))]


def query_edges(data, q, centered, store):
    n = len(q)
    if n < 2:
        return np.zeros((n, 1), int), np.zeros((n, 1), np.float32)
    sim = centered @ centered.T
    np.fill_diagonal(sim, -np.inf)
    neighbors = r.top(sim, min(4, n-1))
    cosine = np.take_along_axis(sim, neighbors, axis=1)
    # Verify query-to-query edges using the same supplied SIFT descriptors.
    geo = store.features(q, q[neighbors][:, :, None])
    geometry = expit(4.0 * (geo[:, :, 1] - np.log(9))) * np.clip(geo[:, :, 4]*6, 0, 1)
    support = np.maximum(expit((cosine - 0.68) / 0.048), geometry)
    return neighbors, support.astype(np.float32)


def kernel_features(data, gallery, query):
    labels, group = np.unique(data.label[gallery], return_inverse=True)
    x = r.unit(data.x[np.r_[gallery,query]] - data.x[gallery].mean(axis=0))
    n = len(gallery)
    g, q = x[:n], x[n:]
    gg, qg = g @ g.T, q @ g.T
    response = np.eye(len(labels), dtype=np.float32)[group]
    counts = response.sum(axis=0)
    # Fit two independent kernel ridge classifiers on the actual gallery.
    def ridge(k_train, k_query, penalty):
        k_train.flat[::n+1] += penalty
        return k_query @ cho_solve(cho_factor(k_train, lower=True, check_finite=False),
                                   response, check_finite=False)
    linear = ridge(gg.copy(), qg, 0.45)
    polynomial = ridge(0.45*gg + 0.55*gg**3, 0.45*qg + 0.55*qg**3, 0.18)
    # Compare a match against each reference's strongest other-animal matches.
    gg[group[:, None] == group[None, :]] = -2
    impostor_level = np.take_along_axis(gg, r.top(gg, 5), axis=1).mean(axis=1)
    corrected = qg - 0.42 * impostor_level[None, :]
    order = np.argsort(group, kind="stable")
    ordered = group[order]
    direct = r.strongest(qg[:, order], ordered, len(labels))
    local = r.strongest(corrected[:, order], ordered, len(labels))
    centroids = q @ r.unit((response.T @ g) / counts[:,None]).T
    # Pool several reference photos with a correction for catalogue size.
    density = np.log(np.maximum(np.exp(12*qg) @ response, 1e-15)) / 12
    density -= 0.5*np.log(counts)[None, :]/12
    raw = np.stack([direct, local, linear, polynomial, centroids, density],axis=2)
    # Include the top alternatives from both the supervised and retrieval experts.
    leaders = [r.top(raw[:,:,i], 12) for i in (3, 1, 0, 2, 5, 4)]
    width = min(24, len(labels))
    candidates = np.empty((len(q),width), dtype=int)
    for i in range(len(q)):
        sequence = np.stack([item[i] for item in leaders], axis=1).ravel()
        unique = list(dict.fromkeys(sequence))
        if len(unique) < width:
            unique = list(dict.fromkeys([*unique, *np.argsort(-polynomial[i])[:width]]))
        candidates[i] = unique[:width]
    features = raw[np.arange(len(q))[:,None], candidates]
    features = np.concatenate([features,
                               np.log1p(counts[candidates])[:,:,None]],axis=2)
    qs = []
    for i in range(raw.shape[2]):
        v = np.take_along_axis(raw[:,:,i], r.top(raw[:,:,i], 4), axis=1)
        qs.extend([v[:,0],v[:,0]-v[:,1],v[:,0]-v[:,3]])
    return labels[candidates], features, np.column_stack(qs), q


def prepare_one(data, gallery, query, store, directory, key):
    path = directory / (key + ".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if not (np.array_equal(saved["gallery_ids"], data.id[gallery]) and
                    np.array_equal(saved["query_ids"],data.id[query])):
                raise ValueError(f"Cache split mismatch for {key}")
        return
    print("Preparing", key, len(gallery), len(query), flush=True)
    base, labels, feats, query_features, links, reliability = [], [], [], [], [], []
    offset = 0
    for sp in r.SPECIES:
        g = gallery[data.species[gallery]==sp]
        q = query[data.species[query]==sp]
        base.append(r.episode_for_species(data,g,q,store))
        c, features, signals, centered = kernel_features(data,g,q)
        neighbors, weights = query_edges(data,q,centered,store)
        labels.append(c)
        feats.append(features)
        query_features.append(signals)
        links.append(neighbors+offset)
        reliability.append(weights)
        offset += len(q)
        print(" ",key,sp,"ready",flush=True)
    ep = r.stack(base)
    values = {f.name: getattr(ep,f.name) for f in fields(ep)}
    temp = path.with_suffix(".partial.npz")
    np.savez_compressed(temp, **values, kernel_labels=np.concatenate(labels),
                        kernel_x=np.concatenate(feats), kernel_q=np.concatenate(query_features),
                        neighbors=np.concatenate(links), weight=np.concatenate(reliability),
                        gallery_ids=data.id[gallery], query_ids=data.id[query])
    temp.replace(path)


def prepare(data, fold_map, store, directory, names):
    directory.mkdir(parents=True,exist_ok=True)
    for name in names:
        prepare_one(data,*fold_map[name],store,directory,name)


def read_episode(directory, key):
    with np.load(directory / (key + ".npz"), allow_pickle=False) as file:
        ep = r.Episode(**{f.name: file[f.name] for f in fields(r.Episode)})
        extras = {k:file[k] for k in ("kernel_labels","kernel_x","kernel_q","neighbors","weight")}
    return ep, extras


def kernel_novelty(ep, extra):
    idx = extra["neighbors"]
    kq = extra["kernel_q"]
    best = extra["kernel_labels"][:,0]
    weights = extra["weight"]
    agreement = (best[idx] == best[:,None]) * weights
    return np.column_stack([
        kq, ep.query_features, weights.max(axis=1), agreement.max(axis=1),
        (weights-agreement).max(axis=1), ep.sift_features[:,:,1].max(axis=1),
        kq[idx, 0].max(axis=1) - kq[:,0],
    ])


def fit_kernel(episodes):
    x, y, novel_x, novel_y, species = [], [], [], [], []
    for ep, extra in episodes:
        hits = extra["kernel_labels"] == ep.labels[:,None]
        keep = hits.any(axis=1)
        x.append(extra["kernel_x"][keep])
        y.append(hits[keep].argmax(axis=1))
        novel_x.append(kernel_novelty(ep,extra))
        novel_y.append(ep.novel)
        species.append(ep.species)
    x, y = np.concatenate(x).astype(np.float64), np.concatenate(y)
    scale = np.maximum(x.std(axis=(0,1)), 1e-5)
    x = (x - x.mean(axis=1,keepdims=True)) / scale
    row = np.arange(len(x))
    def objective(coef):
        logits = x @ coef
        loss = (logsumexp(logits,axis=1)-logits[row,y]).mean()
        residual = softmax(logits,axis=1)
        residual[row,y] -= 1
        grad = np.einsum("ij,ijk->k",residual,x)/len(x)
        return loss + 0.005*np.dot(coef,coef), grad + 0.01*coef
    solution = minimize(objective,np.zeros(x.shape[2]),jac=True,method="L-BFGS-B")
    if not solution.success:
        raise RuntimeError(solution.message)
    novelty = r.fit_novelty(np.concatenate(novel_x),np.concatenate(novel_y),
                            np.concatenate(species),True)
    return solution.x / scale, novelty


def dense_posterior(parts, species, vocabulary):
    labels, conditional, odds = parts
    pi = {s:r.estimate_prior(odds[species==s]) for s in r.SPECIES}
    p_new = expit(odds + logit(np.array([pi[s] for s in species])))
    p = np.zeros((len(labels),len(vocabulary)+1), dtype=np.float64)
    p[np.arange(len(labels))[:,None],np.searchsorted(vocabulary,labels)] = (
        conditional*(1-p_new[:,None]))
    p[:,-1] = p_new
    return p


def ranked(prob, vocabulary):
    labels = np.r_[vocabulary,np.array([r.NEW])]
    return labels[r.top(prob,5)]


def validate_inputs(directory,key,fold_map,data):
    episode = read_episode(directory,key)
    ep = episode[0]
    gallery, query = fold_map[key]
    if set(ep.rows) != set(query) or len(ep.rows)!=len(query) or np.intersect1d(gallery,query).size:
        raise ValueError(f"Invalid split for {key}")
    if key!="test" and np.intersect1d(data.duplicates[query],data.duplicates[gallery]).size:
        raise ValueError("Cross-split duplicate")
    return episode


def possible_labels(episode):
    ep, extra = episode
    return np.union1d(ep.candidates, extra["kernel_labels"])

def candidate_features(data, gallery, episode):
    known_views = set(zip(data.label[gallery], data.view[gallery]))
    views = data.view[episode.rows]
    same_view = np.array([[bool(view) and (label, view) in known_views for label in candidates]
                          for view, candidates in zip(views, episode.candidates)], dtype=np.float32)
    x = r.candidate_inputs(episode, 2)
    return np.concatenate([x, same_view[:, :, None]], axis=2)


def fit_svm_rankers(folds):
    all_x, all_y, all_weights, pairs, pair_weights = [], [], [], [], []
    for ep, features in folds:
        correct = ep.labels[:, None] == ep.candidates
        usable = correct.any(axis=1)
        indices = np.flatnonzero(usable)
        x, y = features[usable], correct[usable]
        weights = np.where(ep.gallery_has_view[indices] | (x[:, :, -1].sum(axis=1) == 0),
                           1.0, 1.35)
        # Each correct candidate competes against three hard negatives and one
        # alternative from deeper in the shortlist.
        scores = np.where(y, -np.inf, x[:, :, 9])
        hard = r.top(scores, 3)
        deep = np.where(y, -np.inf, x[:, :, 5])
        np.put_along_axis(deep, hard, -np.inf, axis=1)
        negatives = np.column_stack([hard, deep.argmax(axis=1)])
        rows = np.arange(len(x))
        right = x[rows, y.argmax(axis=1)]
        wrong = x[rows[:, None], negatives]
        diffs = (right[:, None] - wrong).reshape(-1, x.shape[2])
        pairs.append(diffs)
        pair_weights.append(np.repeat(weights, negatives.shape[1]))
        all_x.append(np.concatenate([right, wrong.reshape(-1, x.shape[2])]))
        all_y.append(np.repeat([True, False], [len(x), len(diffs)]))
        all_weights.append(np.concatenate([2*weights, np.repeat(weights, negatives.shape[1])]))
    diff = np.concatenate(pairs)
    sample_weight = np.tile(np.concatenate(pair_weights), 2)
    linear = make_pipeline(StandardScaler(with_mean=False),
                           LinearSVC(C=0.003, fit_intercept=False, dual=False, max_iter=4000))
    linear.fit(np.concatenate([diff, -diff]), np.repeat([True, False], len(diff)),
               linearsvc__sample_weight=sample_weight)
    rbf = make_pipeline(StandardScaler(), SVC(C=1.2, gamma="scale", cache_size=300))
    rbf.fit(np.concatenate(all_x), np.concatenate(all_y),
            svc__sample_weight=np.concatenate(all_weights))
    return linear, rbf


def margins(model, x):
    return model.decision_function(x.reshape(-1, x.shape[2])).reshape(x.shape[:2])


def novelty_weights(y, species):
    weights = np.empty(len(y))
    for sp in r.SPECIES:
        group = species == sp
        for label in (False, True):
            mask = group & (y == label)
            weights[mask] = group.sum()/(2*mask.sum())
    return weights


def novelty_svm(sets):
    model = make_pipeline(StandardScaler(), SVC(C=0.8, gamma="scale", cache_size=220))
    x, y, sp = (np.concatenate(c) for c in zip(*sets))
    weights = novelty_weights(y, sp)
    return model.fit(x, y, svc__sample_weight=weights)


def novelty_data(ep, x, rbf_values):
    best = rbf_values.argmax(axis=1)
    views = x[:, :, -1]
    return np.column_stack([r.novelty_inputs(ep, 2, rbf_values),
                            views[np.arange(len(views)), best], views.mean(axis=1)])


def train_models(folds):
    episodes = [ep for ep, _ in folds]
    print("Fitting SIFT-assisted ranker and SVMs across", len(episodes), "calibration episodes", flush=True)
    baseline = r.fit_variant(episodes, 2, "blend", True)
    scores = [[], []]
    novelty_sets = []
    for i, (ep, x) in enumerate(folds):
        rivals = fit_svm_rankers([fold for j, fold in enumerate(folds) if i != j])
        values = [margins(model, x) for model in rivals]
        for recorded, score in zip(scores, values):
            recorded.append(score)
        novelty_sets.append((novelty_data(ep, x, values[1]), ep.novel, ep.species))
    labels = np.concatenate([ep.labels for ep in episodes])
    candidates = np.concatenate([ep.candidates for ep in episodes])
    temps = [r.temperature(np.concatenate(v), candidates, labels) for v in scores]
    decisions = []
    for i, (x, _, _) in enumerate(novelty_sets):
        svm = novelty_svm([fold for j, fold in enumerate(novelty_sets) if i != j])
        decisions.append(svm.decision_function(x))
    y = np.concatenate([ep.novel for ep in episodes])
    species = np.concatenate([ep.species for ep in episodes])
    platt = LogisticRegression(C=2).fit(np.concatenate(decisions)[:, None], y,
                                      sample_weight=novelty_weights(y, species))
    return baseline, fit_svm_rankers(folds), temps, novelty_svm(novelty_sets), platt


def svm_branch_evidence(fitted, fold):
    ep, x = fold
    baseline, rankers, temperatures, novelty, platt = fitted
    labels, base_conditional, base_odds = r.evidence(baseline, ep)
    if not np.array_equal(labels, ep.candidates):
        raise ValueError("Candidate ordering differs across rankers")
    linear_margin, rbf_margin = [margins(model, x) for model in rankers]
    linear_conditional = softmax(linear_margin / temperatures[0], axis=1)
    svm_margin = novelty.decision_function(novelty_data(ep, x, rbf_margin))
    svm_odds = platt.decision_function(svm_margin[:, None])
    return (
        labels,
        0.70 * base_conditional + 0.30 * linear_conditional,
        0.65 * base_odds + 0.35 * svm_odds,
    )


def fit_version_two(items, keys, data, folds):
    episodes = [item[0] for item in items]
    sift_episodes = [ep for ep, _ in episodes]
    svm_items = [(ep, candidate_features(data, folds[key][0], ep))
                 for (ep, _), key in zip(episodes, keys)]
    return train_models(svm_items), r.fit_variant(sift_episodes, 2, "blend", True), fit_kernel(episodes)


def version_two_probabilities(models, item, data, folds, key, vocabulary):
    svm_model, sift_model, kernel_model = models
    ep, extra = item[0]
    x = candidate_features(data, folds[key][0], ep)
    svm_result = dense_posterior(svm_branch_evidence(svm_model, (ep, x)), ep.species, vocabulary)
    sift_result = dense_posterior(r.evidence(sift_model, ep), ep.species, vocabulary)
    coef, novelty = kernel_model
    z = kernel_novelty(ep, extra)
    odds = 0.65 * r.log_odds(novelty[0], z) + 0.35 * r.log_odds(novelty[1], z)
    kernel = extra["kernel_labels"], softmax(extra["kernel_x"] @ coef, axis=1), odds
    kernel_result = dense_posterior(kernel, ep.species, vocabulary)
    return 0.50 * svm_result + 0.50 * (0.55 * sift_result + 0.45 * kernel_result)

def light_model(iterations):
    return LGBMRanker(n_estimators=iterations, learning_rate=0.055, num_leaves=15,
                      min_child_samples=64, reg_lambda=9.0, colsample_bytree=0.85,
                      verbosity=-1, n_jobs=2, random_state=885,
                      lambdarank_truncation_level=6, label_gain=[0, 1], objective="lambdarank")


def boost_fold(data, folds, cache, key):
    cached = validate_inputs(cache, key, folds, data)
    ep, extra = cached
    x = candidate_features(data, folds[key][0], ep)
    matches = ep.candidates[:, :, None] == extra["kernel_labels"][:, None, :]
    ri = np.arange(len(ep.rows))[:, None]
    k = extra["kernel_x"][ri, matches.argmax(axis=2)]
    k = np.where(matches.any(axis=2)[:, :, None], k, np.nan)
    relative = k - np.max(extra["kernel_x"], axis=1, keepdims=True)
    x = np.concatenate([x, k, relative], axis=2)
    # Per-query evidence helps the rankers interpret geometric and global scores.
    geometry = ep.sift_features[:, :, [1, 2, 4]].max(axis=1)
    whole = np.column_stack([ep.query_features, geometry, extra["kernel_q"]])
    broadcast = np.broadcast_to(whole[:, None], (*x.shape[:2], whole.shape[1]))
    candidate_x = np.concatenate([x, broadcast], axis=2).astype(np.float32)
    return cached, candidate_x, whole


def histogram():
    return HistGradientBoostingClassifier(max_iter=115, learning_rate=0.07,
                                          max_leaf_nodes=15, min_samples_leaf=50,
                                          l2_regularization=7, early_stopping=False,
                                          random_state=885)


def light_predict(model, x):
    return model.booster_.predict(x, raw_score=True, num_threads=2)


def target(ep):
    return ep.candidates == ep.labels[:, None]


def joint_features(item):
    (ep, _), x, whole = item
    blank = np.full((len(x), 1, x.shape[2]), np.nan, dtype=np.float32)
    blank[:, 0, -whole.shape[1]:] = whole
    expanded = np.concatenate([x, blank], axis=1)
    # The final candidate is the unseen label. Its own features summarize
    # the evidence for the entire query rather than any reference identity.
    flag = np.zeros((*expanded.shape[:2], 1), dtype=np.float32)
    flag[:, -1, :] = 1
    return np.concatenate([expanded, flag], axis=2)


def fit_rank_models(items):
    x = np.concatenate([a[1] for a in items])
    y = np.concatenate([target(a[0][0]) for a in items])
    valid = y.any(axis=1)
    known_x = x[valid].reshape(-1, x.shape[2])
    hit = y[valid].ravel()
    rank = light_model( 145).fit(known_x, hit.astype(int),
                                    group=np.full(valid.sum(), x.shape[1]))
    hgb = histogram().fit(known_x, hit, sample_weight=np.where(hit, 6.0, 1.0))
    joint_x = np.concatenate([joint_features(a) for a in items])
    novel = np.concatenate([a[0][0].novel for a in items])
    good = valid | novel
    joint_y = np.column_stack([y, novel])[good]
    joint = light_model( 155).fit(joint_x[good].reshape(-1, joint_x.shape[2]),
                                      joint_y.ravel().astype(int),
                                      group=np.full(good.sum(), joint_x.shape[1]))
    return rank, hgb, joint


def raw_scores(models, item):
    _, x, _ = item
    rank, hgb, joint = models
    base = x.reshape(-1, x.shape[2])
    j = joint_features(item)
    return (light_predict(rank, base).reshape(x.shape[:2]),
            hgb.decision_function(base).reshape(x.shape[:2]),
            light_predict(joint, j.reshape(-1, j.shape[2])).reshape(j.shape[:2]))


def novelty_vars(item, rank_scores):
    (ep, _), x, whole = item
    direct_parts = [whole]
    for score in rank_scores[:2]:
        direct_parts.append(r.novelty_inputs(ep, 2, score))
    joint = rank_scores[2]
    direct_parts.extend([joint[:, -1], joint[:, -1] - joint[:, :-1].max(axis=1)])
    return np.column_stack(direct_parts)


def fit_version_three(items):
    held_out_scores = [
        raw_scores(fit_rank_models([other for j, other in enumerate(items) if j != i]), item)
        for i, item in enumerate(items)
    ]
    episodes = [item[0][0] for item in items]
    x = np.concatenate([
        novelty_vars(item, scores) for item, scores in zip(items, held_out_scores)
    ])
    novel = np.concatenate([ep.novel for ep in episodes])
    species = np.concatenate([ep.species for ep in episodes])
    calibration = histogram().fit(x, novel, sample_weight=novelty_weights(novel, species))
    labels = np.concatenate([ep.labels for ep in episodes])
    candidates = np.concatenate([ep.candidates for ep in episodes])
    temperature = r.temperature(np.concatenate([row[0] for row in held_out_scores]),
                                candidates, labels)
    return fit_rank_models(items), calibration, temperature


def version_three_probabilities(model, reference, item, data, gallery, vocabulary):
    ep = item[0][0]
    models, calibration, temperature = model
    scores = raw_scores(models, item)
    known = softmax(scores[0] / temperature, axis=1)
    novelty = calibration.decision_function(novelty_vars(item, scores))
    proposal = dense_posterior((ep.candidates, known, novelty), ep.species, vocabulary)

    conditional = reference[:, :-1] / (1 - reference[:, -1, None]).clip(1e-12)
    top_known = vocabulary[conditional.argmax(axis=1)]
    gallery_views = set(zip(data.label[gallery], data.view[gallery]))
    supported = np.array([
        bool(view) and (label, view) in gallery_views
        for label, view in zip(top_known, data.view[ep.rows])
    ])
    mixing = 0.35 * np.where(supported, 1.0, 0.25)[:, None]
    probabilities = (1 - mixing) * reference + mixing * proposal
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Each test sighting must have unit total probability")
    return probabilities

FILENAMES = (
    "submission_v1_reference.csv",
    "submission_svm_kernel.csv",
    "submission_boosting_follow.csv",
)
SCORED_SHA256 = "698f5006da466ff8ade5766af453973556993b84bf6742aff56bbe08b8b38999"


def fit_and_predict_versions(data, gallery, queries, store, cache):
    training = [f"full_inner_{i}" for i in range(3)]
    folds = dict(zip(training, calibration_splits(data, gallery)))
    folds["test"] = (gallery, queries)
    prepare(data, folds, store, cache, [*training, "test"])
    items = [boost_fold(data, folds, cache, name) for name in training]
    query_item = boost_fold(data, folds, cache, "test")
    ep = query_item[0][0]
    vocabulary = possible_labels(query_item[0])

    model_one = r.fit_variant([item[0][0] for item in items], 0, "distance", False)
    p1, _ = r.predictions(r.evidence(model_one, ep), ep.species, adapt=True)
    model_two = fit_version_two(items, training, data, folds)
    p2 = version_two_probabilities(model_two, query_item, data, folds, "test", vocabulary)
    model_three = fit_version_three(items)
    p3 = version_three_probabilities(model_three, p2, query_item, data, gallery, vocabulary)
    return ep, (p1, ranked(p2, vocabulary), ranked(p3, vocabulary))



if __name__ == "__main__":
    data = r.Data(DATA_ROOT)
    store = r.SiftStore(data, 6)
    try:
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            if RUN_VALIDATION:
                shared_train, shared_val = r.split(data, data.train_rows, 2065, novelty=0.075, known=0.16)
                stress_train, stress_val = r.split(data, shared_train, 4001, cross_view=True)
                assert (len(shared_train), len(shared_val), len(stress_val)) == (5707, 1456, 2210)
                for name, gallery, queries in (
                    ("Shared", shared_train, shared_val), ("Viewpoint stress", stress_train, stress_val)
                ):
                    ep, results = fit_and_predict_versions(data, gallery, queries, store, root / name)
                    print(name, {f"V{i}": r.score_details(ep, predictions, data)[0]
                                 for i, predictions in enumerate(results, 1)})
            ep, results = fit_and_predict_versions(data, data.train_rows, data.test_rows, store, root)
            for filename, predictions in zip(FILENAMES, results):
                r.write_submission(data, ep, predictions, OUTPUT / filename)
                print(f"Wrote {filename} with {len(ep.rows):,} predictions")
    finally:
        store.close()
    
    submitted = OUTPUT / FILENAMES[2]
    checksum = hashlib.sha256(submitted.read_bytes()).hexdigest()
    print("Version 3 CSV", submitted)
    print("SHA-256", checksum)
    print("Matches the scored file", checksum == SCORED_SHA256)