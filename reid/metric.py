"""MAP@5 exactly as the competition defines it, and a submission checker."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from .data import NEW


def ap5(pred: Sequence[str], truth: str) -> float:
    for k, lab in enumerate(list(pred)[:5]):
        if lab == truth:
            return 1.0 / (k + 1)
    return 0.0


def map5(preds: Sequence[Sequence[str]], truths: Sequence[str]) -> float:
    assert len(preds) == len(truths)
    return sum(ap5(p, t) for p, t in zip(preds, truths)) / len(truths)


def check_submission(sub: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame) -> None:
    """Raise if the submission would be rejected or has obviously wrong labels."""
    assert list(sub.columns) == ["id", "predictions"], sub.columns
    assert len(sub) == len(test) and set(sub.id) == set(test.id) and sub.id.is_unique
    sp = dict(zip(test.id, test.species))
    valid = {s: set(train.identity[train.species == s]) | {NEW} for s in ("hyena", "leopard")}
    for i, p in zip(sub.id, sub.predictions):
        labs = p.split()
        assert 1 <= len(labs) <= 5 and len(set(labs)) == len(labs), (i, p)
        bad = [l for l in labs if l not in valid[sp[i]]]
        assert not bad, (i, bad)
