"""The learned adjustment set must be fitted inside the fold, not before it.

A representation trained on all the data and then handed to a cross-fitted estimator has
already seen every held-out fold. The estimate that comes back looks cross-fitted and is
not. These tests pin the contract that prevents it.
"""

import numpy as np
import pytest

from armd import causal
from tests.test_causal import simulate


def test_representer_is_called_once_per_fold_with_disjoint_rows():
    X, a, y, groups, _ = simulate(n=1200, n_per_patient=4, seed=7)
    seen = []

    def spy(tr, te):
        seen.append((set(tr.tolist()), set(te.tolist())))
        return X[tr], X[te]

    causal.crossfit_aipw(X, a, y, groups=groups, n_folds=5, seed=0, representer=spy)

    assert len(seen) == 5
    for tr, te in seen:
        assert not (tr & te), "a fold handed the representer its own held-out rows"
    assert set().union(*[te for _, te in seen]) == set(range(len(y)))


def test_representer_never_sees_a_held_out_patient():
    """Grouping is the whole point: leaking a sibling culture leaks the patient."""
    X, a, y, groups, _ = simulate(n=1200, n_per_patient=4, seed=7)

    def spy(tr, te):
        assert not (set(groups[tr]) & set(groups[te])), "patient leaked across the fold"
        return X[tr], X[te]

    causal.crossfit_aipw(X, a, y, groups=groups, n_folds=5, seed=0, representer=spy)


def test_a_representer_that_drops_rows_is_rejected():
    X, a, y, groups, _ = simulate(n=600, n_per_patient=3, seed=7)

    def bad(tr, te):
        return X[tr][:-1], X[te]

    with pytest.raises(ValueError, match="preserve row order and count"):
        causal.crossfit_aipw(X, a, y, groups=groups, n_folds=4, seed=0, representer=bad)


def test_the_representation_actually_reaches_the_nuisance_models():
    """An identity representer must reproduce the un-represented estimate exactly, and a
    destructive one must not -- otherwise the argument is being silently ignored."""
    X, a, y, groups, _ = simulate(n=1500, n_per_patient=3, seed=8)

    plain = causal.crossfit_aipw(X, a, y, groups=groups, n_folds=4, seed=1)
    same = causal.crossfit_aipw(X, a, y, groups=groups, n_folds=4, seed=1,
                                representer=lambda tr, te: (X[tr], X[te]))
    assert same.estimate == pytest.approx(plain.estimate, rel=1e-12)

    blanked = causal.crossfit_aipw(
        X, a, y, groups=groups, n_folds=4, seed=1,
        representer=lambda tr, te: (np.zeros((len(tr), 2)), np.zeros((len(te), 2))),
    )
    assert abs(blanked.estimate - plain.estimate) > 1e-6


def test_representer_factory_excludes_held_out_patients_from_training():
    """The factory's job is choosing training rows; check that directly.

    ``training.train`` is stubbed out so the test measures row selection rather than
    optimisation -- an untrained encoder still produces a well-shaped representation.
    """
    import torch
    from armd import pipeline, training as training_mod
    from armd.config import Config

    n, per = 240, 4
    n_pat = n // per
    anon = np.repeat(np.arange(n_pat), per)
    import pandas as pd
    df = pd.DataFrame({"anon_id": anon})
    Xbase = np.random.default_rng(0).normal(size=(n, 6)).astype(np.float32)
    Xsev_pre = np.zeros((n, 2), dtype=np.float32)
    tensors = {
        "base": torch.tensor(Xbase), "expo": torch.zeros((n, 0)),
        "sev": torch.zeros((n, 3)), "ys": torch.zeros((n, 2)),
        "ms": torch.zeros((n, 2)), "a": torch.zeros(n),
        "yc": torch.zeros(n), "mc": torch.zeros(n),
    }
    ctx = {"Xbase": Xbase, "Xsev_pre": Xsev_pre, "tensors": tensors, "n_agents": 2}
    cfg = Config(rep_dim=4, encoder_dims=(8,), val_fraction=0.2, seed=0)

    captured = {}
    original = training_mod.train

    def fake_train(model, tensors_, fit_rows, val_rows, cfg_, device, verbose=True):
        captured["fit"] = np.asarray(fit_rows)
        captured["val"] = np.asarray(val_rows)
        return None

    training_mod.train = fake_train
    pipeline.training.train = fake_train
    try:
        rows = np.arange(n)
        representer = pipeline._representer_factory(rows, anon, df, ctx, cfg, "cpu")
        te = np.arange(0, 40)
        tr = np.arange(40, n)
        Ztr, Zte = representer(tr, te)
    finally:
        training_mod.train = original
        pipeline.training.train = original

    used = np.concatenate([captured["fit"], captured["val"]])
    held_out_patients = set(anon[te].tolist())
    assert not (set(anon[used].tolist()) & held_out_patients), \
        "the encoder was trained on a patient from the held-out fold"
    assert Ztr.shape == (len(tr), cfg.rep_dim + Xsev_pre.shape[1])
    assert Zte.shape == (len(te), cfg.rep_dim + Xsev_pre.shape[1])
