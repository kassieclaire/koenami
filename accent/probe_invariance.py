"""Can the accent space be made gender-invariant without a gender label at inference?

Koenami's users are changing how their voice reads for gender, so an accent reading must not move
when they do. probe.py centred each speaker by their own gender label, which a user's take cannot
have. Here the gender-predictive directions are removed from every vector instead (iterative
nullspace projection, Ravfogel et al. 2020), fitted on the reference speakers only; the same
projection then applies to any take. Reports region accuracy of a shrinkage-LDA accent subspace
(held-out speakers) and how much gender is still recoverable, before and after.

Run (fwuff, after probe.py has cached embeddings):
  .venv/bin/python accent/probe_invariance.py --cache data/probe-cache --name A_gemma_full
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def balanced(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def gender_nullspace(x, g, rounds=10, seed=0):
    """Projection P removing the directions a linear classifier uses to predict gender."""
    p = np.eye(x.shape[1])
    for r in range(rounds):
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed + r).fit(x @ p, g)
        w = clf.coef_ / np.linalg.norm(clf.coef_)
        p = p @ (np.eye(x.shape[1]) - w.T @ w)
    return p


def cv_acc(model, x, y, folds=10, seed=0):
    pred = cross_val_predict(model, x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    return round(balanced(y, pred), 3)


def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/probe-cache")
    ap.add_argument("--name", default="A_gemma_full")
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20])
    ap.add_argument("--null", type=int, default=30)
    args = ap.parse_args()
    cache = Path(args.cache)
    df = pd.read_csv(cache / "speakers.csv")
    x = unit(np.load(cache / f"{args.name}.npy"))
    g = df.gender.to_numpy()
    keep = (df.region.notna() & (df.region != "mixed")).to_numpy()
    y = df.region.to_numpy()[keep]
    out = {}
    for rounds in args.rounds:
        # Fold-honest: the nullspace is refit inside each fold on training speakers only.
        pred_r = np.empty_like(y)
        for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(x[keep], y):
            idx_tr = np.flatnonzero(keep)[tr]
            p = gender_nullspace(x[idx_tr], g[idx_tr], rounds) if rounds else np.eye(x.shape[1])
            m = lda().fit(x[idx_tr] @ p, y[tr])
            pred_r[te] = m.predict(x[np.flatnonzero(keep)[te]] @ p)
        p_all = gender_nullspace(x, g, rounds) if rounds else np.eye(x.shape[1])
        xp = x @ p_all
        out[rounds] = {
            "region_lda_cv": round(balanced(y, pred_r), 3),
            "gender_residual_cv": cv_acc(LogisticRegression(max_iter=2000), xp, g),
        }
        print(f"nullspace rounds={rounds:>2}", json.dumps(out[rounds]), flush=True)
    # Permutation null for the region score at the chosen setting.
    best = max(r for r in args.rounds if out[r]["gender_residual_cv"] <= 0.6) if any(
        out[r]["gender_residual_cv"] <= 0.6 for r in args.rounds) else args.rounds[-1]
    p_all = gender_nullspace(x, g, best) if best else np.eye(x.shape[1])
    xp = (x @ p_all)[keep]
    null = [cv_acc(lda(), xp, np.random.default_rng(i).permutation(y)) for i in range(args.null)]
    out["chosen_rounds"] = best
    out["region_null_p95"] = round(float(np.quantile(null, 0.95)), 3)
    out["chance"] = round(1 / len(np.unique(y)), 3)
    print("chosen", best, "region null p95", out["region_null_p95"], "chance", out["chance"])
    (cache / f"invariance-{args.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
