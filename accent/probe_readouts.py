"""Score candidate recipes on the readouts the accent page shows, not on classification alone.

The tuning probes selected on LDA classification accuracy; the page leads with closeness to a
region average and with speaker-to-speaker profile overlap. Here every held-out speaker is placed
by a fold model that never saw them, and all comparisons stay inside that fold model's own
coordinates (no cross-fold alignment). Reported per recipe:

  acc     balanced accuracy, nearest region average
  rank    rank of the speaker's own region average among the 8 (0 = nearest; chance 3.5)
  pairAUC same- vs different-region speaker pairs by distance (0.5 = no signal)
  gap     mean distance between region averages / median speaker-pair distance
  leak    gender recoverable from the projected features (chance 0.5)

  .venv-accent/bin/python accent/probe_readouts.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import load_speakers  # noqa: E402
from space import AccentSpace  # noqa: E402


def evaluate(x, g, region, pca, rounds, folds=10, seed=0):
    lab = np.flatnonzero([r is not None for r in region])
    r_all = region[lab].astype(str)
    accs, ranks, aucs, gaps, leak_t, leak_p = [], [], [], [], [], []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(lab, r_all):
        train = np.setdiff1d(np.arange(len(x)), lab[te])
        m = AccentSpace.fit(x[train], g[train], region[train], "probe", pca=pca, rounds=rounds)
        z = m.transform(x[lab[te]])
        y = np.array([m.regions.index(v) for v in r_all[te]])
        dc = np.linalg.norm(z[:, None] - m.centroids[None], axis=-1)
        accs.append((dc.argmin(1), y))
        ranks += list((dc < dc[np.arange(len(z)), y][:, None]).sum(1))
        iu = np.triu_indices(len(z), 1)
        same = (y[:, None] == y[None])[iu]
        d = np.linalg.norm(z[:, None] - z[None], axis=-1)[iu]
        if same.any() and (~same).any():
            aucs.append(roc_auc_score(same, -d))
        cg = np.linalg.norm(m.centroids[:, None] - m.centroids[None], axis=-1)[np.triu_indices(len(m.centroids), 1)]
        gaps.append(cg.mean() / np.median(d))
        clf = LogisticRegression(max_iter=3000).fit(m.project(x[train]), g[train])
        leak_t += list(g[lab[te]])
        leak_p += list(clf.predict(m.project(x[lab[te]])))
    pred = np.concatenate([p for p, _ in accs])
    true = np.concatenate([t for _, t in accs])
    acc = np.mean([np.mean(pred[true == c] == c) for c in np.unique(true)])
    lt, lp = np.array(leak_t), np.array(leak_p)
    leak = np.mean([np.mean(lp[lt == c] == c) for c in np.unique(lt)])
    return {"acc": acc, "rank": np.mean(ranks), "pairAUC": np.mean(aucs), "gap": np.mean(gaps), "leak": leak}


def main():
    df, _ = load_speakers("data/speaker_information.xlsx", "data/saa_mp3")
    g = df.gender.to_numpy()
    region = np.array([r if r and r != "mixed" else None for r in df.region], dtype=object)
    combo = np.load("data/accent-embeddings-gemma2-a10ms+wavlm-l11ms.npz")["full"]
    old = np.load("data/accent-embeddings-wavlm-base-plus-l8.npz")["full"]
    gemma, wavlm11 = combo[:, :2048], combo[:, 2048:]
    candidates = [
        ("wavlm L8 mean (old page)", old, None, 1),
        ("wavlm L8 mean", old, None, 2),
        ("wavlm L8 mean, pca64", old, 64, 2),
        ("wavlm L11 ms", wavlm11, None, 2),
        ("wavlm L11 ms, pca64", wavlm11, 64, 2),
        ("gemma A10 ms", gemma, None, 2),
        ("gemma A10 ms, pca64", gemma, 64, 2),
        ("gemma A10 ms, pca32", gemma, 32, 2),
        ("gemma+wavlm, pca128 (new)", combo, 128, 2),
        ("gemma+wavlm, pca64", combo, 64, 2),
        ("gemma+wavlm, pca32", combo, 32, 2),
        ("gemma+wavlm, no pca", combo, None, 2),
    ]
    print(f"{'recipe':<28} {'acc':>6} {'rank':>6} {'pairAUC':>8} {'gap':>6} {'leak':>6}")
    for name, x, pca, rounds in candidates:
        r = evaluate(x, g, region, pca, rounds)
        print(f"{name:<28} {r['acc']:6.3f} {r['rank']:6.2f} {r['pairAUC']:8.3f} {r['gap']:6.2f} {r['leak']:6.3f}", flush=True)


if __name__ == "__main__":
    main()
