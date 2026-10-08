"""Second tuning round on the cached sweep features (no GPU): multi-layer audio-tower features,
PCA before the LDA, and Gemma + WavLM together. Selection on the dev split only (same split
and CV as probe_gemma.py); the chosen configuration is scored once on the test split.

  .venv-accent/bin/python accent/probe_gemma_refine.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_invariance import balanced  # noqa: E402
from space import gender_direction, unit  # noqa: E402

OUT = Path("data/gemma-sweep")


def standardize(x, train):
    x = x - x[train].mean(0)
    return x / (x[train].std(0) + 1e-6)


def predict(x, g, y, train, test, pca=None):
    xs = standardize(x, train)
    if pca:
        p = PCA(pca, random_state=0).fit(xs[train])
        xs = p.transform(xs)
        xs /= xs[train].std(0) + 1e-6
    w = gender_direction(unit(xs[train]), g[train])
    xp = unit(xs)
    xp = xp - np.outer(xp @ w, w)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(xp[train], y[train])
    return lda.predict(xp[test])


def main():
    feats = np.load(OUT / "features.npz")
    df = pd.read_csv(OUT / "speakers.csv")
    split = json.loads((OUT / "split.json").read_text())
    dev, test = np.array(split["dev"]), np.array(split["test"])
    y, g = df.region.to_numpy(), df.gender.to_numpy()
    f = lambda k: feats[k].astype(np.float32)  # noqa: E731
    z = lambda x: standardize(x, dev)  # noqa: E731 - per-block scale before concatenating

    cands = {
        "audio10_ms": f("g_audio10_meanstd"),
        "audio09-11_ms concat": np.hstack([z(f(f"g_audio{i:02d}_meanstd")) for i in (9, 10, 11)]),
        "audio08-11_ms concat": np.hstack([z(f(f"g_audio{i:02d}_meanstd")) for i in (8, 9, 10, 11)]),
        "audio09-11_ms average": np.mean([z(f(f"g_audio{i:02d}_meanstd")) for i in (9, 10, 11)], 0),
        "audio06-11_ms average": np.mean([z(f(f"g_audio{i:02d}_meanstd")) for i in range(6, 12)], 0),
        "audio10_ms + wavlm11_ms": np.hstack([z(f("g_audio10_meanstd")), z(f("w_layer11_meanstd"))]),
        "audio10_ms + wavlm08_mean": np.hstack([z(f("g_audio10_meanstd")), z(f("w_layer08_mean"))]),
    }
    rows = []
    for name, x in cands.items():
        for pca in (None, 64, 128, 256):
            accs = []
            for seed in (0, 1, 2):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(dev, y[dev]):
                    accs.append(balanced(y[dev[te]], predict(x, g, y, dev[tr], dev[te], pca)))
            m, se = float(np.mean(accs)), float(np.std(accs) / np.sqrt(len(accs)))
            rows.append((name, pca or 0, m, se))
            print(f"{name:<28} pca={pca or '-':<4} dev CV {m:.3f} ± {se:.3f}", flush=True)
    res = pd.DataFrame(rows, columns=["config", "pca", "dev_cv", "se"]).sort_values("dev_cv", ascending=False)
    res.to_csv(OUT / "refine_dev.csv", index=False)

    best = res.iloc[0]
    gemma_only = res[~res.config.str.contains("wavlm")].iloc[0]
    rng = np.random.default_rng(0)
    out = {}
    for label, row in [("best overall", best), ("best Gemma-only", gemma_only)]:
        x = cands[row.config]
        pred = predict(x, g, y, dev, test, int(row.pca) or None)
        acc = balanced(y[test], pred)
        boots = [balanced(y[test][b], pred[b]) for b in (rng.integers(0, len(test), len(test)) for _ in range(1000))]
        ci = [round(float(np.quantile(boots, q)), 3) for q in (0.025, 0.975)]
        out[label] = {"config": row.config, "pca": int(row.pca), "dev_cv": round(row.dev_cv, 3), "test": round(acc, 3), "ci95": ci}
        print(f"TEST {label:<16} {row.config} pca={int(row.pca) or '-'}: {acc:.3f}  95% CI {ci}", flush=True)
    (OUT / "refine_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
