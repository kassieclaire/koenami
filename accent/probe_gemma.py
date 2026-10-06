"""Tune the EmbeddingGemma 2 accent representation, with a held-out test set.

The first probe only used the model's final pooled 768-d embedding, which is trained for
semantic retrieval. This sweeps internal representations instead: every layer of the 12-layer
conformer audio tower (1024-d) and of the 24-layer text backbone the audio is projected into
(512-d), pooled by mean or mean+std, plus windowed averages of the final embedding. WavLM Base+
gets the same layer and pooling sweep so the comparison is like for like.

Selection happens only on a dev split (70 % of labelled speakers, stratified by region) with
repeated 10-fold CV; the chosen configuration per model family is then scored once on the 30 %
test split. Every score uses the page's recipe: remove one gender direction (fitted on training
speakers), shrinkage LDA over regions, balanced accuracy.

  extract:  .venv-accent/bin/python accent/probe_gemma.py extract
  evaluate: .venv-accent/bin/python accent/probe_gemma.py evaluate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import load_speakers  # noqa: E402
from space import SR, unit  # noqa: E402

OUT = Path("data/gemma-sweep")
WINDOW, HOP = 6.0, 3.0


def pools(h):
    """h: (frames, dim) -> mean and mean+std pooled vectors."""
    h = h.float()
    mean = h.mean(0)
    return mean.cpu().numpy(), np.concatenate([mean.cpu().numpy(), h.std(0).cpu().numpy()])


def extract(df):
    import librosa
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoFeatureExtractor, WavLMModel

    st = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                             model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"vision_config": None})
    m = st[0].model
    caught = {}

    def hook(name):
        def f(_mod, _inp, out):
            caught[name] = (out[0] if isinstance(out, tuple) else out)[0].detach()
        return f

    for i, layer in enumerate(m.audio_tower.layers):
        layer.register_forward_hook(hook(f"audio{i:02d}"))
    for i, layer in enumerate(m.language_model.layers):
        layer.register_forward_hook(hook(f"text{i:02d}"))

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval()

    feats = {}

    def put(key, v):
        feats.setdefault(key, []).append(v.astype(np.float16))

    for n, path in enumerate(df.path):
        w, _ = librosa.load(path, sr=SR, mono=True)
        caught.clear()
        put("gemma_final", st.encode({"audio": w}))
        for name, h in caught.items():
            mean, ms = pools(h)
            put(f"g_{name}_mean", mean)
            put(f"g_{name}_meanstd", ms)
        # Windowed: average the final embedding over 6 s windows (3 s hop).
        win, hop = int(WINDOW * SR), int(HOP * SR)
        chunks = [w[s:s + win] for s in range(0, max(1, len(w) - win + 1), hop)]
        put("gemma_final_windowed", unit(np.stack([st.encode({"audio": c}) for c in chunks])).mean(0))
        with torch.no_grad():
            hs = wavlm(fe(w, sampling_rate=SR, return_tensors="pt").input_values.cuda(),
                       output_hidden_states=True).hidden_states
        for i, h in enumerate(hs):
            mean, ms = pools(h[0])
            put(f"w_layer{i:02d}_mean", mean)
            put(f"w_layer{i:02d}_meanstd", ms)
        if n % 50 == 0:
            print(f"extracted {n}/{len(df)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "features.npz", **{k: np.stack(v) for k, v in feats.items()})
    df.drop(columns=["path"]).to_csv(OUT / "speakers.csv", index=False)
    print("features:", len(feats))


def score(x, g, y, train, test):
    """Fit nullspace + LDA on `train`, balanced accuracy on `test` (indices into x)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from space import gender_direction
    from probe_invariance import balanced
    xs = x - x[train].mean(0)
    xs /= xs[train].std(0) + 1e-6  # per-dimension scale: layers differ wildly in range
    w = gender_direction(unit(xs[train]), g[train])
    xp = unit(xs)
    xp = xp - np.outer(xp @ w, w)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(xp[train], y[train])
    return balanced(y[test], lda.predict(xp[test]))


def evaluate():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, train_test_split
    feats = np.load(OUT / "features.npz")
    df = pd.read_csv(OUT / "speakers.csv")
    lab = np.flatnonzero((df.region.notna() & (df.region != "mixed")).to_numpy())
    y_all, g_all = df.region.to_numpy(), df.gender.to_numpy()
    dev, test = train_test_split(lab, test_size=0.3, stratify=y_all[lab], random_state=2026)
    (OUT / "split.json").write_text(json.dumps({"dev": dev.tolist(), "test": test.tolist()}))

    def dev_cv(x):
        accs = []
        for seed in (0, 1, 2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(dev, y_all[dev]):
                accs.append(score(x, g_all, y_all, dev[tr], dev[te]))
        return float(np.mean(accs)), float(np.std(accs) / np.sqrt(len(accs)))

    rows = []
    for key in sorted(feats.files):
        x = feats[key].astype(np.float32)
        m, se = dev_cv(x)
        rows.append((key, m, se))
        print(f"{key:<28} dev CV {m:.3f} ± {se:.3f}", flush=True)
    res = pd.DataFrame(rows, columns=["feature", "dev_cv", "se"]).sort_values("dev_cv", ascending=False)
    res.to_csv(OUT / "dev_results.csv", index=False)

    # One configuration per family, chosen on dev only, scored once on test.
    family = {"gemma": res[res.feature.str.startswith(("g_", "gemma"))].iloc[0].feature,
              "wavlm": res[res.feature.str.startswith("w_")].iloc[0].feature,
              "gemma_final (baseline)": "gemma_final"}
    rng = np.random.default_rng(0)
    out = {}
    for fam, key in family.items():
        x = feats[key].astype(np.float32)
        acc = score(x, g_all, y_all, dev, test)
        # Bootstrap the test speakers for an interval; permutation null for chance.
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: F401
        boots = []
        for _ in range(300):
            bt = rng.choice(test, len(test))
            boots.append(score(x, g_all, y_all, dev, bt))
        out[fam] = {"feature": key, "test": round(acc, 3),
                    "ci95": [round(float(np.quantile(boots, q)), 3) for q in (0.025, 0.975)]}
        print(f"TEST {fam:<24} {key:<28} {acc:.3f}  95% CI {out[fam]['ci95']}", flush=True)
    null = []
    for i in range(100):
        yp = y_all.copy()
        yp[lab] = np.random.default_rng(i).permutation(y_all[lab])
        null.append(score(feats[family["gemma"]].astype(np.float32), g_all, yp, dev, test))
    out["null_p95"] = round(float(np.quantile(null, 0.95)), 3)
    out["chance"] = round(1 / len(np.unique(y_all[lab])), 3)
    print("test null p95", out["null_p95"], "chance", out["chance"])
    (OUT / "test_results.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "evaluate"])
    ap.add_argument("--audio", default="data/saa_mp3")
    ap.add_argument("--meta", default="data/speaker_information.xlsx")
    args = ap.parse_args()
    if args.stage == "extract":
        df, _ = load_speakers(args.meta, args.audio)
        extract(df)
    else:
        evaluate()


if __name__ == "__main__":
    main()
