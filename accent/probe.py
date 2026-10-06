"""Does an audio embedding carry American regional accent? A gate before building any UI.

Every Speech Accent Archive speaker reads the same paragraph, so content is held constant and
what differs is the speaker: accent, but also voice, gender, recording conditions. For each
embedding condition this reports leave-one-out region accuracy (k-NN and nearest-centroid,
balanced over regions) against a label-permutation null, and the same for gender as the
confound to beat. Embeddings are cached per condition in --cache.

Run (fwuff):  .venv/bin/python probe.py --audio data/saa_mp3 --meta speaker_information.xlsx
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SR = 16000
STEER = "task: classification | query: regional accent of the speaker <|audio|>"


def load_speakers(meta_path, audio_dir, regions_path=HERE / "regions.json"):
    cfg = json.loads(Path(regions_path).read_text())
    df = pd.read_excel(meta_path)
    df = df[(df.native_language.str.lower() == "english") & (df.country.str.lower().str.strip() == "usa")].copy()
    df["state"] = df.state_or_province.str.lower().str.strip()
    df["city_key"] = df.state + "|" + df.city.str.lower().str.strip()
    df["region"] = [cfg["cities"].get(c, cfg["states"].get(s)) for c, s in zip(df.city_key, df.state)]
    df["path"] = [str(Path(audio_dir) / s) for s in df.speech_sample]
    df = df[[Path(p).exists() for p in df.path]]
    return df.reset_index(drop=True), cfg


def wavs(df, seconds=None):
    for p in df.path:
        w, _ = librosa.load(p, sr=SR, mono=True)
        yield w[: int(seconds * SR)] if seconds else w


def embed_gemma(df, steer=False, seconds=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"vision_config": None})
    out = [model.encode({"text": STEER, "audio": w} if steer else {"audio": w}) for w in wavs(df, seconds)]
    return np.stack(out).astype(np.float32)


def embed_wavlm(df, layer=6):
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval()
    out = []
    for w in wavs(df):
        x = fe(w, sampling_rate=SR, return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            h = model(x, output_hidden_states=True).hidden_states[layer][0]
        out.append(h.mean(0).float().cpu().numpy())
    return np.stack(out)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def balanced(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def knn_loo(x, y, k=7):
    s = unit(x) @ unit(x).T
    np.fill_diagonal(s, -np.inf)
    nn = np.argsort(-s, axis=1)[:, :k]
    pred = []
    for row in nn:
        votes = pd.Series(y[row]).value_counts()
        pred.append(votes.index[0])
    return balanced(y, np.array(pred))


def centroid_loo(x, y):
    xu = unit(x)
    classes = np.unique(y)
    pred = []
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        cents = unit(np.stack([xu[mask & (y == c)].mean(0) for c in classes]))
        pred.append(classes[np.argmax(cents @ xu[i])])
    return balanced(y, np.array(pred))


def evaluate(x, y, perms=200, seed=0):
    rng = np.random.default_rng(seed)
    knn, cen = knn_loo(x, y), centroid_loo(x, y)
    null = np.array([knn_loo(x, rng.permutation(y)) for _ in range(perms)])
    return {"knn": round(knn, 3), "centroid": round(cen, 3), "chance": round(1 / len(np.unique(y)), 3),
            "null_p95": round(float(np.quantile(null, 0.95)), 3),
            "p": round(float((1 + (null >= knn).sum()) / (1 + perms)), 4)}


def lda_cv(x, y, folds=10, seed=0):
    """Balanced accuracy of a shrinkage LDA accent subspace, scored only on held-out speakers."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x[tr], y[tr])
        pred[te] = lda.predict(x[te])
    return round(balanced(y, pred), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cache", default="data/probe-cache")
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    df, _ = load_speakers(args.meta, args.audio)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    conditions = {
        "A_gemma_full": lambda: embed_gemma(df),
        "B_gemma_steered": lambda: embed_gemma(df, steer=True),
        "C_gemma_first10s": lambda: embed_gemma(df, seconds=10),
        "D_wavlm_l6_full": lambda: embed_wavlm(df),
    }
    keep = (df.region.notna() & (df.region != "mixed")).to_numpy()
    region = df.region.to_numpy()[keep]
    gender = df.gender.to_numpy()
    print(f"speakers: {len(df)} ({keep.sum()} with a region)")
    print(df[keep].region.value_counts().to_string())
    results = {}
    for name, fn in conditions.items():
        f = cache / f"{name}.npy"
        x = np.load(f) if f.exists() else fn()
        np.save(f, x)
        # Remove the gender mean so region is not judged through a gender imbalance.
        xg = unit(x).copy()
        for g in np.unique(gender):
            xg[gender == g] -= xg[gender == g].mean(0)
        results[name] = {"region": evaluate(x[keep], region, args.perms),
                         "region_gender_centered": evaluate(xg[keep], region, args.perms),
                         "gender": evaluate(x, gender, args.perms),
                         "region_lda_cv": lda_cv(xg[keep], region),
                         "region_lda_null_p95": round(float(np.quantile(
                             [lda_cv(xg[keep], np.random.default_rng(i).permutation(region)) for i in range(30)],
                             0.95)), 3)}
        print(name, json.dumps(results[name]))
    (cache / "results.json").write_text(json.dumps(results, indent=1))
    df.drop(columns=["path"]).to_csv(cache / "speakers.csv", index=False)


if __name__ == "__main__":
    main()
