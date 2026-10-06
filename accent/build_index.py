"""Build the American English accent index the studio's accent page reads.

Writes data/accent-index-<embedder>.npz (the fitted space and calibration, read by server.py)
and data/accent-reference-<embedder>.json (one point per reference speaker for the map).

Reference speakers: Speech Accent Archive (Weinberger & Kelley, CC BY-NC-SA 4.0), US-born native
English speakers, all reading the "Please call Stella" paragraph; regions from accent/regions.json.

Map coordinates are cross-fitted: each speaker is placed by a space fitted without them, then
rotated onto the full space (orthogonal Procrustes on the shared training speakers), so the
reference clusters are not tighter on screen than the space can tell apart.

Run (GPU host):
  .venv/bin/python accent/build_index.py --audio data/saa_mp3 --meta speaker_information.xlsx --embedder wavlm
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import load_speakers  # noqa: E402
from space import EMBEDDERS, SR, AccentSpace  # noqa: E402

PASSAGE = ("Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh "
           "snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need "
           "a small plastic snake and a big toy frog for the kids. She can scoop these things into three red "
           "bags, and we will go meet her Wednesday at the train station.")


def embed_all(df, embedder, cache):
    if cache.exists():
        d = np.load(cache)
        return d["full"], d["half1"], d["half2"]
    full, h1, h2 = [], [], []
    for i, p in enumerate(df.path):
        w, _ = librosa.load(p, sr=SR, mono=True)
        mid = len(w) // 2
        full.append(embedder(w)), h1.append(embedder(w[:mid])), h2.append(embedder(w[mid:]))
        if i % 50 == 0:
            print(f"embedded {i}/{len(df)}", flush=True)
    full, h1, h2 = map(np.stack, (full, h1, h2))
    np.savez(cache, full=full, half1=h1, half2=h2)
    return full, h1, h2


def quantiles(v):
    return np.quantile(np.asarray(v, dtype=np.float64), np.linspace(0, 1, 101)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--embedder", choices=sorted(EMBEDDERS), default="wavlm")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, cfg = load_speakers(args.meta, args.audio)
    embedder = EMBEDDERS[args.embedder]()
    x, h1, h2 = embed_all(df, embedder, out / f"accent-embeddings-{embedder.name}.npz")
    gender = df.gender.to_numpy()
    region = np.array([r if r and r != "mixed" else None for r in df.region], dtype=object)
    lab = np.flatnonzero([r is not None for r in region])

    space = AccentSpace.fit(x, gender, region, embedder.name)
    z_full = space.transform(x)

    # Cross-fitted coordinates for the labelled speakers.
    z_cf = z_full.copy()
    correct = []
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(lab, region[lab].astype(str)):
        train = np.setdiff1d(np.arange(len(df)), lab[te])
        fold = AccentSpace.fit(x[train], gender[train], region[train], embedder.name)
        a, b = fold.transform(x[train]), z_full[train]
        ma, mb = a.mean(0), b.mean(0)
        r, _ = orthogonal_procrustes(a - ma, b - mb)
        z_cf[lab[te]] = (fold.transform(x[lab[te]]) - ma) @ r + mb
        for i in lab[te]:
            prof = fold.region_profile(fold.transform(x[i]))
            correct.append((region[i], fold.regions[int(np.argmax(prof))]))
    regions = space.regions
    acc = float(np.mean([np.mean([p == t for t, p in correct if t == r]) for r in regions]))
    print(f"held-out region accuracy (balanced, nearest centroid): {acc:.3f}  chance {1 / len(regions):.3f}")

    # Calibration distributions, all from cross-fitted coordinates.
    zl, rl = z_cf[lab], region[lab].astype(str)
    d = np.sqrt(((zl[:, None] - zl[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(zl), 1)
    same = (rl[:, None] == rl[None])[iu]
    to_centroid = np.linalg.norm(zl - space.centroids[[regions.index(r) for r in rl]], axis=1)
    noise = np.linalg.norm(space.transform(h1) - space.transform(h2), axis=1)
    # Temperature: the raw posterior is overconfident on unseen speakers (held-out log-loss worse
    # than a uniform guess), so soften it to minimise held-out log-loss on the cross-fitted points.
    y_idx = np.array([regions.index(r) for r in rl])
    def nll(t):
        p = np.stack([space.region_profile(v, t) for v in zl])
        return -np.mean(np.log(p[np.arange(len(zl)), y_idx] + 1e-12))
    grid = np.round(np.arange(1, 16.01, 0.25), 2)
    losses = [nll(t) for t in grid]
    space.temperature = float(grid[int(np.argmin(losses))])
    print(f"temperature {space.temperature}: held-out log-loss {min(losses):.3f} (T=1 {losses[0]:.3f}, "
          f"uniform {np.log(len(regions)):.3f})")
    # Profile overlap (sum of min over regions) is the readout the page leads with: on these
    # speakers it separates same- from different-region pairs better than any distance tried.
    prof = np.stack([space.region_profile(v) for v in zl])
    overlap = np.minimum(prof[:, None], prof[None]).sum(-1)[iu]
    from sklearn.metrics import roc_auc_score
    auc_overlap = float(roc_auc_score(same, overlap))
    auc_distance = float(roc_auc_score(same, -d[iu]))
    print(f"same- vs different-region pair AUC: profile overlap {auc_overlap:.3f}, distance {auc_distance:.3f}")
    print(f"median distance: same-region pair {np.median(d[iu][same]):.2f}, different-region pair "
          f"{np.median(d[iu][~same]):.2f}, speaker to own centroid {np.median(to_centroid):.2f}, "
          f"half-vs-half noise {np.median(noise):.2f}")

    space.save(out / f"accent-index-{embedder.name}.npz",
               same_pair=quantiles(d[iu][same]), diff_pair=quantiles(d[iu][~same]),
               to_centroid=quantiles(to_centroid), noise=quantiles(noise),
               overlap_same=quantiles(overlap[same]), overlap_diff=quantiles(overlap[~same]),
               heldout_accuracy=np.float32(acc), ids=df.speech_sample.str.replace(".mp3", "").to_numpy().astype(str),
               z=z_cf)
    reference = {
        "embedder": embedder.name,
        "passage": PASSAGE,
        "source": "Speech Accent Archive (Weinberger & Kelley, George Mason University), CC BY-NC-SA 4.0",
        "regions": [{"id": r, "label": cfg["regions"][r], "n": int((rl == r).sum()),
                     "centroid": space.centroids[i].round(4).tolist()} for i, r in enumerate(regions)],
        "explained": space.explained.round(4).tolist(),
        "heldout_accuracy": round(acc, 3),
        "temperature": space.temperature,
        "pair_auc": {"overlap": round(auc_overlap, 3), "distance": round(auc_distance, 3)},
        "calibration": {k: quantiles(v).round(4).tolist() for k, v in
                        [("same_pair", d[iu][same]), ("diff_pair", d[iu][~same]),
                         ("to_centroid", to_centroid), ("noise", noise),
                         ("overlap_same", overlap[same]), ("overlap_diff", overlap[~same])]},
        "speakers": [{"id": sid.replace(".mp3", ""), "region": df.region[i] or "mixed",
                      "state": df.state[i], "city": str(df.city[i]).strip(),
                      "age": None if np.isnan(df.age[i]) else int(df.age[i]),
                      "z": z_cf[i].round(4).tolist()} for i, sid in enumerate(df.speech_sample)],
    }
    (out / f"accent-reference-{embedder.name}.json").write_text(json.dumps(reference))
    print("wrote", out / f"accent-index-{embedder.name}.npz", "and the reference payload")


if __name__ == "__main__":
    main()
