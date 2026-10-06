"""A gender-invariant American English accent space, fitted on labelled reference speakers.

Recipe (chosen in probe.py / probe_invariance.py on 427 Speech Accent Archive speakers who all
read the same paragraph): pool one embedding per take, remove the direction a linear classifier
uses to predict the speaker's gender (one round of iterative nullspace projection, so no gender
label is needed for a take), then a shrinkage LDA over dialect regions. The LDA coordinates are
whitened by the within-region spread, so Euclidean distance between two takes is in units of
"typical variation among speakers of one region".

Held-out region accuracy with WavLM layer 8 is about 29 % over 8 regions (chance 12.5 %), a
selection-set figure since the layer was picked on the same speakers. Birthplace is a proxy for
accent; treat every reading as approximate.
"""

from dataclasses import dataclass, field

import numpy as np

SR = 16000
MAX_SECONDS = 60


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class WavLMEmbedder:
    """Mean-pooled hidden states of one WavLM Base+ layer (MIT weights)."""

    def __init__(self, layer=8, device=None):
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(self.device).eval()
        self.layer = layer
        self.name = f"wavlm-base-plus-l{layer}"

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)[: SR * MAX_SECONDS]
        inp = self.fe(x, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        with self.torch.no_grad():
            h = self.model(inp, output_hidden_states=True).hidden_states[self.layer][0]
        return unit(h.mean(0).float().cpu().numpy())


class GemmaEmbedder:
    """google/embeddinggemma-2 audio embedding (bf16; the model card warns fp16 returns NaN)."""

    def __init__(self, device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer("google/embeddinggemma-2", device=device,
                                         model_kwargs={"torch_dtype": dtype},
                                         config_kwargs={"vision_config": None})
        self.name = "embeddinggemma-2"

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)[: SR * MAX_SECONDS]
        return unit(self.model.encode({"audio": x}))


EMBEDDERS = {"wavlm": WavLMEmbedder, "gemma": GemmaEmbedder}


def gender_direction(x, gender, seed=0):
    """Unit direction a logistic regression uses to separate the gender labels."""
    from sklearn.linear_model import LogisticRegression
    w = LogisticRegression(max_iter=2000, random_state=seed).fit(x, gender).coef_[0]
    return (w / np.linalg.norm(w)).astype(np.float32)


@dataclass
class AccentSpace:
    embedder: str
    regions: list
    w_gender: np.ndarray      # removed direction
    mean: np.ndarray          # LDA input centre
    scalings: np.ndarray      # (dim, k) LDA axes, whitened by the within-region spread
    centroids: np.ndarray     # (regions, k) region means in accent space
    explained: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def fit(cls, x, gender, region, embedder):
        """x, gender for every reference speaker; region is None where unassigned ('mixed')."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        x = unit(x)
        region = np.asarray(region, dtype=object)
        labelled = np.array([r is not None for r in region])
        w = gender_direction(x, gender)
        xp = x - np.outer(x @ w, w)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(
            xp[labelled], region[labelled].astype(str))
        k = len(lda.classes_) - 1
        mean = xp[labelled].mean(0)
        space = cls(embedder, list(lda.classes_), w, mean.astype(np.float32),
                    lda.scalings_[:, :k].astype(np.float32),
                    np.zeros((len(lda.classes_), k), np.float32), lda.explained_variance_ratio_[:k])
        z = space.transform(x[labelled])
        z_ref = lda.transform(xp[labelled])[:, :k]
        assert np.allclose(z, z_ref - z_ref.mean(0), atol=1e-3), "accent axes failed to reproduce the LDA"
        lab = region[labelled].astype(str)
        space.centroids = np.stack([z[lab == r].mean(0) for r in space.regions]).astype(np.float32)
        return space

    def transform(self, x):
        x = unit(x)
        xp = x - np.outer(np.atleast_2d(x) @ self.w_gender, self.w_gender).reshape(x.shape)
        return ((xp - self.mean) @ self.scalings).astype(np.float32)

    def region_profile(self, z):
        """LDA posterior over regions with equal priors (the reference set is South-heavy)."""
        d2 = ((self.centroids - z) ** 2).sum(1)
        p = np.exp(-0.5 * (d2 - d2.min()))
        return p / p.sum()

    def save(self, path, **extra):
        np.savez(path, embedder=self.embedder, regions=np.array(self.regions), w_gender=self.w_gender,
                 mean=self.mean, scalings=self.scalings, centroids=self.centroids,
                 explained=self.explained, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        space = cls(str(d["embedder"]), [str(r) for r in d["regions"]], d["w_gender"], d["mean"],
                    d["scalings"], d["centroids"], d["explained"])
        return space, d
