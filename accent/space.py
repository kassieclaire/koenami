"""A gender-invariant American English accent space, fitted on labelled reference speakers.

Recipe, chosen on a dev split of 427 Speech Accent Archive speakers who all read the same
paragraph (probe_gemma.py, probe_gemma_refine.py; the earlier probes are kept for the record):

  features   EmbeddingGemma 2 audio-tower layer 10 + WavLM Base+ layer 11, each pooled as the
             mean and standard deviation over frames ("gemma-wavlm", the default)
  standardise per dimension on the reference speakers, PCA to 128, rescale, unit length
  gender     remove the gender-predictive direction twice (iterative nullspace projection,
             fitted on the reference speakers; a take needs no gender label). One round left
             gender 54-62 % recoverable on held-out speakers, two leave it at chance.
  LDA        shrinkage LDA over dialect regions; coordinates are whitened by the within-region
             spread, so distances are in units of typical variation among one region's speakers.

Held-out test accuracy (30 % of speakers, never used for selection) is about 25 % over 8 regions
(chance 12.5 %). Birthplace is a proxy for accent; treat every reading as approximate.
"""

from dataclasses import dataclass, field

import numpy as np

SR = 16000
MAX_SECONDS = 60


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _pool(h, how):
    h = h.float()
    mean = h.mean(0).cpu().numpy()
    return mean if how == "mean" else np.concatenate([mean, h.std(0).cpu().numpy()])


class _WavLM:
    def __init__(self, device):
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel
        self.torch, self.device = torch, device
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    def layer(self, x, layer, how):
        inp = self.fe(x, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        with self.torch.no_grad():
            return _pool(self.model(inp, output_hidden_states=True).hidden_states[layer][0], how)


class _Gemma:
    """google/embeddinggemma-2 in bf16 (the model card warns fp16 returns NaN); a hook keeps one
    audio-tower layer's frames from each encode."""

    def __init__(self, device, audio_layer=None):
        import torch
        from sentence_transformers import SentenceTransformer
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.st = SentenceTransformer("google/embeddinggemma-2", device=device, model_kwargs={"torch_dtype": dtype},
                                      config_kwargs={"vision_config": None})
        self.caught = None
        if audio_layer is not None:
            def keep(_mod, _inp, out):
                self.caught = (out[0] if isinstance(out, tuple) else out)[0].detach()
            self.st[0].model.audio_tower.layers[audio_layer].register_forward_hook(keep)

    def final(self, x):
        return self.st.encode({"audio": x})

    def audio_layer(self, x, how):
        self.caught = None
        self.st.encode({"audio": x})
        return _pool(self.caught, how)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class GemmaWavLMEmbedder:
    """Default: Gemma audio-tower layer 10 and WavLM layer 11, both mean+std pooled."""
    name = "gemma2-a10ms+wavlm-l11ms"
    recipe = {"pca": 128, "rounds": 2}

    def __init__(self, device=None):
        device = device or _device()
        self.gemma, self.wavlm = _Gemma(device, audio_layer=10), _WavLM(device)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)[: SR * MAX_SECONDS]
        return np.concatenate([self.gemma.audio_layer(x, "meanstd"), self.wavlm.layer(x, 11, "meanstd")])


class GemmaAudioEmbedder:
    """Gemma alone: audio-tower layer 10, mean+std pooled (best Gemma-only setting)."""
    name = "gemma2-a10ms"
    recipe = {"pca": None, "rounds": 2}

    def __init__(self, device=None):
        self.gemma = _Gemma(device or _device(), audio_layer=10)

    def __call__(self, x):
        return self.gemma.audio_layer(np.asarray(x, dtype=np.float32)[: SR * MAX_SECONDS], "meanstd")


class WavLMEmbedder:
    """WavLM alone: layer 11, mean+std pooled."""
    name = "wavlm-l11ms"
    recipe = {"pca": None, "rounds": 2}

    def __init__(self, device=None):
        self.wavlm = _WavLM(device or _device())

    def __call__(self, x):
        return self.wavlm.layer(np.asarray(x, dtype=np.float32)[: SR * MAX_SECONDS], 11, "meanstd")


class GemmaFinalEmbedder:
    """The model's own pooled 768-d embedding: the untuned baseline (about 18 % on test)."""
    name = "gemma2-final"
    recipe = {"pca": None, "rounds": 2}

    def __init__(self, device=None):
        self.gemma = _Gemma(device or _device())

    def __call__(self, x):
        return self.gemma.final(np.asarray(x, dtype=np.float32)[: SR * MAX_SECONDS])


EMBEDDERS = {"gemma-wavlm": GemmaWavLMEmbedder, "gemma": GemmaAudioEmbedder,
             "wavlm": WavLMEmbedder, "gemma-final": GemmaFinalEmbedder}
BY_NAME = {cls.name: cls for cls in EMBEDDERS.values()}


def gender_direction(x, gender, seed=0):
    """Unit direction a logistic regression uses to separate the gender labels."""
    from sklearn.linear_model import LogisticRegression
    w = LogisticRegression(max_iter=3000, random_state=seed).fit(x, gender).coef_[0]
    return (w / np.linalg.norm(w)).astype(np.float32)


@dataclass
class AccentSpace:
    embedder: str
    regions: list
    mu: np.ndarray            # per-dimension centre and scale of the reference features
    sd: np.ndarray
    pca: np.ndarray           # (k, dim) PCA axes, or (0, dim) for none
    pca_mean: np.ndarray
    pca_sd: np.ndarray
    w_gender: np.ndarray      # (rounds, k) removed directions, applied in order
    mean: np.ndarray          # LDA input centre
    scalings: np.ndarray      # (k, axes) LDA axes, whitened by the within-region spread
    centroids: np.ndarray     # (regions, axes) region means in accent space
    explained: np.ndarray = field(default_factory=lambda: np.zeros(0))
    temperature: float = 1.0  # fitted on held-out speakers in build_index.py

    def project(self, x):
        """Features -> standardised, PCA, unit length, gender directions removed (LDA input)."""
        x = (np.asarray(x, dtype=np.float32) - self.mu) / self.sd
        if len(self.pca):
            x = ((x - self.pca_mean) @ self.pca.T) / self.pca_sd
        x = unit(x)
        for w in self.w_gender:
            x = x - np.multiply.outer(x @ w, w)
        return x.astype(np.float32)

    @classmethod
    def fit(cls, x, gender, region, embedder, pca=None, rounds=2):
        """x, gender for every reference speaker; region is None where unassigned ('mixed')."""
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        x = np.asarray(x, dtype=np.float32)
        region = np.asarray(region, dtype=object)
        labelled = np.array([r is not None for r in region])
        mu, sd = x.mean(0), x.std(0) + 1e-6
        xs = (x - mu) / sd
        if pca:
            p = PCA(pca, random_state=0).fit(xs)
            comps, pmean = p.components_.astype(np.float32), p.mean_.astype(np.float32)
            psd = ((xs - pmean) @ comps.T).std(0) + 1e-6
        else:
            comps, pmean, psd = np.zeros((0, x.shape[1]), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
        space = cls(embedder, [], mu, sd, comps, pmean, psd.astype(np.float32), np.zeros((0, 0), np.float32),
                    np.zeros(0, np.float32), np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32))
        xp = space.project(x)
        ws = []
        for r in range(rounds):
            w = gender_direction(xp, gender, seed=r)
            ws.append(w)
            xp = xp - np.outer(xp @ w, w)
        space.w_gender = np.stack(ws) if ws else np.zeros((0, xp.shape[1]), np.float32)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(
            xp[labelled], region[labelled].astype(str))
        k = len(lda.classes_) - 1
        space.regions = list(lda.classes_)
        space.mean = xp[labelled].mean(0).astype(np.float32)
        space.scalings = lda.scalings_[:, :k].astype(np.float32)
        z = space.transform(x[labelled])
        z_ref = lda.transform(xp[labelled])[:, :k]
        assert np.allclose(z, z_ref - z_ref.mean(0), atol=1e-3), "accent axes failed to reproduce the LDA"
        lab = region[labelled].astype(str)
        space.centroids = np.stack([z[lab == r].mean(0) for r in space.regions]).astype(np.float32)
        # Share of the spread between region averages along each axis. (sklearn's
        # explained_variance_ratio_ spreads over every eigenvalue under shrinkage, so it is not this.)
        spread = space.centroids.var(0)
        space.explained = (spread / spread.sum()).astype(np.float32)
        return space

    def transform(self, x):
        return ((self.project(x) - self.mean) @ self.scalings).astype(np.float32)

    def region_profile(self, z, temperature=None):
        """LDA posterior over regions with equal priors (the reference set is South-heavy), softened
        by a temperature: the raw posterior is badly overconfident on speakers the space never saw."""
        d2 = ((self.centroids - z) ** 2).sum(1) / (temperature or self.temperature)
        p = np.exp(-0.5 * (d2 - d2.min()))
        return p / p.sum()

    FIELDS = ("mu", "sd", "pca", "pca_mean", "pca_sd", "w_gender", "mean", "scalings", "centroids", "explained")

    def save(self, path, **extra):
        np.savez(path, embedder=self.embedder, regions=np.array(self.regions),
                 temperature=np.float32(self.temperature), **{f: getattr(self, f) for f in self.FIELDS}, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        space = cls(str(d["embedder"]), [str(r) for r in d["regions"]], *(d[f] for f in cls.FIELDS),
                    float(d["temperature"]))
        return space, d
