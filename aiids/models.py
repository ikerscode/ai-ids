"""Training, calibration and persistence of the full detector bundle."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import config
from .detectors import AutoencoderDetector, EntityBaseline, IForestDetector, QuantileNormaliser
from .features import FeatureTransform, feature_set
from .schema import BENIGN
from .supervised import SupervisedDetector

DETECTORS = ["iforest", "autoencoder", "baseline"]
EXCLUDED_LABELS = {"unlabelled", ""}


@dataclass
class ModelBundle:
    transform: FeatureTransform
    iforest: IForestDetector
    autoencoder: AutoencoderDetector
    baseline: EntityBaseline
    norms: dict
    supervised: SupervisedDetector | None
    thresholds: dict = field(default_factory=dict)
    entity_tau: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- scoring
    @property
    def features(self) -> list[str]:
        return self.transform.features

    def ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the time-window features when the model needs them and the flows lack them.

        Datasets built by the loaders already carry them (computed over the whole
        capture); this fallback only covers ad-hoc frames, where the windows can
        only see the flows of that frame.
        """
        if all(f in df.columns for f in self.features):
            return df
        from .windows import add_window_features
        return add_window_features(df)

    def anomaly_scores(self, df: pd.DataFrame, baseline: EntityBaseline | None = None) -> dict:
        """Raw and normalised scores of the three unsupervised detectors."""
        df = self.ensure_features(df)
        z = self.transform.scaled(df)
        bl = baseline or self.baseline
        raw = {"iforest": self.iforest.score(z), "autoencoder": self.autoencoder.score(z), "baseline": bl.score(df)}
        norm = {k: self.norms[k](v) for k, v in raw.items()}
        ens = np.mean([norm[k] for k in DETECTORS], axis=0)
        return {"raw": raw, "norm": norm, "ens": ens}

    def supervised_scores(self, df: pd.DataFrame) -> dict:
        n = len(df)
        if self.supervised is None:
            return {"p_attack": np.zeros(n), "family": np.array(["none"] * n, dtype=object), "family_conf": np.zeros(n)}
        return self.supervised.predict(self.transform.transform(self.ensure_features(df)))

    # ---------------------------------------------------------------- calibration
    def calibrate(self, val: pd.DataFrame, train_benign: pd.DataFrame) -> None:
        """Pick thresholds on benign validation traffic to hit the target false-positive rates."""
        vb = val[val["label"] == BENIGN]
        if len(vb) < 200:  # fall back to training benign when there is no usable validation set
            vb = train_benign
        a = self.anomaly_scores(vb)
        s = self.supervised_scores(vb)
        a_train = self.anomaly_scores(train_benign.sample(min(len(train_benign), 8000), random_state=0))
        th = {
            "tau_ens": float(np.quantile(a["ens"], 1 - config.TARGET_FPR_ANOMALY)),
            "tau_ens_naive": float(np.quantile(a_train["ens"], 0.95)),
            "vote": {k: float(np.quantile(a["norm"][k], 1 - config.VOTE_FPR)) for k in DETECTORS},
            "tau_sup": float(max(config.MIN_SUPERVISED_CONF,
                                 np.quantile(s["p_attack"], 1 - config.TARGET_FPR_SUPERVISED))),
            "target_fpr_anomaly": config.TARGET_FPR_ANOMALY,
            "target_fpr_supervised": config.TARGET_FPR_SUPERVISED,
            "vote_fpr": config.VOTE_FPR, "min_votes": config.MIN_VOTES,
            "calibration_flows": int(len(vb)),
        }
        # Per-host adaptive thresholds: a host whose normal traffic is naturally odd
        # (backup server, OT poller) gets a higher bar than the global one.
        ent = EntityBaseline._entities(vb).to_numpy()
        tau_e = {}
        for name in pd.unique(ent):
            m = ent == name
            if m.sum() >= 50:
                q = float(np.quantile(a["ens"][m], 0.995))
                if q > th["tau_ens"]:
                    tau_e[name] = q
        self.thresholds, self.entity_tau = th, tau_e


def train_bundle(train: pd.DataFrame, val: pd.DataFrame, extra_benign: pd.DataFrame | None = None,
                 exclude_families: tuple = (), seed: int = 0, log=print,
                 windows: bool | None = None) -> ModelBundle:
    """Fit every layer. Anomaly layers see benign flows only.

    windows: use the time-window features (None = use them when the flows have them).
    """
    t0 = time.time()
    benign = train[train["label"] == BENIGN]
    if "label_note" in benign:  # benign-labelled flows a loader flagged as probable attack traffic
        benign = benign[benign["label_note"].astype(str) == ""]
    if extra_benign is not None and len(extra_benign):
        benign = pd.concat([benign, extra_benign], ignore_index=True)
    if len(benign) < 50:
        raise ValueError("need at least 50 benign training flows")
    if windows is None:
        from .windows import has_window_features
        windows = has_window_features(train)
    transform = FeatureTransform(feature_set(windows)).fit(benign)
    z = transform.scaled(benign)
    log(f"training on {len(benign)} benign flows, {len(train) - (train['label'] == BENIGN).sum()} attack flows, "
        f"{len(transform.features)} features{' (with time windows)' if windows else ''}")
    iforest = IForestDetector(seed).fit(z)
    ae_idx = np.random.default_rng(seed).permutation(len(z))[:20000]
    autoencoder = AutoencoderDetector(seed, n_features=len(transform.features)).fit(z[ae_idx])
    log(f"  isolation forest + autoencoder fitted ({time.time() - t0:.1f}s)")
    baseline = EntityBaseline().fit(benign)
    log(f"  per-entity baselines: {len(baseline.profiles) - 1} entities")
    raw = {"iforest": iforest.score(z), "autoencoder": autoencoder.score(z), "baseline": baseline.score(benign)}
    norms = {k: QuantileNormaliser().fit(v) for k, v in raw.items()}

    sup_rows = train[~train["label"].isin(EXCLUDED_LABELS | set(exclude_families))]
    if extra_benign is not None and len(extra_benign):
        sup_rows = pd.concat([sup_rows, extra_benign.assign(label=BENIGN)], ignore_index=True)
    supervised = None
    if sup_rows["label"].nunique() >= 2:
        supervised = SupervisedDetector(seed).fit(transform.transform(sup_rows), sup_rows["label"].to_numpy())
        log(f"  supervised classes: {supervised.classes_} ({time.time() - t0:.1f}s)")
    bundle = ModelBundle(transform, iforest, autoencoder, baseline, norms, supervised)
    bundle.calibrate(val, benign)
    bundle.meta = {"trained_at": time.time(), "train_flows": int(len(train)), "benign_flows": int(len(benign)),
                   "excluded_families": list(exclude_families), "seconds": round(time.time() - t0, 1),
                   "window_features": bool(windows), "n_features": len(transform.features),
                   "classes": supervised.classes_ if supervised else [BENIGN]}
    log(f"  thresholds: tau_ens={bundle.thresholds['tau_ens']:.2f} tau_sup={bundle.thresholds['tau_sup']:.3f} "
        f"entity overrides={len(bundle.entity_tau)} ({time.time() - t0:.1f}s)")
    return bundle


def save_bundle(bundle: ModelBundle, name: str = "current") -> str:
    """Write the bundle into config.MODEL_DIR (set AIIDS_MODEL_DIR to keep dataset models apart)."""
    config.ensure_dirs()
    path = config.MODEL_DIR / f"{name}.joblib"
    joblib.dump(bundle, path, compress=3)
    return str(path)


def load_bundle(name: str = "current") -> ModelBundle | None:
    path = config.MODEL_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None
