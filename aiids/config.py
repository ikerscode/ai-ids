"""Paths and tunable settings in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "datasets"
MODEL_DIR = DATA_DIR / "models"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "aiids.sqlite"

PORT = 8105
N_JOBS = 2  # never more than two worker threads for model training

# False-positive budget used when calibrating thresholds on benign validation traffic.
TARGET_FPR_ANOMALY = 0.003
TARGET_FPR_SUPERVISED = 0.002
# Each detector "votes" when it is above its own (looser) per-detector threshold.
VOTE_FPR = 0.01
MIN_VOTES = 2
# Supervised confidence below this counts as "not sure" for the open-set rule.
MIN_SUPERVISED_CONF = 0.5

# Alert aggregation: flow alerts with the same key closer than this merge into one.
AGG_GAP_SECONDS = 300
# Correlation: alerts that share an entity and are within this gap join an incident.
CORR_GAP_SECONDS = 2 * 3600

CONTEXT_WINDOW_SHORT = 60.0
CONTEXT_WINDOW_LONG = 3600.0


def ensure_dirs() -> None:
    for d in (DATA_DIR, DATASET_DIR, MODEL_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)
