"""ai-ids: a machine-learning intrusion detection lab.

The thread limits below are set before numpy/scikit-learn are imported anywhere so
that model training never saturates every core on a laptop that runs hot.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

__version__ = "0.1.0"
NAME = "ai-ids"
