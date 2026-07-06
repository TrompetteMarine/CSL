"""Shared paths and helpers for the experiment scripts."""
import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # code/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
if os.path.isdir(os.path.join(ROOT, "paper")):
    FIGDIR = os.path.abspath(os.path.join(ROOT, "paper", "figures"))
else:
    FIGDIR = os.path.abspath(os.path.join(ROOT, "..", "paper", "figures"))
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

from credibility.config import SCALE                                    # noqa: E402


def get_scale(default="standard"):
    for a in sys.argv[1:]:
        if a in SCALE:
            return a
    return os.environ.get("ECSL_SCALE", default)


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
