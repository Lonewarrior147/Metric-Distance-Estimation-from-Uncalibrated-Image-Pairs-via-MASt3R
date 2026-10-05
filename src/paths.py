"""
Project paths + import bootstrap for the official MASt3R repository.

WHY this file exists:
The official MASt3R repo is not a pip-installable package. It expects to be run
from its own root directory, and it reaches its submodules through two shim
modules that mutate sys.path at import time:

    mast3r/utils/path_to_dust3r.py  -> inserts  <repo>/dust3r
    dust3r/utils/path_to_croco.py   -> inserts  <repo>/dust3r/croco

Those shims resolve their targets *relative to their own file location*, so all
we have to do from the outside is put the repo root on sys.path. Importing
`mast3r.model` then transitively fixes up dust3r and croco for us. We do NOT
re-implement that path juggling ourselves -- doing so would risk shadowing the
repo's own modules with a stale copy.
"""
from __future__ import annotations

import sys
from pathlib import Path

# <project root>/src/paths.py -> parents[1] is the project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

MAST3R_REPO: Path = PROJECT_ROOT / "mast3r"
DUST3R_REPO: Path = MAST3R_REPO / "dust3r"
CROCO_REPO: Path = DUST3R_REPO / "croco"

CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"

# The one and only publicly released MASt3R ViT-L checkpoint, and it is the
# *metric* one (see README model table). Verified in Phase 1 against the repo.
METRIC_CHECKPOINT_NAME: str = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
METRIC_CHECKPOINT: Path = CHECKPOINT_DIR / METRIC_CHECKPOINT_NAME
METRIC_CHECKPOINT_URL: str = (
    "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
    + METRIC_CHECKPOINT_NAME
)


def bootstrap_mast3r_imports() -> Path:
    """Put the MASt3R repo root on sys.path so `import mast3r.model` works.

    Returns the repo root. Raises with an actionable message if the clone or
    its submodules are missing, rather than failing later with a confusing
    ModuleNotFoundError deep inside croco.
    """
    if not (MAST3R_REPO / "mast3r" / "model.py").is_file():
        raise FileNotFoundError(
            f"MASt3R repo not found at {MAST3R_REPO}. Run:\n"
            f"  git clone --recursive https://github.com/naver/mast3r.git {MAST3R_REPO}"
        )
    if not (DUST3R_REPO / "dust3r" / "model.py").is_file():
        raise FileNotFoundError(
            f"dust3r submodule missing at {DUST3R_REPO}. Run:\n"
            f"  git -C {MAST3R_REPO} submodule update --init --recursive"
        )
    if not (CROCO_REPO / "models").is_dir():
        raise FileNotFoundError(
            f"croco submodule missing at {CROCO_REPO}. Run:\n"
            f"  git -C {MAST3R_REPO} submodule update --init --recursive"
        )

    repo_root = str(MAST3R_REPO)
    if repo_root not in sys.path:
        # append, not insert(0): we don't want the repo shadowing our own `src`
        sys.path.append(repo_root)
    return MAST3R_REPO
