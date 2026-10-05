"""
PHASE 1 -- SETUP AND VERIFICATION
=================================

Objective (simple):
    Before we ask MASt3R to measure anything, prove that the machine, the
    library, and the downloaded brain of the model are all real and working.

Objective (technical):
    Verify the runtime (Python/PyTorch/compute device), verify that the
    official MASt3R repository and its dust3r + croco submodules import
    cleanly, and verify that the *metric* checkpoint deserialises into an
    AsymmetricMASt3R instance whose head configuration matches the config
    recorded in the checkpoint itself.

Nothing here runs inference -- that is Phase 2. This phase only answers:
"is the ground solid?"

Run:
    .venv/bin/python phases/phase1_setup.py
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402


# --------------------------------------------------------------------------
# tiny check harness: every check records a (name, ok, detail) so the script
# can end with an unambiguous PASS/FAIL summary instead of a wall of prints.
# --------------------------------------------------------------------------
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# --------------------------------------------------------------------------
# 1. Runtime
# --------------------------------------------------------------------------
def check_runtime() -> None:
    section("1. RUNTIME")
    print(f"  platform : {platform.platform()}")
    print(f"  python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  cpu cores: {os.cpu_count()}")
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        print(f"  ram      : {mem_gb:.1f} GB")
    except (ValueError, OSError):
        pass
    # dust3r/mast3r use f-strings with '=' and match-free syntax; 3.10 is the floor.
    check("python >= 3.10", sys.version_info >= (3, 10), f"{sys.version_info.major}.{sys.version_info.minor}")


# --------------------------------------------------------------------------
# 2. PyTorch + compute device
# --------------------------------------------------------------------------
def check_torch() -> "object":
    section("2. PYTORCH / COMPUTE DEVICE")
    import torch

    print(f"  torch        : {torch.__version__}")
    print(f"  cuda built   : {torch.version.cuda}")
    print(f"  cuda runtime : {torch.cuda.is_available()}")
    print(f"  torch threads: {torch.get_num_threads()}")

    check("torch imports", True, torch.__version__)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  gpu          : {torch.cuda.get_device_name(0)}")
        print(f"  gpu memory   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        check("CUDA GPU available", True, torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        # NOT a hard failure: MASt3R inference is device-agnostic. It is recorded
        # loudly because it changes runtime per pair from seconds to minutes.
        check("CUDA GPU available", False, "no CUDA device -- falling back to CPU (slow but correct)")

    # A real tensor op, so we catch a broken BLAS/wheel rather than trusting the import.
    x = torch.randn(256, 256, device=device)
    y = (x @ x).sum().item()
    check(f"tensor matmul on '{device}'", y == y, f"256x256 matmul ok, sum={y:.3f}")
    return torch


# --------------------------------------------------------------------------
# 3. Repository + submodules
# --------------------------------------------------------------------------
def check_repo() -> None:
    section("3. MAST3R REPOSITORY")
    repo = paths.bootstrap_mast3r_imports()
    check("repo + submodules present", True, str(repo))

    for name, path in (("mast3r", paths.MAST3R_REPO),
                       ("dust3r", paths.DUST3R_REPO),
                       ("croco", paths.CROCO_REPO)):
        rev = subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%h %ad %s",
                              "--date=short"], capture_output=True, text=True).stdout.strip()
        print(f"  {name:<7}: {rev}")

    # The import order matters and is the repo's own: importing mast3r.model
    # triggers path_to_dust3r, which triggers path_to_croco.
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from models.croco import CroCoNet  # croco lives at top level once the shim ran

    check("import mast3r.model.AsymmetricMASt3R", callable(AsymmetricMASt3R))
    check("import mast3r.fast_nn.fast_reciprocal_NNs", callable(fast_reciprocal_NNs))
    check("import dust3r.inference.inference", callable(inference))
    check("import dust3r.utils.image.load_images", callable(load_images))
    check("import croco (models.croco.CroCoNet)", callable(CroCoNet))
    check("AsymmetricMASt3R subclasses CroCoNet", issubclass(AsymmetricMASt3R, CroCoNet))


# --------------------------------------------------------------------------
# 4. The metric checkpoint
# --------------------------------------------------------------------------
def check_checkpoint(torch) -> None:
    section("4. METRIC CHECKPOINT")
    ckpt_path = paths.METRIC_CHECKPOINT
    if not check("checkpoint file exists", ckpt_path.is_file(), str(ckpt_path)):
        print(f"\n  Download it with:\n    wget {paths.METRIC_CHECKPOINT_URL} -P {paths.CHECKPOINT_DIR}")
        return

    size_gb = ckpt_path.stat().st_size / 1024**3
    check("checkpoint size plausible (>2 GB)", size_gb > 2.0, f"{size_gb:.2f} GB")
    check("checkpoint filename says 'metric'", "metric" in ckpt_path.name, ckpt_path.name)

    # weights_only=False is required: the checkpoint stores an argparse.Namespace
    # under 'args' (the full training config), not just a state_dict. This mirrors
    # exactly what mast3r/model.py::load_model does.
    print("\n  loading checkpoint (this reads ~2.6 GB from disk)...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check("checkpoint deserialises", isinstance(ckpt, dict), f"top-level keys: {sorted(ckpt.keys())}")
    check("has 'model' state_dict", "model" in ckpt, f"{len(ckpt.get('model', {}))} tensors")
    check("has 'args' training config", "args" in ckpt)

    # --- the evidence that this really is the METRIC checkpoint ---------------
    args = ckpt["args"]
    print("\n  --- model architecture string recorded IN the checkpoint ---")
    print(f"  {args.model}")
    train_crit = getattr(args, "train_criterion", None)
    print("\n  --- training criterion recorded IN the checkpoint ---")
    print(f"  {train_crit}")

    # norm_mode='?avg_dis': the leading '?' tells mast3r/losses.py::Regr3D to SKIP
    # scale-normalisation for samples flagged is_metric_scale, i.e. the network was
    # forced to regress true real-world scale on the metric datasets. That flag is
    # the mechanism by which this checkpoint is "metric" at all.
    check("trained with metric-scale loss (norm_mode='?avg_dis')",
          train_crit is not None and "?avg_dis" in train_crit,
          "'?' = do NOT normalise scale on metric datasets")
    check("output_mode includes descriptors ('pts3d+desc24')",
          "desc24" in args.model, "24-D dense descriptor per pixel")
    check("two_confs=True (separate geometry & descriptor confidence)",
          "two_confs=True" in args.model.replace(" ", ""))

    # --- instantiate the model from the checkpoint ---------------------------
    from mast3r.model import AsymmetricMASt3R
    print("\n  instantiating AsymmetricMASt3R from checkpoint...")
    model = AsymmetricMASt3R.from_pretrained(str(ckpt_path))
    model.eval()

    from mast3r.model import AsymmetricMASt3R as _A
    check("model is AsymmetricMASt3R", isinstance(model, _A), type(model).__name__)

    n_params = sum(p.numel() for p in model.parameters())
    check("parameter count ~0.6-1.0 B", 0.4e9 < n_params < 1.2e9, f"{n_params/1e6:.1f} M params")

    print("\n  --- head configuration as instantiated ---")
    for attr in ("output_mode", "head_type", "depth_mode", "conf_mode", "desc_mode",
                 "desc_conf_mode", "two_confs"):
        print(f"  {attr:<15}: {getattr(model, attr, '<absent>')}")
    print(f"  {'patch_size':<15}: {model.patch_embed.patch_size}")

    # depth_mode 'exp' -> reg_dense_depth returns  (xyz/|xyz|) * expm1(|xyz|):
    # an unbounded positive radial distance. Nothing in the head clamps or
    # normalises it, so the scale that comes out is whatever the weights encode.
    check("depth_mode is unbounded 'exp'",
          tuple(model.depth_mode)[0] == "exp" and model.depth_mode[1] == float("-inf"),
          f"{model.depth_mode} -> pts3d = unit(xyz)*expm1(|xyz|), no rescaling")
    check("conf_mode is ('exp', 1, inf)", tuple(model.conf_mode) == ("exp", 1, float("inf")),
          f"{model.conf_mode} -> conf = 1 + exp(x) in (1, inf)")
    check("desc_conf_mode is ('exp', 0, inf)",
          tuple(model.desc_conf_mode) == ("exp", 0, float("inf")),
          f"{model.desc_conf_mode} -> desc_conf = exp(x) in (0, inf)")

    del ckpt, model


def main() -> int:
    print("=" * 74)
    print("PHASE 1 -- MASt3R METRIC DISTANCE PROJECT :: SETUP VERIFICATION")
    print("=" * 74)
    check_runtime()
    torch = check_torch()
    check_repo()
    check_checkpoint(torch)

    section("SUMMARY")
    failed = [c for c in CHECKS if not c[1]]
    print(f"  {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  --  {detail}")
    # The CUDA check is informational on a CPU-only box; don't fail the phase on it.
    blocking = [c for c in failed if "CUDA GPU available" not in c[0]]
    print(f"\n  PHASE 1: {'OK' if not blocking else 'BLOCKED'}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
