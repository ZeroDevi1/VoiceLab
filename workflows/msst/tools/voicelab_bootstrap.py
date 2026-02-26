from __future__ import annotations

import os
import sys
from pathlib import Path


def workflow_root() -> Path:
    # .../VoiceLab/workflows/msst/tools/voicelab_bootstrap.py -> .../VoiceLab/workflows/msst
    return Path(__file__).resolve().parents[1]


def voicelab_root() -> Path:
    # .../VoiceLab/workflows/msst -> .../VoiceLab
    return workflow_root().parents[1]


def msst_vendor_root() -> Path:
    """
    Resolve upstream MSST-WebUI repo root under vendor/.

    This workspace conventionally uses:
      vendor/MSST-WebUI
    """
    env = os.environ.get("MSST_VENDOR_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (voicelab_root() / "vendor" / "MSST-WebUI").resolve()


def runtime_root() -> Path:
    env = os.environ.get("MSST_RUNTIME_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return workflow_root() / "runtime"


def assets_src_root() -> Path:
    """
    Where to copy large MSST models (ckpt/th) from.

    Default matches the user's existing Windows-side install visible in WSL:
      /mnt/c/AIGC/MSST-WebUI/pretrain
    """
    env = os.environ.get("MSST_ASSETS_SRC_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path("/mnt/c/AIGC/MSST-WebUI/pretrain").resolve()


def ensure_runtime_pythonpath() -> dict[str, Path]:
    """
    Make runtime root importable (so `import inference`, `import modules`, `import utils` work).

    We import from `workflows/msst/runtime/` instead of vendor/ to avoid modifying upstream files.
    """
    rt = runtime_root()
    if str(rt) not in sys.path:
        sys.path.insert(0, str(rt))
    return {"runtime": rt}

