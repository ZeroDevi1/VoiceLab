from __future__ import annotations

import os
import sys
from pathlib import Path


def workflow_root() -> Path:
    # .../VoiceLab/workflows/rvc/tools/voicelab_bootstrap.py -> .../VoiceLab/workflows/rvc
    return Path(__file__).resolve().parents[1]


def voicelab_root() -> Path:
    # .../VoiceLab/workflows/rvc -> .../VoiceLab
    return workflow_root().parents[1]


def rvc_vendor_root() -> Path:
    """
    Resolve the upstream RVC repo directory without requiring a fixed vendor name.

    This workspace currently uses:
      vendor/Retrieval-based-Voice-Conversion-WebUI

    but `voicelab vendor sync` may also create:
      vendor/RVC
    """
    env = os.environ.get("RVC_VENDOR_DIR")
    if env:
        return Path(env).expanduser().resolve()

    root = voicelab_root() / "vendor"
    candidates = [
        root / "Retrieval-based-Voice-Conversion-WebUI",
        root / "RVC",
        root / "Retrieval-based-Voice-Conversion-WebUI" / ".",  # no-op, but keeps ordering explicit
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Fall back to the primary expected path; callers will error with a better message.
    return (root / "Retrieval-based-Voice-Conversion-WebUI").resolve()


def runtime_root() -> Path:
    return workflow_root() / "runtime"


def assets_src_root() -> Path:
    """
    Where to link large RVC assets (hubert/rmvpe/pretrained) from.

    Default matches the user's existing Windows-side install that is visible in WSL:
      /mnt/c/AIGC/RVC20240604Nvidia/assets
    """
    env = os.environ.get("RVC_ASSETS_SRC_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path("/mnt/c/AIGC/RVC20240604Nvidia/assets").resolve()


def ensure_runtime_pythonpath() -> dict[str, Path]:
    """
    Make runtime root importable (so `import configs`, `import infer.*` work).

    We intentionally import from `workflows/rvc/runtime/` instead of `vendor/`
    to avoid writing into the upstream tree (e.g. `configs/inuse/*`).
    """
    rt = runtime_root()
    if str(rt) not in sys.path:
        sys.path.insert(0, str(rt))
    return {"runtime": rt}

