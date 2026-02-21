from __future__ import annotations

import os
import sys
from pathlib import Path


def workflow_root() -> Path:
    # .../VoiceLab/workflows/cosyvoice/tools/voicelab_bootstrap.py -> .../VoiceLab/workflows/cosyvoice
    return Path(__file__).resolve().parents[1]


def voicelab_root() -> Path:
    # .../VoiceLab/workflows/cosyvoice -> .../VoiceLab
    return workflow_root().parents[2]


def cosyvoice_vendor_root() -> Path:
    env = os.environ.get("COSYVOICE_VENDOR_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return voicelab_root() / "vendor" / "CosyVoice"


def matcha_vendor_root() -> Path:
    return cosyvoice_vendor_root() / "third_party" / "Matcha-TTS"


def ensure_sys_path() -> dict[str, Path]:
    """
    Make workflow + vendor CosyVoice importable without modifying vendor.
    Returns useful resolved paths.
    """
    wf = workflow_root()
    vendor = cosyvoice_vendor_root()
    matcha = matcha_vendor_root()

    # Allow importing voicelab_cosyvoice.* (custom patches referenced by YAML)
    if str(wf) not in sys.path:
        sys.path.insert(0, str(wf))

    # Allow importing cosyvoice.* from vendor
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    # Allow importing matcha.* required by CosyVoice3 configs
    if matcha.exists() and str(matcha) not in sys.path:
        sys.path.insert(0, str(matcha))

    return {"workflow": wf, "vendor": vendor, "matcha": matcha}

