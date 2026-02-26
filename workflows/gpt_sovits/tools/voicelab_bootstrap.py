from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_root_pythonpath() -> None:
    # Make `import voicelab.*` work when executing tools from `workflows/gpt_sovits/`.
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root_pythonpath()


def workflow_root() -> Path:
    # .../VoiceLab/workflows/gpt_sovits/tools/voicelab_bootstrap.py -> .../VoiceLab/workflows/gpt_sovits
    return Path(__file__).resolve().parents[1]


def voicelab_root() -> Path:
    # .../VoiceLab/workflows/gpt_sovits -> .../VoiceLab
    return workflow_root().parents[1]


def gpt_sovits_vendor_root() -> Path:
    env = os.environ.get("GPT_SOVITS_VENDOR_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (voicelab_root() / "vendor" / "GPT-SoVITS").resolve()


def data_root() -> Path:
    return workflow_root() / "data"


def runtime_root() -> Path:
    return workflow_root() / "runtime"

