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


def voicelab_assets_root() -> Path:
    env = os.environ.get("VOICELAB_ASSETS_DIR")
    base = (env or str(voicelab_root() / ".cache" / "voicelab" / "assets")).strip()
    return Path(base).expanduser().resolve()


def shared_assets_root() -> Path:
    env = os.environ.get("GPT_SOVITS_ASSETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (voicelab_assets_root() / "gpt_sovits").resolve()


def shared_pretrained_root() -> Path:
    return shared_assets_root() / "GPT_SoVITS" / "pretrained_models"


def shared_g2pw_root() -> Path:
    return shared_assets_root() / "GPT_SoVITS" / "text" / "G2PWModel"


def default_pretrained_s1(version: str) -> Path:
    mapping = {
        "v1": "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "v2": "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "v3": "s1v3.ckpt",
        "v4": "s1v3.ckpt",
        "v2Pro": "s1v3.ckpt",
        "v2ProPlus": "s1v3.ckpt",
    }
    return (shared_pretrained_root() / mapping.get(version, mapping["v2"])).resolve()


def default_pretrained_s2g(version: str) -> Path:
    mapping = {
        "v1": "s2G488k.pth",
        "v2": "gsv-v2final-pretrained/s2G2333k.pth",
        "v3": "s2Gv3.pth",
        "v4": "gsv-v4-pretrained/s2Gv4.pth",
        "v2Pro": "v2Pro/s2Gv2Pro.pth",
        "v2ProPlus": "v2Pro/s2Gv2ProPlus.pth",
    }
    return (shared_pretrained_root() / mapping.get(version, mapping["v2"])).resolve()


def default_pretrained_s2d(version: str) -> Path:
    s2g = default_pretrained_s2g(version)
    return s2g.with_name(s2g.name.replace("s2G", "s2D", 1)).resolve()


def data_root() -> Path:
    return workflow_root() / "data"


def runtime_root() -> Path:
    return workflow_root() / "runtime"
