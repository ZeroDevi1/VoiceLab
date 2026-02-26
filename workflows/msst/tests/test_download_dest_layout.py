from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../workflows/msst/tests -> .../VoiceLab
    return Path(__file__).resolve().parents[3]


def _import_tools() -> None:
    tools = _repo_root() / "workflows" / "msst" / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


def test_dest_layout_runtime_keeps_pretrain_prefix() -> None:
    _import_tools()
    from msst_download_models import _resolve_dest_path

    dest = _resolve_dest_path(
        dest_root=Path("/x"),
        rel_dest=Path("pretrain/vocal_models/inst_v1e.ckpt"),
        layout="runtime",
    )
    assert str(dest) == "/x/pretrain/vocal_models/inst_v1e.ckpt"


def test_dest_layout_pretrain_strips_pretrain_prefix() -> None:
    _import_tools()
    from msst_download_models import _resolve_dest_path

    dest = _resolve_dest_path(
        dest_root=Path("/x/pretrain"),
        rel_dest=Path("pretrain/vocal_models/inst_v1e.ckpt"),
        layout="pretrain",
    )
    assert str(dest) == "/x/pretrain/vocal_models/inst_v1e.ckpt"
