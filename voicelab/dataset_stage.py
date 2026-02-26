from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from voicelab.list_annotations import find_same_name_list


def _ensure_rsync() -> None:
    if shutil.which("rsync") is None:
        raise SystemExit(
            "rsync not found. Install it first:\n"
            "  sudo apt-get update && sudo apt-get install -y rsync"
        )


def stage_dataset_rsync(src: Path, dst: Path, *, force: bool) -> Path:
    """
    Copy a dataset dir into a faster local filesystem (e.g. WSL ext4) using rsync.
    """
    _ensure_rsync()
    if not src.exists():
        raise SystemExit(f"Dataset source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["rsync", "-a", "--info=progress2"]
    if force:
        cmd += ["--delete"]
    cmd += [str(src) + "/", str(dst) + "/"]

    print("[stage] rsync dataset:")
    print("[stage]   src:", src)
    print("[stage]   dst:", dst)
    subprocess.run(cmd, check=True)
    return dst


def ensure_list_present(
    src: Path,
    dst: Path,
    *,
    annotation_dir: Path = Path("/mnt/c/AIGC/数据集/标注文件"),
) -> Path | None:
    """
    Ensure a "same-name" *.list file exists under dst.

    Resolution order:
    1) if dst already has same-name list -> keep it
    2) else if src has same-name list -> copy it into dst
    3) else if annotation_dir has <src.name>.list or <src.name.lower()>.list -> copy into dst
    """
    existing = find_same_name_list(dst)
    if existing is not None:
        return existing

    src_list = find_same_name_list(src)
    if src_list is not None and src_list.exists():
        dst_list = dst / src_list.name
        dst_list.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_list, dst_list)
        return dst_list

    name = src.name
    candidates = [
        annotation_dir / f"{name}.list",
        annotation_dir / f"{name.lower()}.list",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            dst_list = dst / c.name
            dst_list.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c, dst_list)
            return dst_list

    return None

