from __future__ import annotations

import argparse
from pathlib import Path

from voicelab_bootstrap import voicelab_root
from voicelab.dataset_stage import ensure_list_present as _ensure_list_present
from voicelab.dataset_stage import stage_dataset_rsync


def _is_wsl_mnt_path(p: Path) -> bool:
    # Heuristic: WSL mounts Windows drives under /mnt/<drive>/
    try:
        return p.resolve().as_posix().startswith("/mnt/")
    except Exception:
        return p.as_posix().startswith("/mnt/")

def stage_dataset(
    src: Path,
    dst: Path,
    *,
    force: bool,
    copy_list: bool = True,
    annotation_dir: Path = Path("/mnt/c/AIGC/数据集/标注文件"),
) -> Path:
    """
    Copy a dataset from Windows mount (/mnt/c/...) into WSL native ext4 to avoid 9P I/O bottlenecks.
    Uses rsync for speed and resumability.
    """
    stage_dataset_rsync(src, dst, force=force)
    if copy_list:
        copied = _ensure_list_present(src, dst, annotation_dir=annotation_dir)
        if copied is not None:
            print(f"[rvc] list available: {copied}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy dataset into WSL native path for faster I/O.")
    ap.add_argument("--src", default="/mnt/c/AIGC/数据集/XingTong")
    ap.add_argument(
        "--dst",
        default=str(voicelab_root() / "datasets" / "XingTong"),
        help="WSL-native destination directory (default: VoiceLab/datasets/XingTong).",
    )
    ap.add_argument("--force", action="store_true", help="Make destination match source (rsync --delete).")
    ap.add_argument(
        "--annotation-dir",
        default="/mnt/c/AIGC/数据集/标注文件",
        help="Fallback directory containing centralized *.list annotation files.",
    )
    ap.add_argument(
        "--no-copy-list",
        action="store_true",
        help="Do not copy/ensure same-name *.list into destination dataset directory.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not _is_wsl_mnt_path(src):
        print(f"[rvc] NOTE: src does not look like a /mnt/<drive>/ path: {src}")

    stage_dataset(
        src,
        dst,
        force=bool(args.force),
        copy_list=not bool(args.no_copy_list),
        annotation_dir=Path(args.annotation_dir),
    )
    print("")
    print("[rvc] Next (recommended):")
    print(f"  uv run python tools/rvc_train.py --dataset-dir {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
