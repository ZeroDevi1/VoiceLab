from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from voicelab_bootstrap import voicelab_root


def _is_wsl_mnt_path(p: Path) -> bool:
    # Heuristic: WSL mounts Windows drives under /mnt/<drive>/
    try:
        return p.resolve().as_posix().startswith("/mnt/")
    except Exception:
        return p.as_posix().startswith("/mnt/")


def _ensure_rsync() -> None:
    if shutil.which("rsync") is None:
        raise SystemExit(
            "rsync not found. Install it first:\n"
            "  sudo apt-get update && sudo apt-get install -y rsync"
        )


def stage_dataset(src: Path, dst: Path, *, force: bool) -> Path:
    """
    Copy a dataset from Windows mount (/mnt/c/...) into WSL native ext4 to avoid 9P I/O bottlenecks.
    Uses rsync for speed and resumability.
    """
    _ensure_rsync()
    if not src.exists():
        raise SystemExit(f"Dataset source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Trailing slash copies "contents of src dir" into dst.
    # Example: rsync src/ dst/
    cmd = [
        "rsync",
        "-a",
        "--info=progress2",
    ]
    if force:
        cmd += ["--delete"]
    cmd += [str(src) + "/", str(dst) + "/"]

    print("[rvc] staging dataset (this can take a while on first run):")
    print("[rvc]   src:", src)
    print("[rvc]   dst:", dst)
    subprocess.run(cmd, check=True)
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
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not _is_wsl_mnt_path(src):
        print(f"[rvc] NOTE: src does not look like a /mnt/<drive>/ path: {src}")

    stage_dataset(src, dst, force=bool(args.force))
    print("")
    print("[rvc] Next (recommended):")
    print(f"  uv run python tools/rvc_train.py --dataset-dir {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

