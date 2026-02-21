from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from voicelab_bootstrap import assets_src_root, rvc_vendor_root, runtime_root


def _rm_rf(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path)


def _symlink(src: Path, dst: Path, *, force: bool) -> None:
    if force:
        _rm_rf(dst)
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(src), str(dst))


def _copytree(src: Path, dst: Path, *, force: bool) -> None:
    if force:
        _rm_rf(dst)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def init_runtime(*, force: bool, assets_src: Path | None) -> Path:
    vendor = rvc_vendor_root()
    if not vendor.exists():
        raise SystemExit(
            f"Upstream RVC repo not found: {vendor}\n"
            "Expected `vendor/Retrieval-based-Voice-Conversion-WebUI`.\n"
            "Run: uv run -m voicelab vendor sync"
        )

    rt = runtime_root()
    _ensure_dir(rt)

    # 1) Symlink code dirs from vendor.
    _symlink(vendor / "infer", rt / "infer", force=force)
    _symlink(vendor / "i18n", rt / "i18n", force=force)
    _symlink(vendor / "tools", rt / "tools", force=force)

    # 2) Copy configs so runtime can freely write configs/inuse without touching vendor.
    _copytree(vendor / "configs", rt / "configs", force=force)

    # 3) Prepare runtime asset dirs.
    _ensure_dir(rt / "assets" / "weights")  # training output goes here
    _ensure_dir(rt / "indices")  # feature index output goes here
    _ensure_dir(rt / "logs")

    # 4) Link "mute" dataset from vendor (used by upstream filelist logic).
    vendor_mute = vendor / "logs" / "mute"
    if vendor_mute.exists():
        _symlink(vendor_mute, rt / "logs" / "mute", force=force)

    # 5) Link large upstream models from the user's existing RVC install.
    src_assets = assets_src or assets_src_root()
    if not src_assets.exists():
        raise SystemExit(
            f"Assets source dir not found: {src_assets}\n"
            "Set RVC_ASSETS_SRC_DIR or pass --assets-src."
        )

    # hubert / rmvpe are required for feature/f0 extraction.
    _ensure_dir(rt / "assets" / "hubert")
    _ensure_dir(rt / "assets" / "rmvpe")
    _symlink(src_assets / "hubert" / "hubert_base.pt", rt / "assets" / "hubert" / "hubert_base.pt", force=force)
    _symlink(src_assets / "rmvpe" / "rmvpe.pt", rt / "assets" / "rmvpe" / "rmvpe.pt", force=force)

    # pretrained weights for v1/v2 (we use v2 in this workflow).
    _symlink(src_assets / "pretrained", rt / "assets" / "pretrained", force=force)
    _symlink(src_assets / "pretrained_v2", rt / "assets" / "pretrained_v2", force=force)

    return rt


def main() -> int:
    ap = argparse.ArgumentParser(description="Initialize workflows/rvc/runtime without modifying vendor.")
    ap.add_argument("--force", action="store_true", help="Recreate runtime links/copies.")
    ap.add_argument(
        "--assets-src",
        type=Path,
        default=None,
        help="Path to an existing RVC assets directory (default: /mnt/c/AIGC/RVC20240604Nvidia/assets).",
    )
    args = ap.parse_args()

    rt = init_runtime(force=bool(args.force), assets_src=args.assets_src)
    print(f"[rvc] runtime ready: {rt}")
    print("")
    print("[rvc] Next steps:")
    print("  - Train (50 epoch): uv run python tools/rvc_train.py --total-epoch 50")
    print("  - Train index:      uv run python tools/rvc_train_index.py")
    print("  - Infer one file:   uv run python tools/rvc_infer_one.py --pitch 12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

