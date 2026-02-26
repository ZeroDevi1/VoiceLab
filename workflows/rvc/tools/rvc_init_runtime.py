from __future__ import annotations

import argparse
import os
import shutil
import time
import urllib.error
import urllib.request
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


def _runtime_has_core_assets(rt: Path) -> bool:
    """
    If runtime already has the core assets in-place, callers shouldn't need an assets_src.

    This is important for shared-cache setups where bootstrap runs init once (with --assets-src),
    and subsequent train/infer commands call init_runtime() again without knowing the assets_src.
    """
    hubert = rt / "assets" / "hubert" / "hubert_base.pt"
    rmvpe = rt / "assets" / "rmvpe" / "rmvpe.pt"
    return hubert.exists() and rmvpe.exists()


def _http_download(*, url: str, dest: Path, force: bool, timeout_s: int = 60, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    if force and partial.exists():
        partial.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            resume_from = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "voicelab-rvc-init/0.1"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resume_from > 0 and getattr(resp, "status", None) == 200:
                    resume_from = 0
                mode = "ab" if resume_from > 0 else "wb"
                with partial.open(mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(partial, dest)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt >= retries:
                raise
            sleep_s = min(10, 1 + attempt * 2)
            print(f"[rvc] WARN: download failed (attempt {attempt}/{retries}): {e}; retry in {sleep_s}s")
            time.sleep(sleep_s)


def _rvc_asset_urls(*, hf_base: str) -> dict[str, str]:
    hf = hf_base.rstrip("/")
    base = f"{hf}/lj1995/VoiceConversionWebUI/resolve/main"
    return {
        "hubert_base.pt": f"{base}/hubert_base.pt",
        "rmvpe.pt": f"{base}/rmvpe.pt",
        "f0G48k.pth": f"{base}/pretrained_v2/f0G48k.pth",
        "f0D48k.pth": f"{base}/pretrained_v2/f0D48k.pth",
    }


def _missing_required_assets(src_assets: Path) -> list[Path]:
    # Core assets required for feature/f0 extraction.
    required = [
        src_assets / "hubert" / "hubert_base.pt",
        src_assets / "rmvpe" / "rmvpe.pt",
    ]
    return [p for p in required if not p.exists()]


def _missing_optional_assets(src_assets: Path) -> list[Path]:
    # Pretrained weights are optional for training (upstream can train from scratch).
    optional = [
        src_assets / "pretrained_v2" / "f0G48k.pth",
        src_assets / "pretrained_v2" / "f0D48k.pth",
        src_assets / "pretrained",
        src_assets / "pretrained_v2",
    ]
    return [p for p in optional if not p.exists()]


def _download_missing_assets(*, src_assets: Path, hf_base: str, force: bool) -> None:
    urls = _rvc_asset_urls(hf_base=hf_base)
    # Ensure folder structure matches what init_runtime expects.
    _ensure_dir(src_assets / "hubert")
    _ensure_dir(src_assets / "rmvpe")
    _ensure_dir(src_assets / "pretrained_v2")

    mapping = [
        (urls["hubert_base.pt"], src_assets / "hubert" / "hubert_base.pt"),
        (urls["rmvpe.pt"], src_assets / "rmvpe" / "rmvpe.pt"),
        (urls["f0G48k.pth"], src_assets / "pretrained_v2" / "f0G48k.pth"),
        (urls["f0D48k.pth"], src_assets / "pretrained_v2" / "f0D48k.pth"),
    ]
    for url, dest in mapping:
        print(f"[rvc] download: {dest.name}", flush=True)
        _http_download(url=url, dest=dest, force=force)


def init_runtime(*, force: bool, assets_src: Path | None, download_missing: bool, hf_base: str) -> Path:
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
    #
    # If runtime already has core assets, don't require assets_src (shared-cache bootstrap flow).
    if not force and _runtime_has_core_assets(rt):
        return rt

    src_assets = assets_src or assets_src_root()
    if not src_assets.exists():
        if not download_missing:
            raise SystemExit(
                f"Assets source dir not found: {src_assets}\n"
                "Set RVC_ASSETS_SRC_DIR or pass --assets-src.\n"
                "Or run with --download-missing to fetch assets via HuggingFace mirror."
            )
        _ensure_dir(src_assets)

    missing_required = _missing_required_assets(src_assets)
    if missing_required and download_missing:
        print(f"[rvc] Missing {len(missing_required)} core asset(s); downloading into {src_assets} ...", flush=True)
        _download_missing_assets(src_assets=src_assets, hf_base=hf_base, force=False)
        missing_required = _missing_required_assets(src_assets)
    if missing_required:
        raise SystemExit(
            "[rvc] Missing required assets:\n"
            + "\n".join(f"  - {p}" for p in missing_required)
            + "\nPass --assets-src to an existing assets dir, or use --download-missing."
        )

    # hubert / rmvpe are required for feature/f0 extraction.
    _ensure_dir(rt / "assets" / "hubert")
    _ensure_dir(rt / "assets" / "rmvpe")
    _symlink(src_assets / "hubert" / "hubert_base.pt", rt / "assets" / "hubert" / "hubert_base.pt", force=force)
    _symlink(src_assets / "rmvpe" / "rmvpe.pt", rt / "assets" / "rmvpe" / "rmvpe.pt", force=force)

    # pretrained weights for v1/v2 (optional).
    if (src_assets / "pretrained").exists():
        _symlink(src_assets / "pretrained", rt / "assets" / "pretrained", force=force)
    if (src_assets / "pretrained_v2").exists():
        _symlink(src_assets / "pretrained_v2", rt / "assets" / "pretrained_v2", force=force)
    else:
        opt = _missing_optional_assets(src_assets)
        if opt:
            print("[rvc] NOTE: pretrained weights not found; training can still run from scratch.", flush=True)

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
    ap.add_argument("--download-missing", action="store_true", help="Download missing assets via HuggingFace mirror.")
    ap.add_argument(
        "--hf-base",
        default="https://hf-mirror.com",
        help="HuggingFace base URL (default: https://hf-mirror.com).",
    )
    args = ap.parse_args()

    rt = init_runtime(
        force=bool(args.force),
        assets_src=args.assets_src,
        download_missing=bool(args.download_missing),
        hf_base=str(args.hf_base),
    )
    print(f"[rvc] runtime ready: {rt}")
    print("")
    print("[rvc] Next steps:")
    print("  - Train (50 epoch): uv run python tools/rvc_train.py --total-epoch 50")
    print("  - Train index:      uv run python tools/rvc_train_index.py")
    print("  - Infer one file:   uv run python tools/rvc_infer_one.py --pitch 12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
