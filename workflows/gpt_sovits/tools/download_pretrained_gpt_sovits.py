from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from voicelab_bootstrap import (
    shared_assets_root,
    shared_g2pw_root,
    shared_pretrained_root,
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _download_with_fallback(urls: list[str], dest: Path) -> None:
    last_error: Exception | None = None
    for url in urls:
        for attempt in range(1, 4):
            try:
                print(f"[gpt_sovits] download: {url} (attempt {attempt}/3)")
                _download(url, dest)
                return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(min(10, attempt * 2))
        print(f"[gpt_sovits] WARN: failed url -> {url}")
    if last_error is not None:
        raise last_error


def _extract(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _pretrained_urls(hf_base: str) -> tuple[list[str], list[str]]:
    base = hf_base.rstrip("/")
    root = f"{base}/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
    return (
        [
            f"{root}/pretrained_models.zip",
            "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip",
        ],
        [
            f"{root}/G2PWModel.zip",
            "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip",
            "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip",
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download common GPT-SoVITS pretrained assets into VoiceLab shared cache."
    )
    ap.add_argument(
        "--hf-base",
        default="https://hf-mirror.com",
        help="HuggingFace base URL (e.g. https://hf-mirror.com or https://huggingface.co).",
    )
    ap.add_argument(
        "--dest-root",
        default=None,
        help="Shared cache root for GPT-SoVITS assets (default: VOICELAB_ASSETS_DIR/gpt_sovits).",
    )
    ap.add_argument(
        "--force", action="store_true", help="Re-download and re-extract assets."
    )
    args = ap.parse_args()

    dest_root = (
        Path(args.dest_root).expanduser().resolve()
        if args.dest_root
        else shared_assets_root()
    )
    pretrained_root = (
        shared_pretrained_root()
        if not args.dest_root
        else dest_root / "GPT_SoVITS" / "pretrained_models"
    )
    g2pw_root = (
        shared_g2pw_root()
        if not args.dest_root
        else dest_root / "GPT_SoVITS" / "text" / "G2PWModel"
    )

    pretrained_urls, g2pw_urls = _pretrained_urls(str(args.hf_base))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        pretrained_zip = tmp / "pretrained_models.zip"
        if bool(args.force) or not pretrained_root.exists():
            _download_with_fallback(pretrained_urls, pretrained_zip)
            if bool(args.force) and pretrained_root.exists():
                shutil.rmtree(pretrained_root, ignore_errors=True)
            _extract(pretrained_zip, dest_root / "GPT_SoVITS")
        else:
            print(f"[gpt_sovits] skip pretrained_models: {pretrained_root}")

        g2pw_zip = tmp / "G2PWModel.zip"
        if bool(args.force) or not g2pw_root.exists():
            _download_with_fallback(g2pw_urls, g2pw_zip)
            if bool(args.force) and g2pw_root.exists():
                shutil.rmtree(g2pw_root, ignore_errors=True)
            _extract(g2pw_zip, dest_root / "GPT_SoVITS" / "text")
            legacy = dest_root / "GPT_SoVITS" / "text" / "G2PWModel_1.1"
            if not g2pw_root.exists() and legacy.exists():
                legacy.rename(g2pw_root)
        else:
            print(f"[gpt_sovits] skip G2PWModel: {g2pw_root}")

    print(f"[gpt_sovits] assets ready: {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
