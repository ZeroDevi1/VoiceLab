from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from voicelab_bootstrap import assets_src_root, msst_vendor_root, runtime_root


def _rm_rf(path: Path) -> None:
    # Windows 下如果符号链接类型创建错了（例如：把目录当成“文件链接”创建），
    # Path.exists()/stat() 可能会抛 PermissionError；这里要保证 --force 时仍可清理。
    try:
        exists = path.exists()
    except OSError:
        exists = False

    if not exists and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _symlink(src: Path, dst: Path, *, force: bool) -> None:
    if force:
        _rm_rf(dst)
    try:
        dst_exists = dst.exists()
    except OSError:
        dst_exists = False

    if dst_exists or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Windows 上创建符号链接时必须显式告诉它目标是否为目录；
    # 否则会创建成“文件符号链接”，Python 会无法按目录遍历（Path.exists/stat 可能报错）。
    os.symlink(str(src), str(dst), target_is_directory=src.is_dir())


def _copytree(src: Path, dst: Path, *, force: bool) -> None:
    if force:
        _rm_rf(dst)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def _copyfile(src: Path, dst: Path, *, force: bool) -> None:
    if force and dst.exists():
        dst.unlink(missing_ok=True)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_model(src: Path, dst: Path, *, force: bool) -> None:
    if force and dst.exists():
        dst.unlink(missing_ok=True)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _link_or_copy_model(src: Path, dst: Path, *, force: bool) -> None:
    """
    Prefer symlink to avoid duplicating large model files; fall back to copy.
    """
    if force and (dst.exists() or dst.is_symlink()):
        if dst.is_symlink() or dst.is_file():
            dst.unlink(missing_ok=True)
        else:
            shutil.rmtree(dst)
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def init_runtime(
    *, force: bool, assets_src: Path | None, download_missing: bool, hf_base: str
) -> Path:
    vendor = msst_vendor_root()
    if not vendor.exists():
        raise SystemExit(
            f"Upstream MSST-WebUI repo not found: {vendor}\n"
            "Run: uv run -m voicelab vendor sync"
        )

    rt = runtime_root()
    if force and rt.exists():
        _rm_rf(rt)

    _ensure_dir(rt)

    # 1) Symlink code dirs from vendor.
    for d in ("inference", "modules", "utils"):
        _symlink(vendor / d, rt / d, force=force)

    # 2) Copy clean default configs (avoid WebUI-saved mutations in configs/).
    configs_backup = vendor / "configs_backup"
    if not configs_backup.exists():
        raise SystemExit(f"MSST-WebUI configs_backup not found: {configs_backup}")
    _copytree(configs_backup, rt / "configs", force=force)

    # 3) Copy data/model metadata (for download + sha validation).
    # Upstream may keep these under data_backup/ and generate data/ at runtime.
    if (vendor / "data_backup" / "models_info.json").exists():
        _copyfile(
            vendor / "data_backup" / "models_info.json",
            rt / "data" / "models_info.json",
            force=force,
        )
    else:
        _copyfile(
            vendor / "data" / "models_info.json",
            rt / "data" / "models_info.json",
            force=force,
        )

    # 4) Install becruily karaoke config under runtime/configs/vocal_models/.
    # Keep it in runtime so inference always uses a stable config even if vendor changes.
    local_cfg = (
        Path(__file__).resolve().parents[1] / "configs" / "config_karaoke_becruily.yaml"
    )
    _copyfile(
        local_cfg,
        rt / "configs" / "vocal_models" / "config_karaoke_becruily.yaml",
        force=force,
    )

    # 5) Prepare runtime model dirs.
    _ensure_dir(rt / "pretrain" / "vocal_models")
    _ensure_dir(rt / "pretrain" / "single_stem_models")

    # If runtime already has all required model files, do not require an assets_src.
    required = [
        rt / "pretrain" / "vocal_models" / "inst_v1e.ckpt",
        rt / "pretrain" / "vocal_models" / "big_beta5e.ckpt",
        rt
        / "pretrain"
        / "vocal_models"
        / "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        rt
        / "pretrain"
        / "single_stem_models"
        / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        rt
        / "pretrain"
        / "single_stem_models"
        / "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    ]
    if not force and all(p.exists() for p in required):
        return rt

    # 6) Copy models from the user's existing MSST install (WSL-visible Windows path by default).
    src_assets = assets_src or assets_src_root()
    if not src_assets.exists():
        raise SystemExit(
            f"Assets source dir not found: {src_assets}\n"
            "Set MSST_ASSETS_SRC_DIR or pass --assets-src."
        )

    models = [
        (
            src_assets / "vocal_models" / "inst_v1e.ckpt",
            rt / "pretrain" / "vocal_models" / "inst_v1e.ckpt",
        ),
        (
            src_assets / "vocal_models" / "big_beta5e.ckpt",
            rt / "pretrain" / "vocal_models" / "big_beta5e.ckpt",
        ),
        (
            src_assets
            / "vocal_models"
            / "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            rt
            / "pretrain"
            / "vocal_models"
            / "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        ),
        (
            src_assets
            / "single_stem_models"
            / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            rt
            / "pretrain"
            / "single_stem_models"
            / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        ),
        (
            src_assets
            / "single_stem_models"
            / "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            rt
            / "pretrain"
            / "single_stem_models"
            / "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        ),
    ]

    for src, dst in models:
        if src.exists():
            _link_or_copy_model(src, dst, force=force)

    # 7) Download missing models into assets_src if requested (new environment).
    missing = [src for src, dst in models if not dst.exists()]
    if missing and download_missing:
        from msst_download_models import build_model_specs, download_models

        print(
            f"[msst] Missing {len(missing)} model(s); downloading into {src_assets} ...",
            flush=True,
        )
        specs = build_model_specs(hf_base=hf_base)
        download_models(
            specs=specs, force=False, dest_root=src_assets, dest_layout="pretrain"
        )

        # Re-link after download.
        for src, dst in models:
            if src.exists() and not dst.exists():
                _link_or_copy_model(src, dst, force=force)

    # 8) Verify all required model files exist.
    missing_required = [p for p in required if not p.exists()]
    if missing_required:
        raise SystemExit(
            "[msst] Missing required model files:\n"
            + "\n".join(f"  - {p}" for p in missing_required)
            + "\nRun: uv run python tools/msst_download_models.py"
        )

    return rt


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Initialize workflows/msst/runtime without modifying vendor."
    )
    ap.add_argument(
        "--force", action="store_true", help="Recreate runtime links/copies."
    )
    ap.add_argument(
        "--assets-src",
        type=Path,
        default=None,
        help="Path to an existing MSST pretrain directory (default: /mnt/c/AIGC/MSST-WebUI/pretrain).",
    )
    ap.add_argument(
        "--no-download-missing",
        action="store_true",
        help="Disable downloading missing models.",
    )
    ap.add_argument(
        "--hf-base",
        default="https://hf-mirror.com",
        help="HuggingFace base URL (default: https://hf-mirror.com)",
    )
    args = ap.parse_args()

    rt = init_runtime(
        force=bool(args.force),
        assets_src=args.assets_src,
        download_missing=not bool(args.no_download_missing),
        hf_base=str(args.hf_base),
    )
    print(f"[msst] runtime ready: {rt}")
    print("")
    print("[msst] Next steps:")
    print(
        "  - Process: uv run python tools/msst_process_chain.py --input /path/to/song.wav"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
