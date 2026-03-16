from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_tools() -> None:
    tools = _repo_root() / "workflows" / "msst" / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


def _required_model_relpaths() -> list[Path]:
    return [
        Path("vocal_models/inst_v1e.ckpt"),
        Path("vocal_models/big_beta5e.ckpt"),
        Path("vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"),
        Path("single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"),
        Path("single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"),
    ]


def _create_vendor_tree(root: Path) -> Path:
    vendor = root / "vendor" / "MSST-WebUI"
    for name in ("inference", "modules", "utils", "configs_backup"):
        (vendor / name).mkdir(parents=True, exist_ok=True)
    data_backup = vendor / "data_backup"
    data_backup.mkdir(parents=True, exist_ok=True)
    models_info = {
        rel.name: {
            "link": f"https://huggingface.co/Sucial/MSST-WebUI/resolve/main/{rel.as_posix()}",
            "sha256": None,
            "model_size": None,
        }
        for rel in _required_model_relpaths()
    }
    (data_backup / "models_info.json").write_text(
        json.dumps(models_info),
        encoding="utf-8",
    )
    return vendor


def _populate_pretrain(root: Path, *, prefix: bytes = b"model") -> None:
    for rel in _required_model_relpaths():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prefix + b":" + rel.name.encode("utf-8"))


def _import_runtime_modules():
    _import_tools()
    import msst_download_models
    import msst_init_runtime
    import voicelab_bootstrap

    return msst_download_models, msst_init_runtime, voicelab_bootstrap


def test_assets_src_root_defaults_to_voicelab_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, _, bootstrap = _import_runtime_modules()
    monkeypatch.delenv("MSST_ASSETS_SRC_DIR", raising=False)
    monkeypatch.delenv("VOICELAB_ASSETS_DIR", raising=False)

    got = bootstrap.assets_src_root()

    assert got == (_repo_root() / ".cache" / "voicelab" / "assets" / "msst" / "pretrain").resolve()


def test_init_runtime_creates_default_cache_and_downloads_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    download_mod, init_mod, _ = _import_runtime_modules()
    vendor = _create_vendor_tree(tmp_path)
    runtime = tmp_path / "runtime"
    cache = tmp_path / "assets" / "msst" / "pretrain"
    calls: list[tuple[Path, str]] = []

    monkeypatch.setattr(init_mod, "msst_vendor_root", lambda: vendor)
    monkeypatch.setattr(init_mod, "runtime_root", lambda: runtime)
    monkeypatch.setattr(init_mod, "assets_src_root", lambda: cache)

    def fake_build_model_specs(*, hf_base: str):
        return [{"hf_base": hf_base}]

    def fake_download_models(*, specs, force: bool, dest_root: Path, dest_layout: str) -> None:
        calls.append((dest_root, dest_layout))
        assert force is False
        assert dest_layout == "pretrain"
        assert dest_root == cache
        assert list(specs) == [{"hf_base": "https://hf-mirror.com"}]
        _populate_pretrain(dest_root, prefix=b"downloaded")

    monkeypatch.setattr(download_mod, "build_model_specs", fake_build_model_specs)
    monkeypatch.setattr(download_mod, "download_models", fake_download_models)

    got = init_mod.init_runtime(
        force=False,
        assets_src=None,
        download_missing=True,
        hf_base="https://hf-mirror.com",
    )

    assert got == runtime
    assert cache.is_dir()
    assert calls == [(cache, "pretrain")]
    for path in init_mod._required_model_paths(runtime):
        assert path.exists()


def test_init_runtime_reuses_cached_models_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    download_mod, init_mod, _ = _import_runtime_modules()
    vendor = _create_vendor_tree(tmp_path)
    runtime = tmp_path / "runtime"
    cache = tmp_path / "assets" / "msst" / "pretrain"
    _populate_pretrain(cache, prefix=b"cached")

    monkeypatch.setattr(init_mod, "msst_vendor_root", lambda: vendor)
    monkeypatch.setattr(init_mod, "runtime_root", lambda: runtime)
    monkeypatch.setattr(init_mod, "assets_src_root", lambda: cache)
    monkeypatch.setattr(
        download_mod,
        "download_models",
        lambda *args, **kwargs: pytest.fail("download_models should not be called"),
    )

    init_mod.init_runtime(
        force=False,
        assets_src=None,
        download_missing=True,
        hf_base="https://hf-mirror.com",
    )

    for path in init_mod._required_model_paths(runtime):
        assert path.exists()


def test_init_runtime_prefers_explicit_assets_src(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    download_mod, init_mod, _ = _import_runtime_modules()
    vendor = _create_vendor_tree(tmp_path)
    runtime = tmp_path / "runtime"
    default_cache = tmp_path / "assets" / "msst" / "pretrain"
    explicit_assets = tmp_path / "manual-msst" / "pretrain"
    _populate_pretrain(explicit_assets, prefix=b"explicit")

    monkeypatch.setattr(init_mod, "msst_vendor_root", lambda: vendor)
    monkeypatch.setattr(init_mod, "runtime_root", lambda: runtime)
    monkeypatch.setattr(init_mod, "assets_src_root", lambda: default_cache)
    monkeypatch.setattr(
        download_mod,
        "download_models",
        lambda *args, **kwargs: pytest.fail("download_models should not be called"),
    )

    init_mod.init_runtime(
        force=False,
        assets_src=explicit_assets,
        download_missing=False,
        hf_base="https://hf-mirror.com",
    )

    assert not default_cache.exists()
    for path in init_mod._required_model_paths(runtime):
        assert path.exists()


def test_init_runtime_reports_clear_error_when_download_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, init_mod, _ = _import_runtime_modules()
    vendor = _create_vendor_tree(tmp_path)
    runtime = tmp_path / "runtime"
    cache = tmp_path / "assets" / "msst" / "pretrain"

    monkeypatch.setattr(init_mod, "msst_vendor_root", lambda: vendor)
    monkeypatch.setattr(init_mod, "runtime_root", lambda: runtime)
    monkeypatch.setattr(init_mod, "assets_src_root", lambda: cache)

    with pytest.raises(SystemExit) as excinfo:
        init_mod.init_runtime(
            force=False,
            assets_src=None,
            download_missing=False,
            hf_base="https://hf-mirror.com",
        )

    msg = str(excinfo.value)
    assert "Assets source dir not found" in msg
    assert "--assets-src" in msg
    assert "--no-download-missing" in msg
