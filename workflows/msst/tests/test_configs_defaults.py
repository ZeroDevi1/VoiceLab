from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _repo_root() -> Path:
    # .../workflows/msst/tests -> .../VoiceLab
    return Path(__file__).resolve().parents[3]


def _runtime_cfg(path: str) -> Path:
    return _repo_root() / "workflows" / "msst" / "runtime" / "configs" / path


def _vendor_cfg_backup(path: str) -> Path:
    return _repo_root() / "vendor" / "MSST-WebUI" / "configs_backup" / path


def _load_yaml(p: Path) -> dict:
    # MSST configs use !!python/tuple tags; FullLoader matches upstream behavior.
    return yaml.load(p.read_text(encoding="utf-8"), Loader=yaml.FullLoader)  # type: ignore[return-value]


def _pick_config(path: str) -> Path:
    rt = _runtime_cfg(path)
    if rt.exists():
        return rt
    vb = _vendor_cfg_backup(path)
    if vb.exists():
        return vb
    pytest.skip(f"Config not found (runtime not initialized and vendor missing): {path}")


@pytest.mark.parametrize(
    "rel,chunk_size,num_overlap,sr",
    [
        ("vocal_models/config_melbandroformer_inst.yaml", 485100, 4, 44100),
        ("vocal_models/big_beta5e.yaml", 485100, 2, 44100),
        ("single_stem_models/dereverb_mel_band_roformer_anvuew.yaml", 352800, 4, 44100),
        ("single_stem_models/model_mel_band_roformer_denoise.yaml", 352800, 4, 44100),
    ],
)
def test_official_configs_defaults(rel: str, chunk_size: int, num_overlap: int, sr: int) -> None:
    p = _pick_config(rel)
    cfg = _load_yaml(p)
    assert int(cfg["audio"]["chunk_size"]) == chunk_size
    assert int(cfg["inference"]["num_overlap"]) == num_overlap
    assert int(cfg["audio"]["sample_rate"]) == sr


def test_becruily_karaoke_config_defaults() -> None:
    p = _repo_root() / "workflows" / "msst" / "configs" / "config_karaoke_becruily.yaml"
    cfg = _load_yaml(p)
    assert int(cfg["audio"]["chunk_size"]) == 485100
    assert int(cfg["inference"]["num_overlap"]) == 8
    assert int(cfg["audio"]["sample_rate"]) == 44100

