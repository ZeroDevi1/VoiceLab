from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from voicelab_bootstrap import msst_vendor_root, runtime_root


@dataclass(frozen=True)
class ModelSpec:
    name: str
    rel_dest: Path  # under runtime/
    url: str
    sha256: str | None = None
    size_bytes: int | None = None


_OFFICIAL_MODELS: tuple[tuple[str, Path], ...] = (
    ("inst_v1e.ckpt", Path("pretrain/vocal_models/inst_v1e.ckpt")),
    ("big_beta5e.ckpt", Path("pretrain/vocal_models/big_beta5e.ckpt")),
    (
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        Path("pretrain/single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"),
    ),
    (
        "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        Path("pretrain/single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"),
    ),
)


# Third-party karaoke/lead-vs-backing model (not in MSST-WebUI's official model map).
_BECRUILY_KARAOKE = ModelSpec(
    name="mel_band_roformer_karaoke_becruily.ckpt",
    rel_dest=Path("pretrain/vocal_models/mel_band_roformer_karaoke_becruily.ckpt"),
    url="https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt",
    sha256="f66c4c2a3f93fc9373354fbf104f6925b0ac5020c4d714883940f72c2b6d4f17",
)


def _replace_hf_base(url: str, hf_base: str) -> str:
    hf_base = hf_base.rstrip("/")
    return (
        url.replace("https://huggingface.co", hf_base)
        .replace("http://huggingface.co", hf_base)
        .replace("https://hf-mirror.com", hf_base)
        .replace("http://hf-mirror.com", hf_base)
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_official_model_url(model_name: str) -> str:
    """
    Extract download URL from vendor/data/msst_model_map.json (official models only).
    """
    vendor = msst_vendor_root()
    mm = vendor / "data" / "msst_model_map.json"
    if not mm.exists():
        raise FileNotFoundError(f"Missing vendor model map: {mm}")

    obj = _load_json(mm)
    if not isinstance(obj, dict):
        raise ValueError("Unexpected msst_model_map.json shape")

    for _cat, lst in obj.items():
        if not isinstance(lst, list):
            continue
        for m in lst:
            if isinstance(m, dict) and m.get("name") == model_name and isinstance(m.get("link"), str):
                return str(m["link"])

    raise KeyError(f"Model not found in msst_model_map.json: {model_name}")


def _load_models_info() -> dict[str, dict]:
    info_path = runtime_root() / "data" / "models_info.json"
    if not info_path.exists():
        vendor = msst_vendor_root()
        candidate = vendor / "data" / "models_info.json"
        if candidate.exists():
            info_path = candidate
        else:
            raise FileNotFoundError(
                f"models_info.json not found under runtime or vendor: {runtime_root() / 'data'}"
            )
    obj = _load_json(info_path)
    if not isinstance(obj, dict):
        raise ValueError("Unexpected models_info.json shape")
    return obj  # type: ignore[return-value]


def build_model_specs(*, hf_base: str) -> list[ModelSpec]:
    info = _load_models_info()

    out: list[ModelSpec] = []
    for name, rel in _OFFICIAL_MODELS:
        url = _replace_hf_base(_find_official_model_url(name), hf_base)
        meta = info.get(name, {})
        sha = meta.get("sha256")
        size = meta.get("model_size")
        out.append(
            ModelSpec(
                name=name,
                rel_dest=rel,
                url=url,
                sha256=str(sha) if isinstance(sha, str) else None,
                size_bytes=int(size) if isinstance(size, int) else None,
            )
        )

    out.append(
        ModelSpec(
            name=_BECRUILY_KARAOKE.name,
            rel_dest=_BECRUILY_KARAOKE.rel_dest,
            url=_replace_hf_base(_BECRUILY_KARAOKE.url, hf_base),
            sha256=_BECRUILY_KARAOKE.sha256,
            size_bytes=_BECRUILY_KARAOKE.size_bytes,
        )
    )
    return out


def _download_one(*, spec: ModelSpec, dest: Path, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return

    partial = dest.with_suffix(dest.suffix + ".partial")
    resume_from = partial.stat().st_size if partial.exists() and not force else 0
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(spec.url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        # If server ignored Range and returned full content, restart from scratch.
        if resume_from > 0 and r.status_code == 200:
            resume_from = 0
        mode = "ab" if resume_from > 0 else "wb"
        with partial.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)

    os.replace(partial, dest)

    if spec.size_bytes is not None:
        actual = dest.stat().st_size
        if actual != spec.size_bytes:
            raise ValueError(f"Size mismatch for {spec.name}: expected={spec.size_bytes}, got={actual}")
    if spec.sha256:
        actual_sha = _sha256_file(dest)
        if actual_sha.lower() != spec.sha256.lower():
            raise ValueError(f"SHA256 mismatch for {spec.name}: expected={spec.sha256}, got={actual_sha}")


def download_models(*, specs: Iterable[ModelSpec], force: bool) -> None:
    rt = runtime_root()
    rt.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        dest = rt / spec.rel_dest
        print(f"[msst] download: {spec.name} -> {dest.name}")
        _download_one(spec=spec, dest=dest, force=force)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download MSST models into workflows/msst/runtime/pretrain/ ...")
    ap.add_argument("--hf-base", default="https://huggingface.co", help="HuggingFace base URL (e.g. https://hf-mirror.com)")
    ap.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = ap.parse_args()

    specs = build_model_specs(hf_base=str(args.hf_base))
    download_models(specs=specs, force=bool(args.force))
    print("[msst] OK: downloads complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
