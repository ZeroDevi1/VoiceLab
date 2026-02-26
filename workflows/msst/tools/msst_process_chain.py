from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from voicelab_bootstrap import ensure_runtime_pythonpath, runtime_root


@dataclass(frozen=True)
class ChainOutputs:
    other: Path
    vocals: Path
    vocals_karaoke: Path
    vocals_other: Path
    vocals_karaoke_noreverb: Path
    vocals_karaoke_noreverb_dry: Path


def _ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise SystemExit(f"ffmpeg not found or not runnable: {exc}")


def _wav_subtype_arg(s: str) -> str:
    v = str(s).strip().upper()
    if v in {"FLOAT", "PCM_16", "PCM_24"}:
        return v
    raise argparse.ArgumentTypeError("wav bit depth must be one of: FLOAT, PCM_16, PCM_24")


def _collect_inputs(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise SystemExit(f"--input does not exist: {p}")

    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
    out = [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in exts]
    if not out:
        raise SystemExit(f"No audio files found under: {p}")
    return out


def _preconvert_to_wav(*, src: Path, dst_dir: Path) -> Path:
    """
    Convert to WAV 44.1kHz stereo for stability/reproducibility.
    Keep basename stem to preserve downstream naming.
    """
    _ensure_ffmpeg()
    out = dst_dir / f"{src.stem}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "44100",
        "-ac",
        "2",
        str(out),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg preconvert failed for {src}:\n{p.stderr.strip() or p.stdout.strip()}")
    return out


def _assert_nonempty(path: Path, *, hint: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing expected output: {path}\nHint: {hint}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"Empty output file: {path}\nHint: {hint}")


def _runtime_paths() -> dict[str, Path]:
    rt = runtime_root()
    return {
        "rt": rt,
        "inst_model": rt / "pretrain" / "vocal_models" / "inst_v1e.ckpt",
        "inst_cfg": rt / "configs" / "vocal_models" / "inst_v1e.ckpt.yaml",
        "vocal_model": rt / "pretrain" / "vocal_models" / "big_beta5e.ckpt",
        "vocal_cfg": rt / "configs" / "vocal_models" / "big_beta5e.ckpt.yaml",
        # Step2: official karaoke model shipped by MSST-WebUI (stable config + compatible code).
        "karaoke_model": rt
        / "pretrain"
        / "vocal_models"
        / "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "karaoke_cfg": rt
        / "configs"
        / "vocal_models"
        / "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt.yaml",
        "dereverb_model": rt
        / "pretrain"
        / "single_stem_models"
        / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        "dereverb_cfg": rt
        / "configs"
        / "single_stem_models"
        / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt.yaml",
        "denoise_model": rt
        / "pretrain"
        / "single_stem_models"
        / "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        "denoise_cfg": rt
        / "configs"
        / "single_stem_models"
        / "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt.yaml",
    }


def _assert_runtime_ready() -> None:
    rt = runtime_root()
    if not rt.exists():
        raise SystemExit(f"MSST runtime not initialized: {rt}\nRun: uv run python tools/msst_init_runtime.py")

    rp = _runtime_paths()
    required = [
        rp["inst_model"],
        rp["inst_cfg"],
        rp["vocal_model"],
        rp["vocal_cfg"],
        rp["karaoke_model"],
        rp["karaoke_cfg"],
        rp["dereverb_model"],
        rp["dereverb_cfg"],
        rp["denoise_model"],
        rp["denoise_cfg"],
        rt / "inference",
        rt / "modules",
        rt / "utils",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "[msst] runtime is incomplete:\n" + "\n".join(f"  - missing: {p}" for p in missing) + "\n"
            "Run: uv run python tools/msst_init_runtime.py"
        )


def _msseparator():
    # Import lazily after runtime pythonpath is set.
    from inference.msst_infer import MSSeparator

    return MSSeparator


def _run_separator(
    *,
    model_type: str,
    model_path: Path,
    config_path: Path,
    input_dir: Path,
    output_dir: Path | dict[str, str],
    device: str,
    gpu_ids: list[int],
    output_format: str,
    wav_bit_depth: str,
    use_tta: bool,
) -> None:
    MSSeparator = _msseparator()
    sep = MSSeparator(
        model_type=model_type,
        config_path=str(config_path),
        model_path=str(model_path),
        device=device,
        device_ids=gpu_ids,
        output_format=output_format,
        use_tta=use_tta,
        store_dirs=output_dir,
        audio_params={
            "wav_bit_depth": wav_bit_depth,
            "flac_bit_depth": "PCM_24",
            "mp3_bit_rate": "320k",
        },
        debug=False,
    )
    sep.process_folder(str(input_dir))
    sep.del_cache()


def expected_outputs(*, base_stem: str, output_dir: Path, ext: str = ".wav") -> ChainOutputs:
    # Keep naming consistent; only the extension changes.
    if not ext.startswith("."):
        ext = "." + ext
    return ChainOutputs(
        other=output_dir / f"{base_stem}_other{ext}",
        vocals=output_dir / f"{base_stem}_vocals{ext}",
        vocals_karaoke=output_dir / f"{base_stem}_vocals_karaoke{ext}",
        vocals_other=output_dir / f"{base_stem}_vocals_other{ext}",
        vocals_karaoke_noreverb=output_dir / f"{base_stem}_vocals_karaoke_noreverb{ext}",
        vocals_karaoke_noreverb_dry=output_dir / f"{base_stem}_vocals_karaoke_noreverb_dry{ext}",
    )


def process_one(
    *,
    input_path: Path,
    output_dir: Path,
    device_mode: str,
    gpu_ids: list[int],
    output_format: str,
    wav_bit_depth: str,
    preconvert: bool,
) -> ChainOutputs:
    _assert_runtime_ready()
    ensure_runtime_pythonpath()

    # Upstream MSST uses relative imports; run from the runtime root.
    os.chdir(runtime_root())

    rp = _runtime_paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "." + output_format.lower().lstrip(".")
    outs = expected_outputs(base_stem=input_path.stem, output_dir=output_dir, ext=ext)

    # Device: MSSeparator's device handling is "auto" unless you pass cpu.
    if device_mode == "cpu":
        msst_device = "cpu"
    elif device_mode == "cuda":
        # Keep MSSeparator in "auto" mode but require CUDA.
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False. Use --device cpu instead.")
        msst_device = "auto"
    else:
        msst_device = "auto"

    with tempfile.TemporaryDirectory(prefix="voicelab-msst-") as td:
        td_path = Path(td)

        # Stage 0 input (wav 44.1k stereo).
        mix_dir = td_path / "mix"
        mix_dir.mkdir(parents=True, exist_ok=True)
        if preconvert:
            _preconvert_to_wav(src=input_path, dst_dir=mix_dir)
        else:
            shutil.copy2(input_path, mix_dir / input_path.name)

        # Step0: Instrumental separation (inst_v1e) -> *_other.wav
        _run_separator(
            model_type="mel_band_roformer",
            model_path=rp["inst_model"],
            config_path=rp["inst_cfg"],
            input_dir=mix_dir,
            output_dir={"other": str(output_dir)},
            device=msst_device,
            gpu_ids=gpu_ids,
            output_format=output_format,
            wav_bit_depth=wav_bit_depth,
            use_tta=False,
        )
        _assert_nonempty(outs.other, hint="Step0(inst) failed. Check model/config paths and GPU/CPU availability.")

        # Step1: Lead vocals extraction (big_beta5e) -> *_vocals.wav
        _run_separator(
            model_type="mel_band_roformer",
            model_path=rp["vocal_model"],
            config_path=rp["vocal_cfg"],
            input_dir=mix_dir,
            output_dir={"vocals": str(output_dir)},
            device=msst_device,
            gpu_ids=gpu_ids,
            output_format=output_format,
            wav_bit_depth=wav_bit_depth,
            use_tta=False,
        )
        _assert_nonempty(outs.vocals, hint="Step1(vocals) failed. Check big_beta5e model/config.")

        # Step2: Karaoke/backing split (becruily) on Step1 vocals -> *_vocals_karaoke.wav + *_vocals_other.wav
        step2_dir = td_path / "step2"
        step2_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outs.vocals, step2_dir / outs.vocals.name)

        _run_separator(
            model_type="mel_band_roformer",
            model_path=rp["karaoke_model"],
            config_path=rp["karaoke_cfg"],
            input_dir=step2_dir,
            output_dir=str(output_dir),
            device=msst_device,
            gpu_ids=gpu_ids,
            output_format=output_format,
            wav_bit_depth=wav_bit_depth,
            use_tta=False,
        )

        # Default karaoke model outputs *_karaoke + *_other directly.
        # Keep a compatibility path for models that output Vocals/Instrumental.
        if outs.vocals_karaoke.exists() and outs.vocals_other.exists():
            _assert_nonempty(outs.vocals_karaoke, hint="Step2(karaoke) produced empty output.")
            _assert_nonempty(outs.vocals_other, hint="Step2(other) produced empty output.")
        else:
            src_vocals = output_dir / f"{outs.vocals.stem}_Vocals{ext}"
            src_instr = output_dir / f"{outs.vocals.stem}_Instrumental{ext}"
            _assert_nonempty(
                src_vocals, hint="Step2(karaoke) missing output. Expected *_karaoke or *_Vocals depending on model."
            )
            _assert_nonempty(
                src_instr, hint="Step2(other) missing output. Expected *_other or *_Instrumental depending on model."
            )
            os.replace(src_vocals, outs.vocals_karaoke)
            os.replace(src_instr, outs.vocals_other)
            _assert_nonempty(outs.vocals_karaoke, hint="Step2(karaoke) rename failed.")
            _assert_nonempty(outs.vocals_other, hint="Step2(other) rename failed.")

        # Step3: De-reverb on karaoke -> *_noreverb.wav
        step3_dir = td_path / "step3"
        step3_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outs.vocals_karaoke, step3_dir / outs.vocals_karaoke.name)

        _run_separator(
            model_type="mel_band_roformer",
            model_path=rp["dereverb_model"],
            config_path=rp["dereverb_cfg"],
            input_dir=step3_dir,
            output_dir={"noreverb": str(output_dir)},
            device=msst_device,
            gpu_ids=gpu_ids,
            output_format=output_format,
            wav_bit_depth=wav_bit_depth,
            use_tta=False,
        )
        _assert_nonempty(
            outs.vocals_karaoke_noreverb, hint="Step3(dereverb) failed. Check dereverb model/config."
        )

        # Step4: De-noise on noreverb -> *_dry.wav (suffix becomes _noreverb_dry.wav)
        step4_dir = td_path / "step4"
        step4_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outs.vocals_karaoke_noreverb, step4_dir / outs.vocals_karaoke_noreverb.name)

        _run_separator(
            model_type="mel_band_roformer",
            model_path=rp["denoise_model"],
            config_path=rp["denoise_cfg"],
            input_dir=step4_dir,
            output_dir={"dry": str(output_dir)},
            device=msst_device,
            gpu_ids=gpu_ids,
            output_format=output_format,
            wav_bit_depth=wav_bit_depth,
            use_tta=False,
        )
        _assert_nonempty(outs.vocals_karaoke_noreverb_dry, hint="Step4(denoise) failed. Check denoise model/config.")

    return outs


def main() -> int:
    ap = argparse.ArgumentParser(description="MSST one-shot processing chain (inst->vocals->karaoke->dereverb->denoise)")
    ap.add_argument("--input", required=True, help="Input audio file or directory.")
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "out_wav"),
        help="Output directory (default: workflows/msst/out_wav).",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device mode.")
    ap.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids (default: 0).")
    ap.add_argument("--output-format", default="wav", choices=["wav", "flac", "mp3"], help="Output format.")
    ap.add_argument("--wav-bit-depth", default="FLOAT", type=_wav_subtype_arg, help="WAV subtype: FLOAT/PCM_16/PCM_24.")
    ap.add_argument(
        "--no-ffmpeg-preconvert",
        action="store_true",
        help="Disable ffmpeg preconvert (default: enabled to force 44.1kHz stereo wav).",
    )
    args = ap.parse_args()

    input_path = Path(str(args.input)).expanduser().resolve()
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    gpu_ids = [int(x) for x in str(args.gpu_ids).split(",") if str(x).strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]

    files = _collect_inputs(input_path)
    for f in files:
        print(f"[msst] process: {f.name}")
        outs = process_one(
            input_path=f,
            output_dir=output_dir,
            device_mode=str(args.device),
            gpu_ids=gpu_ids,
            output_format=str(args.output_format),
            wav_bit_depth=str(args.wav_bit_depth),
            preconvert=not bool(args.no_ffmpeg_preconvert),
        )
        print(f"[msst] OK: {outs.vocals_karaoke_noreverb_dry.name}")

    print(f"[msst] done -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
