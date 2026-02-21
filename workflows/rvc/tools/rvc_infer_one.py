from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voicelab_bootstrap import ensure_runtime_pythonpath
from rvc_init_runtime import init_runtime


def _default_output(input_path: Path, *, pitch: int) -> Path:
    # Match the plan's expected filename for the default test clip.
    name = input_path.name
    if "马头有大" in name:
        safe = "马头有大_xingtong"
    else:
        safe = f"{input_path.stem}_xingtong" if input_path.stem else "xuan_to_xingtong"
    return Path(__file__).resolve().parents[1] / "out_wav" / f"{safe}_pitch{pitch}.wav"


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer one file with a trained RVC model (RMVPE locked).")
    ap.add_argument("--exp-name", default="xingtong_v2_48k_f0", help="Model/index prefix.")
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Optional explicit model path or filename under runtime/assets/weights/. "
            "If omitted, uses <exp-name>.pth."
        ),
    )
    ap.add_argument("--input", default="/mnt/c/AIGC/炫神/马头有大马头来了_karaoke_noreverb_dry.wav")
    ap.add_argument("--output", default=None)

    ap.add_argument("--pitch", type=int, default=12, help="Pitch shift in semitones (+12 is one octave up).")
    ap.add_argument("--index-rate", type=float, default=0.8)
    ap.add_argument("--filter-radius", type=int, default=3)
    ap.add_argument("--resample-sr", type=int, default=0)
    ap.add_argument("--rms-mix-rate", type=float, default=0.25)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable peak-normalization (default is enabled to avoid silent/clipped output).",
    )
    ap.add_argument(
        "--subtype",
        default="PCM_16",
        choices=["PCM_16", "FLOAT"],
        help="Output WAV subtype. PCM_16 is the most compatible with players/editors.",
    )

    ap.add_argument("--device", default=None, help="Override device (e.g. cuda:0 / cpu).")
    ap.add_argument("--is-half", action="store_true", default=True, help="Use FP16 where applicable.")
    args = ap.parse_args()
    normalize = not bool(args.no_normalize)

    rt = init_runtime(force=False, assets_src=None)
    ensure_runtime_pythonpath()
    # Upstream RVC uses lots of relative paths like `configs/...`, `assets/...`, `logs/...`.
    # Ensure we run from the runtime root so `Config()` and VC pipeline can find files.
    os.chdir(rt)

    # Prevent upstream Config() from parsing this script's CLI args.
    sys.argv = sys.argv[:1]

    weights_dir = rt / "assets" / "weights"
    model_path = Path(args.model) if args.model else (weights_dir / f"{args.exp_name}.pth")
    if not model_path.is_absolute():
        # Treat as a filename under weights_dir.
        model_path = (weights_dir / model_path).resolve()
    index_path = rt / "indices" / f"{args.exp_name}.index"
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Run: uv run python tools/rvc_train.py\n"
            "Or pass --model <filename.pth> (saved under runtime/assets/weights/)."
        )
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}\nRun: uv run python tools/rvc_train_index.py")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input audio not found: {input_path}")

    out_path = Path(args.output) if args.output else _default_output(input_path, pitch=int(args.pitch))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Required by upstream VC.get_vc() and index lookup helper.
    os.environ["weight_root"] = str((rt / "assets" / "weights").resolve())
    os.environ["index_root"] = str((rt / "indices").resolve())
    # Required by upstream RMVPE inference (see runtime/infer/modules/vc/pipeline.py).
    os.environ["rmvpe_root"] = str((rt / "assets" / "rmvpe").resolve())

    try:
        from configs.config import Config  # type: ignore
        from infer.modules.vc.modules import VC  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"Missing dependencies for inference: {e}\n"
            "Run: cd workflows/rvc && uv sync"
        )

    config = Config()
    if args.device:
        config.device = args.device
    config.is_half = bool(args.is_half)

    vc = VC(config)
    # Upstream expects the sid to be the filename in weight_root.
    vc.get_vc(model_path.name)
    info, wav_opt = vc.vc_single(
        0,  # speaker id
        str(input_path),
        int(args.pitch),
        None,  # f0 file
        "rmvpe",  # locked
        str(index_path),
        None,
        float(args.index_rate),
        int(args.filter_radius),
        int(args.resample_sr),
        float(args.rms_mix_rate),
        float(args.protect),
    )

    sr, audio = wav_opt
    if sr is None or audio is None:
        raise SystemExit(f"Inference failed:\n{info}")

    # Upstream pipeline sometimes returns int16-scaled float arrays (range ~[-32768, 32767]).
    # Writing those as FLOAT produces extreme clipping / "electric noise". Normalize to [-1, 1].
    import numpy as np

    y = np.asarray(audio)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    absmax = float(np.max(np.abs(y))) if y.size else 0.0
    if absmax > 1.5:
        # Heuristic: treat as int16-scaled audio if within a plausible range.
        if absmax <= 40000:
            y = y / 32768.0
        else:
            y = y / absmax
    if normalize and y.size:
        absmax2 = float(np.max(np.abs(y)))
        if absmax2 > 0:
            y = y * (0.95 / absmax2)

    sf.write(str(out_path), y.astype(np.float32, copy=False), int(sr), subtype=str(args.subtype))
    print(info)
    print(f"[rvc] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
