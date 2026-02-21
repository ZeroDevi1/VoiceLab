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
    ap.add_argument("--input", default="/mnt/c/AIGC/炫神/马头有大！马头来了！.mp3")
    ap.add_argument("--output", default=None)

    ap.add_argument("--pitch", type=int, default=12, help="Pitch shift in semitones (+12 is one octave up).")
    ap.add_argument("--index-rate", type=float, default=0.8)
    ap.add_argument("--filter-radius", type=int, default=3)
    ap.add_argument("--resample-sr", type=int, default=0)
    ap.add_argument("--rms-mix-rate", type=float, default=0.25)
    ap.add_argument("--protect", type=float, default=0.33)

    ap.add_argument("--device", default=None, help="Override device (e.g. cuda:0 / cpu).")
    ap.add_argument("--is-half", action="store_true", default=True, help="Use FP16 where applicable.")
    args = ap.parse_args()

    rt = init_runtime(force=False, assets_src=None)
    ensure_runtime_pythonpath()
    # Upstream RVC uses lots of relative paths like `configs/...`, `assets/...`, `logs/...`.
    # Ensure we run from the runtime root so `Config()` and VC pipeline can find files.
    os.chdir(rt)

    # Prevent upstream Config() from parsing this script's CLI args.
    sys.argv = sys.argv[:1]

    model_path = rt / "assets" / "weights" / f"{args.exp_name}.pth"
    index_path = rt / "indices" / f"{args.exp_name}.index"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nRun: uv run python tools/rvc_train.py")
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
    # Required by upstream RMVPE f0 extraction in the inference pipeline.
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

    sf.write(str(out_path), audio, int(sr), subtype="FLOAT")
    print(info)
    print(f"[rvc] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
