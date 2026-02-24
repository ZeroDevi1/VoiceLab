from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _default_out(*, pitch: int) -> Path:
    # Keep filenames ASCII to avoid shell/FS quirks; user can override with --output.
    return Path(__file__).resolve().parents[1] / "out_wav" / f"taifeng_jj_to_xuan_pitch{pitch}.wav"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Infer a fixed vocals test song (Taifeng - JiangJiang) -> xuan using tools/rvc_infer_one.py."
    )
    ap.add_argument("--exp-name", default="xuan_v2_48k_f0")
    ap.add_argument(
        "--model",
        default=None,
        help="Optional model path/filename, or 'latest' to auto-pick newest weights for this exp-name.",
    )
    ap.add_argument(
        "--input",
        default="/mnt/c/AIGC/音乐/台风/台风 - 蒋蒋_vocals_karaoke_noreverb_dry.wav",
        help="Vocals-only WAV path.",
    )
    ap.add_argument("--output", default=None)

    # Preset: tuned for vocals; keep pitch=0 by default to preserve the song key.
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--index-rate", type=float, default=0.65)
    ap.add_argument("--filter-radius", type=int, default=3)
    ap.add_argument("--rms-mix-rate", type=float, default=0.25)
    ap.add_argument("--protect", type=float, default=0.4)
    ap.add_argument(
        "--stereo-mode",
        default="pan",
        choices=["mono", "pan", "dual"],
        help="Default to 'pan' for vocals stems to preserve panning without phase inversion.",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input audio not found: {input_path}")

    out_path = Path(args.output) if args.output else _default_out(pitch=int(args.pitch))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    infer_one = Path(__file__).resolve().parent / "rvc_infer_one.py"
    cmd = [
        sys.executable,
        str(infer_one),
        "--exp-name",
        str(args.exp_name),
    ]
    if args.model:
        cmd += ["--model", str(args.model)]
    cmd += [
        "--input",
        str(input_path),
        "--output",
        str(out_path),
        "--pitch",
        str(int(args.pitch)),
        "--index-rate",
        str(float(args.index_rate)),
        "--filter-radius",
        str(int(args.filter_radius)),
        "--rms-mix-rate",
        str(float(args.rms_mix_rate)),
        "--protect",
        str(float(args.protect)),
        "--stereo-mode",
        str(args.stereo_mode),
    ]

    # Flush so the wrapper's intention prints before the long-running subprocess output.
    print("[rvc] delegating to rvc_infer_one.py ...", flush=True)
    print("[rvc] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
