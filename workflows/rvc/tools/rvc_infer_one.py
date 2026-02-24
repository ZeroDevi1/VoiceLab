from __future__ import annotations

import argparse
import os
import sys
import tempfile
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


def _coerce_mono(y):
    import numpy as np

    a = np.asarray(y)
    if a.ndim == 1:
        return a
    # (T, C) -> mono
    return a.mean(axis=1)


def _normalize_for_write(y, *, normalize: bool):
    """
    Normalize to avoid silent/clipped output and to handle upstream returning int16-scaled floats.
    Works with mono (T,) or multi-channel (T, C) arrays.
    """
    import numpy as np

    a = np.asarray(y, dtype=np.float32)
    if a.ndim == 1:
        a2 = a
    else:
        a2 = a.reshape(-1)

    a2 = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)
    absmax = float(np.max(np.abs(a2))) if a2.size else 0.0
    if absmax > 1.5:
        if absmax <= 40000:
            a = a / 32768.0
        else:
            a = a / absmax
    if normalize:
        absmax2 = float(np.max(np.abs(a))) if a.size else 0.0
        if absmax2 > 0:
            a = a * (0.95 / absmax2)
    return a.astype(np.float32, copy=False)


def _pan_gains_from_input(
    lr: "np.ndarray",
    *,
    sr_in: int,
    out_len: int,
    sr_out: int,
    window_ms: int,
    hop_ms: int,
    strength: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Derive per-sample L/R gains (at output sr) from the input stereo RMS envelope.
    This preserves panning without introducing phase inversion (pure gain only).
    """
    import numpy as np

    if lr.ndim != 2 or lr.shape[1] != 2:
        raise ValueError("expected stereo input (T, 2)")
    xL = lr[:, 0].astype(np.float32, copy=False)
    xR = lr[:, 1].astype(np.float32, copy=False)

    frame = max(16, int(sr_in * (window_ms / 1000.0)))
    hop = max(1, int(sr_in * (hop_ms / 1000.0)))

    def _rms_frames(x: "np.ndarray") -> "np.ndarray":
        if x.size < frame:
            return np.array([float(np.sqrt(np.mean(x * x) + 1e-12))], dtype=np.float32)
        n = 1 + (x.size - frame) // hop
        out = np.empty((n,), dtype=np.float32)
        for i in range(n):
            s = i * hop
            w = x[s : s + frame]
            out[i] = float(np.sqrt(np.mean(w * w) + 1e-12))
        return out

    rmsL = _rms_frames(xL)
    rmsR = _rms_frames(xR)
    eps = 1e-8
    r = rmsL / (rmsL + rmsR + eps)  # [0..1]
    # Strength blends ratio towards center (0.5).
    strength = float(max(0.0, min(1.0, strength)))
    r = 0.5 + strength * (r - 0.5)
    r = np.clip(r, 0.0, 1.0)

    # Equal-power panning gains.
    gL_f = np.sqrt(r).astype(np.float32)
    gR_f = np.sqrt(1.0 - r).astype(np.float32)

    # Map frame centers (input time) -> sample-level gains (output time).
    centers = (np.arange(len(r), dtype=np.float32) * hop + (frame * 0.5)) / float(sr_in)
    t_out = np.arange(int(out_len), dtype=np.float32) / float(sr_out)
    gL = np.interp(t_out, centers, gL_f, left=float(gL_f[0]), right=float(gL_f[-1])).astype(np.float32)
    gR = np.interp(t_out, centers, gR_f, left=float(gR_f[0]), right=float(gR_f[-1])).astype(np.float32)
    return gL, gR


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer one file with a trained RVC model (RMVPE locked).")
    ap.add_argument("--exp-name", default="xingtong_v2_48k_f0", help="Model/index prefix.")
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Optional explicit model path, a filename under runtime/assets/weights/, or 'latest'. "
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
        "--stereo-mode",
        default="mono",
        choices=["mono", "pan", "dual"],
        help=(
            "How to handle stereo input. "
            "'mono': downmix to mono (fast, default). "
            "'pan': convert mono then re-apply input panning as gain envelopes (recommended for vocals). "
            "'dual': convert L and R independently (slow; may cause width/phase artifacts)."
        ),
    )
    ap.add_argument("--pan-window-ms", type=int, default=50, help="RMS window for --stereo-mode pan.")
    ap.add_argument("--pan-hop-ms", type=int, default=10, help="RMS hop for --stereo-mode pan.")
    ap.add_argument("--pan-strength", type=float, default=1.0, help="0..1, blend panning towards center.")
    ap.add_argument(
        "--dual-polarity-fix",
        action="store_true",
        default=True,
        help="For --stereo-mode dual, flip one channel if outputs are strongly negatively correlated.",
    )
    ap.add_argument("--dual-polarity-threshold", type=float, default=-0.05)
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
    model_arg = str(args.model).strip() if args.model else None
    if not model_arg:
        model_path = (weights_dir / f"{args.exp_name}.pth").resolve()
    elif model_arg.lower() in {"latest", "auto"}:
        # Requires inference-ready weights under runtime/assets/weights/.
        # These are produced at:
        # - training end (final <exp-name>.pth), OR
        # - during training if you enable `--save-every-weights 1` in tools/rvc_train.py.
        candidates = sorted(
            weights_dir.glob(f"{args.exp_name}*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(
                f"No weights found under: {weights_dir} (exp-name prefix: {args.exp_name})\n"
                "If you stopped training early, rerun training with:\n"
                "  uv run python tools/rvc_train.py --save-every-weights 1 --save-every-epoch 10 ...\n"
                "Then infer with:\n"
                "  uv run python tools/rvc_infer_one.py --model latest ..."
            )
        model_path = candidates[0].resolve()
        print(f"[rvc] auto-selected latest model: {model_path.name}")
    else:
        model_path = Path(model_arg)
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

    def _infer_one(in_file: Path):
        info, wav_opt = vc.vc_single(
            0,  # speaker id
            str(in_file),
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
        return info, int(sr), _coerce_mono(audio)

    # Stereo handling: default stays mono for compatibility with existing workflow.
    in_data, in_sr = sf.read(str(input_path), always_2d=True)
    y_out = None
    info_out = None
    sr_out = None

    if args.stereo_mode == "mono" or in_data.shape[1] == 1:
        info_out, sr_out, y_mono = _infer_one(input_path)
        y_out = y_mono
    elif args.stereo_mode == "pan":
        # Convert mono (downmix input) then re-apply input panning as time-varying gains.
        mono_in = in_data.mean(axis=1)
        with tempfile.TemporaryDirectory(prefix="rvc_infer_pan_") as td:
            td_p = Path(td)
            tmp_in = td_p / "in_mono.wav"
            sf.write(str(tmp_in), mono_in.astype("float32", copy=False), int(in_sr))
            info_out, sr_out, y_mono = _infer_one(tmp_in)

        gL, gR = _pan_gains_from_input(
            in_data,
            sr_in=int(in_sr),
            out_len=int(y_mono.shape[0]),
            sr_out=int(sr_out),
            window_ms=int(args.pan_window_ms),
            hop_ms=int(args.pan_hop_ms),
            strength=float(args.pan_strength),
        )
        import numpy as np

        y_out = np.stack([y_mono * gL, y_mono * gR], axis=1)
    elif args.stereo_mode == "dual":
        import numpy as np

        L = in_data[:, 0]
        R = in_data[:, 1]
        with tempfile.TemporaryDirectory(prefix="rvc_infer_dual_") as td:
            td_p = Path(td)
            in_L = td_p / "in_L.wav"
            in_R = td_p / "in_R.wav"
            sf.write(str(in_L), L.astype("float32", copy=False), int(in_sr))
            sf.write(str(in_R), R.astype("float32", copy=False), int(in_sr))
            infoL, srL, yL = _infer_one(in_L)
            infoR, srR, yR = _infer_one(in_R)
        if srL != srR:
            raise SystemExit(f"Dual inference SR mismatch: L={srL}, R={srR}")
        sr_out = srL
        info_out = infoL
        n = min(int(yL.shape[0]), int(yR.shape[0]))
        yL = np.asarray(yL[:n], dtype=np.float32)
        yR = np.asarray(yR[:n], dtype=np.float32)
        if args.dual_polarity_fix:
            m = min(n, int(sr_out * 5))  # first ~5s for correlation estimate
            if m > 1024:
                a = yL[:m]
                b = yR[:m]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0
                if corr < float(args.dual_polarity_threshold):
                    yR = -yR
                    print(f"[rvc] dual polarity fix: corr={corr:.3f} < {float(args.dual_polarity_threshold):.3f}, flipped R")
        y_out = np.stack([yL, yR], axis=1)
    else:
        raise SystemExit(f"Unsupported stereo mode: {args.stereo_mode}")

    assert sr_out is not None and info_out is not None and y_out is not None
    y_out = _normalize_for_write(y_out, normalize=normalize)

    sf.write(str(out_path), y_out, int(sr_out), subtype=str(args.subtype))
    print(info_out)
    print(f"[rvc] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
