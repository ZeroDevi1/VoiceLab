from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from rvc_init_runtime import init_runtime
from voicelab_bootstrap import ensure_runtime_pythonpath


def _coerce_mono(y):
    import numpy as np

    a = np.asarray(y)
    if a.ndim == 1:
        return a
    return a.mean(axis=1)


def _normalize_for_write(y, *, normalize: bool):
    import numpy as np

    a = np.asarray(y, dtype=np.float32)
    flat = a if a.ndim == 1 else a.reshape(-1)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    absmax = float(np.max(np.abs(flat))) if flat.size else 0.0
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
):
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
    r = rmsL / (rmsL + rmsR + eps)

    strength = float(max(0.0, min(1.0, strength)))
    r = 0.5 + strength * (r - 0.5)
    r = np.clip(r, 0.0, 1.0)

    gL_f = np.sqrt(r).astype(np.float32)
    gR_f = np.sqrt(1.0 - r).astype(np.float32)

    centers = (np.arange(len(r), dtype=np.float32) * hop + (frame * 0.5)) / float(sr_in)
    t_out = np.arange(int(out_len), dtype=np.float32) / float(sr_out)
    gL = np.interp(t_out, centers, gL_f, left=float(gL_f[0]), right=float(gL_f[-1])).astype(np.float32)
    gR = np.interp(t_out, centers, gR_f, left=float(gR_f[0]), right=float(gR_f[-1])).astype(np.float32)
    return gL, gR


def _iter_inputs(root: Path, *, glob_pat: str, recursive: bool) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Input dir not found: {root}")
    if recursive:
        return sorted([p for p in root.rglob(glob_pat) if p.is_file()])
    return sorted([p for p in root.glob(glob_pat) if p.is_file()])


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch infer a directory with a trained RVC model (single process).")
    ap.add_argument("--exp-name", required=True, help="Model/index prefix.")
    ap.add_argument("--model", default="latest", help="Model path/filename, or 'latest'.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--glob", dest="glob_pat", default="*.wav")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N files (debug).")

    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--f0-method", default="rmvpe", choices=["pm", "harvest", "crepe", "rmvpe", "fcpe"])
    ap.add_argument("--index-rate", type=float, default=0.7)
    ap.add_argument("--filter-radius", type=int, default=3)
    ap.add_argument("--resample-sr", type=int, default=0)
    ap.add_argument("--rms-mix-rate", type=float, default=0.25)
    ap.add_argument("--protect", type=float, default=0.33)

    ap.add_argument("--stereo-mode", default="mono", choices=["mono", "pan", "dual"])
    ap.add_argument("--pan-window-ms", type=int, default=50)
    ap.add_argument("--pan-hop-ms", type=int, default=10)
    ap.add_argument("--pan-strength", type=float, default=1.0)
    ap.add_argument("--dual-polarity-fix", action="store_true", default=True)
    ap.add_argument("--dual-polarity-threshold", type=float, default=-0.05)

    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--subtype", default="PCM_16", choices=["PCM_16", "FLOAT"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--is-half", action="store_true", default=True)
    args = ap.parse_args()
    normalize = not bool(args.no_normalize)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap runtime + imports.
    rt = init_runtime(force=False, assets_src=None, download_missing=False, hf_base="https://hf-mirror.com")
    ensure_runtime_pythonpath()
    os.chdir(rt)
    sys.argv = sys.argv[:1]

    weights_dir = rt / "assets" / "weights"
    model_arg = str(args.model).strip() if args.model else ""
    if model_arg.lower() in {"latest", "auto"}:
        candidates = sorted(
            weights_dir.glob(f"{args.exp_name}*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(f"No weights found under: {weights_dir} (exp-name prefix: {args.exp_name})")
        model_path = candidates[0].resolve()
        print(f"[rvc] auto-selected latest model: {model_path.name}")
    else:
        model_path = Path(model_arg)
        if not model_path.is_absolute():
            model_path = (weights_dir / model_path).resolve()

    index_path = rt / "indices" / f"{args.exp_name}.index"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}\nRun: uv run python tools/rvc_train_index.py --exp-name {args.exp_name}")

    os.environ["weight_root"] = str((rt / "assets" / "weights").resolve())
    os.environ["index_root"] = str((rt / "indices").resolve())
    os.environ["rmvpe_root"] = str((rt / "assets" / "rmvpe").resolve())

    try:
        from configs.config import Config  # type: ignore
        from infer.modules.vc.modules import VC  # type: ignore
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependencies for inference: {e}\nRun: cd workflows/rvc && uv sync")

    config = Config()
    if args.device:
        config.device = args.device
    config.is_half = bool(args.is_half)

    vc = VC(config)
    vc.get_vc(model_path.name)

    def _infer_one(in_file: Path):
        info, wav_opt = vc.vc_single(
            0,
            str(in_file),
            int(args.pitch),
            None,
            str(args.f0_method),
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
            raise RuntimeError(info)
        return str(info), int(sr), _coerce_mono(audio)

    inputs = _iter_inputs(input_dir, glob_pat=str(args.glob_pat), recursive=bool(args.recursive))
    if args.limit and int(args.limit) > 0:
        inputs = inputs[: int(args.limit)]
    if not inputs:
        raise SystemExit(f"No input files matched: {input_dir} / {args.glob_pat}")

    ok = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []

    for i, in_path in enumerate(inputs, start=1):
        out_path = output_dir / (in_path.stem + ".wav")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            info_out = None
            sr_out = None
            y_out = None

            info = sf.info(str(in_path))
            channels = int(info.channels)
            sr_in = int(info.samplerate)

            if args.stereo_mode == "mono" or channels == 1:
                info_out, sr_out, y_mono = _infer_one(in_path)
                y_out = y_mono
            elif args.stereo_mode == "pan":
                in_lr, _ = sf.read(str(in_path), always_2d=True, dtype="float32")
                mono_in = in_lr.mean(axis=1)
                with tempfile.TemporaryDirectory(prefix="rvc_infer_pan_") as td:
                    tmp_in = Path(td) / "in_mono.wav"
                    sf.write(str(tmp_in), mono_in, sr_in)
                    info_out, sr_out, y_mono = _infer_one(tmp_in)
                gL, gR = _pan_gains_from_input(
                    in_lr,
                    sr_in=sr_in,
                    out_len=int(y_mono.shape[0]),
                    sr_out=int(sr_out),
                    window_ms=int(args.pan_window_ms),
                    hop_ms=int(args.pan_hop_ms),
                    strength=float(args.pan_strength),
                )
                y_out = np.stack([y_mono * gL, y_mono * gR], axis=1)
            else:  # dual
                in_lr, _ = sf.read(str(in_path), always_2d=True, dtype="float32")
                L = in_lr[:, 0]
                R = in_lr[:, 1]
                with tempfile.TemporaryDirectory(prefix="rvc_infer_dual_") as td:
                    td_p = Path(td)
                    in_L = td_p / "in_L.wav"
                    in_R = td_p / "in_R.wav"
                    sf.write(str(in_L), L, sr_in)
                    sf.write(str(in_R), R, sr_in)
                    infoL, srL, yL = _infer_one(in_L)
                    infoR, srR, yR = _infer_one(in_R)
                if srL != srR:
                    raise RuntimeError(f"Dual SR mismatch: L={srL}, R={srR}")
                info_out = infoL
                sr_out = srL
                n = min(int(yL.shape[0]), int(yR.shape[0]))
                yL = np.asarray(yL[:n], dtype=np.float32)
                yR = np.asarray(yR[:n], dtype=np.float32)
                if args.dual_polarity_fix:
                    m = min(n, int(sr_out * 5))
                    if m > 1024:
                        a = yL[:m]
                        b = yR[:m]
                        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                        corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0
                        if corr < float(args.dual_polarity_threshold):
                            yR = -yR
                y_out = np.stack([yL, yR], axis=1)

            assert info_out is not None and sr_out is not None and y_out is not None
            y_out = _normalize_for_write(y_out, normalize=normalize)

            # Write atomically to avoid half-written files on interruption.
            # Keep a real audio suffix so libsndfile can infer the format.
            # Example: xuan_1.wav -> xuan_1.partial.wav
            tmp_out = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
            sf.write(str(tmp_out), y_out, int(sr_out), subtype=str(args.subtype))
            os.replace(tmp_out, out_path)

            ok += 1
            if i == 1 or i % 50 == 0:
                print(f"[rvc] {i}/{len(inputs)} ok={ok} skipped={skipped} failed={len(failed)}", flush=True)
        except Exception as e:
            failed.append((in_path, str(e)))
            print(f"[rvc] FAIL: {in_path.name}: {e}", flush=True)

    print(f"[rvc] done: total={len(inputs)} ok={ok} skipped={skipped} failed={len(failed)}")
    fail_log = output_dir / "_failed.txt"
    if failed:
        fail_log.write_text("\n".join(f"{p}\t{msg}" for p, msg in failed) + "\n", encoding="utf-8")
        print(f"[rvc] failures written: {fail_log}")
    else:
        # Avoid stale failure logs confusing batch runs.
        if fail_log.exists():
            fail_log.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
