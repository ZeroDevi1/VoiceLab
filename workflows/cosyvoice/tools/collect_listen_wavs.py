#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import wave
from pathlib import Path


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return float(w.getnframes()) / float(w.getframerate())


def _safe_name(s: str) -> str:
    # Keep names portable for Windows/macOS/Linux.
    s = s.strip().replace(" ", "_")
    for ch in ['"', "'", ":", ";", "|", "\n", "\r", "\t"]:
        s = s.replace(ch, "")
    return s


def _read_run_json(out_dir: Path) -> dict | None:
    p = out_dir / "run.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _fmt_int_tag(name: str, value: int, width: int) -> str:
    return f"{name}{int(value):0{width}d}"

def _fmt_float_tag(name: str, value: float, scale: int, width: int) -> str:
    # Example: top_p=0.6 -> tp060 if scale=100.
    v = int(round(float(value) * scale))
    return f"{name}{v:0{width}d}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Collect CosyVoice inference wav files into one folder with parameter-consistent names. "
            "By default this collects infer_sft.py outputs (chunk_0000.wav)."
        )
    )
    ap.add_argument("--src", type=str, required=True, help="Root directory to scan (e.g. out_wav/dream_pick_sampling06k10).")
    ap.add_argument("--dst", type=str, required=True, help="Output directory for listening A/B files.")
    ap.add_argument(
        "--mode",
        choices=["copy", "link"],
        default="copy",
        help="copy = duplicate wavs; link = symlink to save space.",
    )
    ap.add_argument(
        "--pattern",
        default="**/chunk_0000.wav",
        help="Glob pattern relative to --src (default: **/chunk_0000.wav).",
    )
    ap.add_argument("--prefix", default="", help="Optional filename prefix (e.g. dream_).")
    ap.add_argument("--spk_id", default="", help="Optional speaker id override for output filenames (e.g. dream).")
    ap.add_argument(
        "--assume_speed",
        type=float,
        default=0.0,
        help="If run.json is missing, optionally assume this speed and append it to the filename (e.g. 1.0).",
    )
    ap.add_argument(
        "--include_params",
        action="store_true",
        default=True,
        help="Append sampling/seed/split tags derived from run.json to the filename (default: enabled).",
    )
    ap.add_argument(
        "--no-include_params",
        action="store_false",
        dest="include_params",
        help="Disable appending parameter tags; keep only <spk>__<variant>__full.wav style naming.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"--src not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    # Avoid re-collecting from a destination folder inside src.
    dst_resolved = dst.resolve()

    seen_real: set[str] = set()
    manifest_rows: list[dict[str, str]] = []

    for wav_path in sorted(src.glob(args.pattern)):
        if not wav_path.is_file():
            continue
        try:
            # Skip anything under dst if dst is inside src.
            if dst_resolved in wav_path.resolve().parents:
                continue
        except Exception:
            pass

        out_dir = wav_path.parent
        # De-dupe symlinks / aliases: many A/B dirs are symlinks to the same target.
        real_full = os.path.realpath(wav_path)
        if real_full in seen_real:
            continue
        seen_real.add(real_full)

        run = _read_run_json(out_dir) or {}
        spk_id = _safe_name(str(args.spk_id).strip() or str(run.get("spk_id") or "spk"))
        variant = _safe_name(out_dir.name)
        prefix = _safe_name(str(args.prefix))

        speed = None
        try:
            if "speed" in run:
                speed = float(run["speed"])
        except Exception:
            speed = None
        if speed is None and float(args.assume_speed) > 0:
            speed = float(args.assume_speed)

        tags: list[str] = []

        if speed is not None:
            tags.append(_fmt_float_tag("speed", speed, 100, 3))

        if bool(args.include_params):
            # Prefer explicit keys written by infer_sft.py.
            tp = run.get("llm_sampling_top_p", run.get("top_p"))
            tk = run.get("llm_sampling_top_k", run.get("top_k"))
            tr = run.get("llm_sampling_tau_r", run.get("tau_r"))
            temp = run.get("llm_sampling_temperature", run.get("temperature"))
            seed = run.get("seed")
            no_split = run.get("no_split")
            split_mode = run.get("split_mode")
            text_frontend = run.get("text_frontend")

            if temp is not None:
                tags.append(_fmt_float_tag("temp", float(temp), 100, 3))
            if tp is not None:
                tags.append(_fmt_float_tag("tp", float(tp), 100, 3))
            if tk is not None:
                tags.append(_fmt_int_tag("tk", int(tk), 3))
            if tr is not None:
                tags.append(_fmt_float_tag("tr", float(tr), 100, 3))
            if seed is not None:
                tags.append(_fmt_int_tag("seed", int(seed), 4))
            if no_split is not None:
                tags.append("nosplit" if bool(no_split) else "split")
            elif split_mode is not None:
                tags.append("nosplit" if str(split_mode).strip().lower() == "none" else "split")
            if text_frontend is not None:
                tags.append("wetext1" if bool(text_frontend) else "wetext0")

        # Avoid obvious duplicate tags if variant already includes them.
        variant_l = variant.lower()
        tags = [t for t in tags if t.lower() not in variant_l]
        tag_blob = ("__" + "__".join(tags)) if tags else ""

        out_name = f"{prefix}{spk_id}__{variant}{tag_blob}__{wav_path.name}"
        out_path = dst / out_name

        if out_path.exists():
            # If user re-runs, keep deterministic filenames and avoid accidental overwrite.
            # Append a numeric suffix.
            stem = out_path.stem
            for i in range(1, 10000):
                cand = dst / f"{stem}__{i:04d}{out_path.suffix}"
                if not cand.exists():
                    out_path = cand
                    break

        if args.mode == "link":
            # Relative symlink makes moving the folder easier.
            rel = os.path.relpath(real_full, start=str(out_path.parent))
            os.symlink(rel, out_path)
        else:
            shutil.copy2(real_full, out_path)

        sec = _wav_duration_sec(Path(real_full))
        manifest_rows.append(
            {
                "file": out_path.name,
                "seconds": f"{sec:.3f}",
                "src_full_wav": str(wav_path),
                "src_realpath": str(real_full),
                "out_dir": str(out_dir),
            }
        )

    manifest = dst / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "seconds", "src_full_wav", "src_realpath", "out_dir"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"[collect] wrote {len(manifest_rows)} wav(s) into: {dst}")
    print(f"[collect] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
