#!/usr/bin/env python3
"""
Rewrite GPT-SoVITS-style `.list` files so the audio path (1st field) points to a
target dataset root (e.g. /mnt/c/AIGC/数据集).

Input line format (common in this repo):
  <audio_path>|<speaker>|<lang>|<text...>

We only rewrite the first field, keeping the other fields intact.
If the original filename does not exist under the target speaker directory,
we will try to resolve by stem with a small extension whitelist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


EXT_PREFERENCE = [
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
]


@dataclass(frozen=True)
class RewriteRule:
    list_path: Path
    speaker_dir: Path


def _resolve_audio(speaker_dir: Path, original_audio_field: str) -> Path | None:
    p = Path(original_audio_field)
    basename = p.name
    direct = speaker_dir / basename
    if direct.exists():
        return direct

    stem = Path(basename).stem
    # Try a stem-based match with preferred extensions.
    for ext in EXT_PREFERENCE:
        cand = speaker_dir / f"{stem}{ext}"
        if cand.exists():
            return cand

    # Last resort: any match by stem.
    matches = list(speaker_dir.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    return None


def _rewrite_file(rule: RewriteRule) -> tuple[int, int]:
    src = rule.list_path.read_text(encoding="utf-8", errors="strict").splitlines()
    out_lines: list[str] = []

    changed = 0
    missing = 0
    for i, line in enumerate(src, 1):
        if not line.strip():
            out_lines.append(line)
            continue

        parts = line.split("|", 3)
        if len(parts) < 4:
            raise SystemExit(f"Invalid .list format: {rule.list_path} line {i}: {line!r}")

        audio_field, speaker, lang, text = parts
        resolved = _resolve_audio(rule.speaker_dir, audio_field)
        if resolved is None:
            missing += 1
            out_lines.append(line)
            continue

        new_audio = str(resolved)
        if new_audio != audio_field:
            changed += 1
        out_lines.append("|".join([new_audio, speaker, lang, text]))

    rule.list_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/c/AIGC/数据集"),
        help="Target dataset root. Default: /mnt/c/AIGC/数据集",
    )
    ap.add_argument(
        "--speaker",
        action="append",
        required=True,
        help=(
            "Speaker name. Used to locate list file(s) and speaker dir under dataset root. "
            "Example: --speaker XingTong --speaker Xuan"
        ),
    )
    ap.add_argument(
        "--list-subpath",
        action="append",
        default=["标注文件/{speaker}.list", "{speaker}/{speaker}.list", "{speaker}/{speaker_lc}.list"],
        help=(
            "Relative path template(s) to find list files. Available vars: {speaker}, {speaker_lc}. "
            "Default covers /标注文件 and per-speaker dirs."
        ),
    )
    ap.add_argument(
        "--speaker-dir",
        action="append",
        default=[],
        help=(
            "Optional override mapping in the form Speaker=DirName (case-sensitive dir under dataset root). "
            "Example: --speaker-dir Xuan=Xuan"
        ),
    )
    args = ap.parse_args()

    overrides: dict[str, str] = {}
    for item in args.speaker_dir:
        if "=" not in item:
            raise SystemExit(f"--speaker-dir must be Speaker=DirName, got: {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = v

    total_changed = 0
    total_missing = 0
    processed_files: list[Path] = []

    for speaker in args.speaker:
        spk_dir_name = overrides.get(speaker, speaker)
        speaker_dir = (args.dataset_root / spk_dir_name).resolve()
        if not speaker_dir.exists():
            raise SystemExit(f"Speaker dir not found: {speaker_dir}")

        ctx = {"speaker": speaker, "speaker_lc": speaker.lower()}
        list_files: list[Path] = []
        for tmpl in args.list_subpath:
            rel = tmpl.format(**ctx)
            p = (args.dataset_root / rel).resolve()
            if p.exists():
                list_files.append(p)

        if not list_files:
            raise SystemExit(f"No list files found for speaker={speaker!r} under dataset root={args.dataset_root}")

        for lp in list_files:
            changed, missing = _rewrite_file(RewriteRule(list_path=lp, speaker_dir=speaker_dir))
            total_changed += changed
            total_missing += missing
            processed_files.append(lp)
            print(f"[ok] {lp} changed_lines={changed} missing_audio={missing}")

    if total_missing:
        print(f"[warn] total missing audio references (left unchanged): {total_missing}")
        return 2

    print(f"[done] files={len(processed_files)} changed_lines={total_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

