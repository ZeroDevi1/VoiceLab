#!/usr/bin/env python3
"""
Generate GPT-SoVITS-style annotation `.list` files from CosyVoice `metadata.jsonl`.

Expected output line format:
  <rel_audio_path>|<speaker>|<lang>|<text>

Example:
  ./raw/Dream/028f....wav|Dream|ZH|你太入迷了
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_WS_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    # Preserve readability while keeping the `|` separator safe.
    text = text.replace("|", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    return _WS_RE.sub(" ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to CosyVoice metadata.jsonl (one JSON object per line).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .list file path.",
    )
    ap.add_argument(
        "--speaker",
        type=str,
        required=True,
        help="Speaker name (2nd field).",
    )
    ap.add_argument(
        "--lang",
        type=str,
        default="ZH",
        help="Language tag (3rd field). Default: ZH",
    )
    ap.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Raw audio directory prefix used in the list (1st field). Example: ./raw/Dream",
    )
    ap.add_argument(
        "--ext",
        type=str,
        default=".wav",
        help="Audio extension used in the list. Default: .wav",
    )

    args = ap.parse_args()

    if not args.ext.startswith("."):
        args.ext = "." + args.ext

    lines: list[str] = []
    for idx, raw in enumerate(args.metadata.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSONL at line {idx}: {e}") from e

        utt = str(obj.get("utt") or "").strip()
        if not utt:
            raise SystemExit(f"Missing 'utt' at line {idx}")

        text = obj.get("text") or obj.get("text_raw") or ""
        text = _clean_text(str(text))

        rel_audio = f"{args.raw_dir.rstrip('/')}/{utt}{args.ext}"
        lines.append(f"{rel_audio}|{args.speaker}|{args.lang}|{text}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {len(lines)} line(s): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

