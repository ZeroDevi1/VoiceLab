from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Common audio extensions seen across datasets in this workspace.
# Keep this conservative; callers can extend if needed.
AUDIO_EXTS: set[str] = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
}


@dataclass(frozen=True)
class ListRow:
    audio: str
    speaker: str | None
    lang: str | None
    text: str | None
    raw_line: str
    lineno: int


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1].strip()
    return s


def parse_list(path: Path) -> list[ListRow]:
    """
    Parse GPT-SoVITS-style annotation `.list` file:
      audio_path|speaker|lang|text

    Tolerant parsing rules:
    - skip empty lines and comment lines starting with '#'
    - if fewer than 4 fields, missing fields become None
    - if more than 4 fields, the extra '|' are kept inside text
    """
    rows: list[ListRow] = []
    txt = path.read_text(encoding="utf-8")
    for i, raw in enumerate(txt.splitlines(), 1):
        line = raw.strip()
        if not line or line.lstrip().startswith("#"):
            continue

        # Remove accidental BOM on the first non-empty line.
        if line and line[0] == "\ufeff":
            line = line.lstrip("\ufeff")

        parts = line.split("|")
        audio = _strip_quotes(parts[0]) if parts else ""
        speaker = _strip_quotes(parts[1]) if len(parts) >= 2 and parts[1].strip() else None
        lang = _strip_quotes(parts[2]) if len(parts) >= 3 and parts[2].strip() else None
        text = None
        if len(parts) >= 4:
            text_joined = "|".join(parts[3:]).strip()
            text = _strip_quotes(text_joined) if text_joined else None

        rows.append(
            ListRow(
                audio=audio,
                speaker=speaker,
                lang=lang,
                text=text,
                raw_line=raw,
                lineno=i,
            )
        )
    return rows


def find_same_name_list(dataset_dir: Path) -> Path | None:
    """
    Prefer <dataset_dir>/<dataset_dir.name>.list (case-preserving),
    then <dataset_dir>/<dataset_dir.name.lower()>.list, then fallback to
    "the only *.list file" if the directory contains exactly one.
    """
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return None

    name = dataset_dir.name
    candidates = [
        dataset_dir / f"{name}.list",
        dataset_dir / f"{name.lower()}.list",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    lists: list[Path] = []
    for p in dataset_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".list":
            lists.append(p)
    if len(lists) == 1:
        return lists[0]
    return None


def resolve_audio_for_dataset(audio_field: str, dataset_dir: Path) -> Path | None:
    """
    Resolve an audio reference from a .list row into an existing local file.

    Priority:
    1) dataset_dir / basename(audio_field)
    2) dataset_dir / (stem(basename) + ext) for ext in AUDIO_EXTS (ordered by preference)
    3) if audio_field is absolute and exists, use it
    4) if audio_field is relative and exists under dataset_dir, use it
    """
    s = _strip_quotes(str(audio_field or "").strip())
    if not s:
        return None

    p = Path(s)
    basename = p.name
    if basename:
        direct = dataset_dir / basename
        if direct.exists():
            return direct

        stem = Path(basename).stem
        # Use a stable preference order.
        for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"]:
            cand = dataset_dir / f"{stem}{ext}"
            if cand.exists():
                return cand

    try:
        if p.is_absolute() and p.exists():
            return p
    except Exception:
        # Some malformed paths may fail is_absolute/exists on certain platforms.
        pass

    # Relative path fallback
    if not os.path.isabs(s):
        rel = dataset_dir / p
        if rel.exists():
            return rel

    return None

