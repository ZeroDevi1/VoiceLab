from __future__ import annotations

import argparse
from pathlib import Path

from voicelab_bootstrap import voicelab_root
from voicelab.dataset_stage import ensure_list_present, stage_dataset_rsync
from voicelab.list_annotations import find_same_name_list


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage a GPT-SoVITS dataset into a fast local path (ext4).")
    ap.add_argument("--src", required=True, help="Source dataset dir (often under /mnt/c/...).")
    ap.add_argument(
        "--dst",
        default=None,
        help="Destination dir (default: VoiceLab/datasets/<src_name>).",
    )
    ap.add_argument("--force", action="store_true", help="Make destination match source (rsync --delete).")
    ap.add_argument(
        "--annotation-dir",
        default="/mnt/c/AIGC/数据集/标注文件",
        help="Fallback directory containing centralized *.list annotation files.",
    )
    ap.add_argument(
        "--no-copy-list",
        action="store_true",
        help="Do not copy/ensure same-name *.list into destination dataset directory.",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = (
        Path(args.dst).expanduser()
        if args.dst
        else (voicelab_root() / "datasets" / src.name)
    )

    stage_dataset_rsync(src, dst, force=bool(args.force))
    if not args.no_copy_list:
        lp = ensure_list_present(src, dst, annotation_dir=Path(args.annotation_dir))
        if lp is not None:
            print(f"[gpt_sovits] list available: {lp}")
        else:
            print("[gpt_sovits] WARN: no same-name .list found/copied.")
    else:
        lp = find_same_name_list(dst)
        if lp is not None:
            print(f"[gpt_sovits] list present: {lp}")

    print("")
    print("[gpt_sovits] Next:")
    print(f"  uv run python tools/gpt_sovits_prepare_dataset.py --dataset-dir {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
