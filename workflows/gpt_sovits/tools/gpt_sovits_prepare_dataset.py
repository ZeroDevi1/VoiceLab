from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from voicelab_bootstrap import data_root, gpt_sovits_vendor_root, runtime_root
from voicelab.list_annotations import AUDIO_EXTS, find_same_name_list, parse_list, resolve_audio_for_dataset


def _clean_text_field(text: str) -> str:
    # Keep the `|` separator safe for `.list`.
    return str(text).replace("|", " ").replace("\r", " ").replace("\n", " ").strip()


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _pick_list_path(dataset_dir: Path, annotation_dir: Path, explicit: str) -> Path:
    if explicit.strip():
        p = Path(explicit).expanduser()
        if not p.exists():
            raise SystemExit(f"[gpt_sovits] list not found: {p}")
        return p

    p = find_same_name_list(dataset_dir)
    if p is not None:
        return p

    # Centralized fallback
    cand1 = annotation_dir / f"{dataset_dir.name}.list"
    cand2 = annotation_dir / f"{dataset_dir.name.lower()}.list"
    for c in (cand1, cand2):
        if c.exists():
            return c
    raise SystemExit(
        "[gpt_sovits] no .list found.\n"
        f"  dataset_dir={dataset_dir}\n"
        f"  tried={cand1} / {cand2}\n"
        "Pass --list explicitly, or put a same-name .list into the dataset directory."
    )


def _build_effective_list_and_wav_dir(
    *,
    dataset_dir: Path,
    list_path: Path,
    out_dir: Path,
) -> tuple[Path, Path, dict[str, int]]:
    """
    Create:
    - an effective list under out_dir that uses basenames only (no absolute paths)
    - a wav input dir under out_dir containing symlinks/copies named exactly as list basenames

    This makes vendor scripts prefer the fast local dataset dir, while staying compatible
    with original `.list` that may contain absolute `/mnt/c/...` paths.
    """
    rows = parse_list(list_path)
    wav_inp_dir = out_dir / "_wav_input"
    _ensure_empty_dir(wav_inp_dir)

    effective_lines: list[str] = []
    selected = 0
    missing_audio = 0
    missing_text = 0

    seen_names: set[str] = set()
    for row in rows:
        audio_field = (row.audio or "").strip()
        if not audio_field:
            continue
        if row.text is None or not str(row.text).strip():
            missing_text += 1
            continue

        resolved = resolve_audio_for_dataset(audio_field, dataset_dir)
        if resolved is None or not resolved.exists() or resolved.suffix.lower() not in AUDIO_EXTS:
            missing_audio += 1
            continue

        name = Path(audio_field).name
        if not name:
            missing_audio += 1
            continue

        if name in seen_names:
            # Duplicate entry; keep first.
            continue
        seen_names.add(name)

        dst = wav_inp_dir / name
        try:
            os.symlink(str(resolved.resolve()), str(dst))
        except Exception:
            shutil.copy2(resolved, dst)

        spk = (row.speaker or "").strip() or "spk"
        lang = (row.lang or "").strip() or "ZH"
        text = _clean_text_field(row.text)
        effective_lines.append(f"{name}|{spk}|{lang}|{text}")
        selected += 1

    eff_list = out_dir / "_effective.list"
    eff_list.write_text("\n".join(effective_lines) + ("\n" if effective_lines else ""), encoding="utf-8")

    return eff_list, wav_inp_dir, {"selected": selected, "missing_audio": missing_audio, "missing_text": missing_text}


def _run_vendor(script_rel: str, *, vendor_root: Path, env: dict[str, str]) -> None:
    script = vendor_root / script_rel
    if not script.exists():
        raise SystemExit(f"[gpt_sovits] vendor script not found: {script}")
    cmd = [sys.executable, str(script)]
    print("[gpt_sovits] $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(vendor_root), env=env, check=True)


def _merge_parts_text(opt_dir: Path, parts: int) -> None:
    merged: list[str] = []
    for i in range(parts):
        p = opt_dir / f"2-name2text-{i}.txt"
        if not p.exists():
            continue
        merged += p.read_text(encoding="utf-8").strip("\n").split("\n")
        p.unlink(missing_ok=True)
    out = opt_dir / "2-name2text.txt"
    out.write_text("\n".join([x for x in merged if x.strip()]) + "\n", encoding="utf-8")


def _merge_parts_semantic(opt_dir: Path, parts: int) -> None:
    merged: list[str] = ["item_name\tsemantic_audio"]
    for i in range(parts):
        p = opt_dir / f"6-name2semantic-{i}.tsv"
        if not p.exists():
            continue
        merged += p.read_text(encoding="utf-8").strip("\n").split("\n")
        p.unlink(missing_ok=True)
    out = opt_dir / "6-name2semantic.tsv"
    out.write_text("\n".join([x for x in merged if x.strip()]) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare GPT-SoVITS dataset using a same-name .list (preferred).")
    ap.add_argument("--dataset-dir", required=True, help="Dataset dir containing audio files.")
    ap.add_argument("--exp-name", default="", help="Experiment name (default: dataset dir name).")
    ap.add_argument("--list", default="", help="Optional .list path. If empty, auto-detect same-name list.")
    ap.add_argument(
        "--annotation-dir",
        default="/mnt/c/AIGC/数据集/标注文件",
        help="Centralized list dir used when dataset-dir has no same-name .list.",
    )
    ap.add_argument("--parts", type=int, default=1, help="Parallel parts (matches vendor all_parts). Default: 1")
    ap.add_argument(
        "--gpu-numbers",
        default="0",
        help="GPU numbers split by '-' (e.g. '0-1'). Used to set _CUDA_VISIBLE_DEVICES per part.",
    )
    ap.add_argument("--is-half", action="store_true", default=True, help="Use fp16 where supported (default: enabled).")
    ap.add_argument("--no-is-half", action="store_false", dest="is_half", help="Disable fp16.")
    ap.add_argument(
        "--version",
        default="v2",
        help="GPT-SoVITS version tag used by vendor scripts (sets env var 'version'). Default: v2",
    )

    ap.add_argument(
        "--bert-pretrained-dir",
        default="",
        help="BERT dir for 1-get-text.py (default: vendor pretrained_models/chinese-roberta-wwm-ext-large).",
    )
    ap.add_argument(
        "--cnhubert-base-dir",
        default="",
        help="CN-HuBERT dir for 2-get-hubert-wav32k.py (default: vendor pretrained_models/chinese-hubert-base).",
    )
    ap.add_argument(
        "--pretrained-s2g",
        default="",
        help="SoVITS pretrained generator weights for 3-get-semantic.py (default: vendor pretrained_models/s2G488k.pth).",
    )
    ap.add_argument(
        "--s2config-path",
        default="",
        help="SoVITS s2 config json path (default: vendor GPT_SoVITS/configs/s2.json).",
    )
    args = ap.parse_args()

    vendor = gpt_sovits_vendor_root()
    if not vendor.exists():
        raise SystemExit(f"[gpt_sovits] vendor not found: {vendor} (run: uv run -m voicelab vendor sync)")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise SystemExit(f"[gpt_sovits] dataset-dir not found: {dataset_dir}")

    annotation_dir = Path(args.annotation_dir).expanduser().resolve()
    list_path = _pick_list_path(dataset_dir, annotation_dir, args.list)

    exp_name = args.exp_name.strip() or dataset_dir.name
    opt_dir = (data_root() / exp_name).resolve()
    opt_dir.mkdir(parents=True, exist_ok=True)

    eff_list, wav_inp_dir, stats = _build_effective_list_and_wav_dir(
        dataset_dir=dataset_dir,
        list_path=list_path,
        out_dir=opt_dir,
    )
    print(
        f"[gpt_sovits] list: {list_path}\n"
        f"[gpt_sovits] effective_list: {eff_list}\n"
        f"[gpt_sovits] wav_input_dir: {wav_inp_dir}\n"
        f"[gpt_sovits] selected={stats['selected']} missing_audio={stats['missing_audio']} missing_text={stats['missing_text']}"
    )
    if stats["selected"] <= 0:
        raise SystemExit("[gpt_sovits] no usable rows in list (missing audio/text).")

    parts = max(1, int(args.parts))
    gpu_names = [x for x in str(args.gpu_numbers).split("-") if x.strip()]
    if len(gpu_names) < parts:
        gpu_names = (gpu_names * (parts // max(1, len(gpu_names)) + 1))[:parts]

    bert_dir = (
        Path(args.bert_pretrained_dir).expanduser()
        if str(args.bert_pretrained_dir).strip()
        else (vendor / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large")
    ).resolve()
    cnhubert_dir = (
        Path(args.cnhubert_base_dir).expanduser()
        if str(args.cnhubert_base_dir).strip()
        else (vendor / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base")
    ).resolve()
    pretrained_s2g = (
        Path(args.pretrained_s2g).expanduser()
        if str(args.pretrained_s2g).strip()
        else (vendor / "GPT_SoVITS" / "pretrained_models" / "s2G488k.pth")
    ).resolve()
    s2config = (
        Path(args.s2config_path).expanduser()
        if str(args.s2config_path).strip()
        else (vendor / "GPT_SoVITS" / "configs" / "s2.json")
    ).resolve()

    # 1) text -> 2-name2text.txt (+ bert)
    for i in range(parts):
        env = os.environ.copy()
        env.update(
            {
                "inp_text": str(eff_list),
                "inp_wav_dir": str(wav_inp_dir),
                "exp_name": exp_name,
                "i_part": str(i),
                "all_parts": str(parts),
                "opt_dir": str(opt_dir),
                "bert_pretrained_dir": str(bert_dir),
                "is_half": "True" if args.is_half else "False",
                "version": str(args.version),
                "hz": "25hz",
                "_CUDA_VISIBLE_DEVICES": str(gpu_names[i]),
            }
        )
        _run_vendor("GPT_SoVITS/prepare_datasets/1-get-text.py", vendor_root=vendor, env=env)
    _merge_parts_text(opt_dir, parts)

    # 2) hubert + wav32k
    for i in range(parts):
        env = os.environ.copy()
        env.update(
            {
                "inp_text": str(eff_list),
                "inp_wav_dir": str(wav_inp_dir),
                "exp_name": exp_name,
                "i_part": str(i),
                "all_parts": str(parts),
                "opt_dir": str(opt_dir),
                "cnhubert_base_dir": str(cnhubert_dir),
                "is_half": "True" if args.is_half else "False",
                "version": str(args.version),
                "_CUDA_VISIBLE_DEVICES": str(gpu_names[i]),
            }
        )
        _run_vendor("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py", vendor_root=vendor, env=env)

    # 3) semantic tokens -> 6-name2semantic.tsv
    for i in range(parts):
        env = os.environ.copy()
        env.update(
            {
                "inp_text": str(eff_list),
                "exp_name": exp_name,
                "i_part": str(i),
                "all_parts": str(parts),
                "opt_dir": str(opt_dir),
                "pretrained_s2G": str(pretrained_s2g),
                "s2config_path": str(s2config),
                "is_half": "True" if args.is_half else "False",
                "version": str(args.version),
                "_CUDA_VISIBLE_DEVICES": str(gpu_names[i]),
            }
        )
        _run_vendor("GPT_SoVITS/prepare_datasets/3-get-semantic.py", vendor_root=vendor, env=env)
    _merge_parts_semantic(opt_dir, parts)

    # Mirror config outputs in runtime/ for convenience (the actual dataset lives in data/).
    cfg_dir = (runtime_root() / exp_name / "configs").resolve()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(eff_list, cfg_dir / "effective.list")

    print("")
    print("[gpt_sovits] prepared dataset dir:", opt_dir)
    print("[gpt_sovits] expected files:")
    print(f"  - {opt_dir / '2-name2text.txt'}")
    print(f"  - {opt_dir / '4-cnhubert'}")
    print(f"  - {opt_dir / '5-wav32k'}")
    print(f"  - {opt_dir / '6-name2semantic.tsv'}")
    print("")
    print("[gpt_sovits] next:")
    print(f"  uv run python tools/gpt_sovits_train_s1.py --exp-name {exp_name}")
    print(f"  uv run python tools/gpt_sovits_train_s2.py --exp-name {exp_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
