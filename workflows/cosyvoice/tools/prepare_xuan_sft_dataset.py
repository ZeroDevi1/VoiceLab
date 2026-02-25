import argparse
import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from voicelab_bootstrap import ensure_sys_path

ensure_sys_path()

from voicelab.list_annotations import find_same_name_list, parse_list  # noqa: E402

INSTRUCT_DEFAULT = "You are a helpful assistant.<|endofprompt|>"


def _read_text_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text_utf8(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _append_jsonl_utf8(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_existing_utts(metadata_jsonl: Path) -> set[str]:
    if not metadata_jsonl.exists():
        return set()
    utts: set[str] = set()
    with metadata_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            utt = obj.get("utt")
            if isinstance(utt, str) and utt:
                utts.add(utt)
    return utts


def _simple_cleanup(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return text


class _TextNormalizer:
    def __init__(self, use_wetext: bool):
        self._use_wetext = use_wetext
        self._zh = None

        if not use_wetext:
            return
        try:
            from wetext import Normalizer  # type: ignore

            self._zh = Normalizer(remove_erhua=False)
        except Exception:
            self._zh = None

    def normalize(self, text_raw: str) -> str:
        return _normalize_text(text_raw, self._zh)


def _normalize_text(text_raw: str, zh_normalizer) -> str:
    text_raw = _simple_cleanup(text_raw)
    if not text_raw:
        return ""

    if zh_normalizer is not None:
        try:
            text = zh_normalizer.normalize(text_raw)
        except Exception:
            text = text_raw
    else:
        text = text_raw

    text = _simple_cleanup(text)
    return text


def _iter_wavs(wav_dir: Path) -> Iterable[Path]:
    # Keep deterministic order
    yield from sorted(wav_dir.glob("*.wav"))


def _utt_from_path(wav_path: Path) -> str:
    return wav_path.stem


def _load_audio_16k_mono(wav_path: Path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # audio: (n, ch)
    audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype("float32"), sr


def _audio_duration_sec(wav_path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(wav_path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


class _WhisperTranscriber:
    def __init__(self, whisper_model: str, device: str, language: str, batch_size: int, beam_size: int):
        import whisper

        self._device = device
        self._language = language
        self._batch_size = batch_size
        self._beam_size = beam_size
        self._model = whisper.load_model(whisper_model, device=device)

    def transcribe(self, audio_16k) -> str:
        fp16 = self._device.lower() == "cuda"
        result = self._model.transcribe(
            audio_16k,
            language=self._language,
            task="transcribe",
            fp16=fp16,
            batch_size=self._batch_size,
            beam_size=self._beam_size,
        )
        text = result.get("text", "")
        return text if isinstance(text, str) else ""


class _FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        device_index: int,
        compute_type: str,
        language: str,
        beam_size: int,
        vad_filter: bool,
        download_root: str | None,
        local_files_only: bool,
    ):
        from faster_whisper import WhisperModel  # type: ignore

        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        try:
            self._model = WhisperModel(
                model_name,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                download_root=download_root,
                local_files_only=local_files_only,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "Cannot load the target vocabulary from the model directory" in msg:
                cache_hint = download_root or "~/.cache/huggingface/hub"
                print(
                    "[prepare_xuan] faster-whisper model load failed: cache likely corrupted/incomplete.\n"
                    f"[prepare_xuan] Hint: delete the cached model under {cache_hint} "
                    "(e.g. models--Systran--faster-whisper-*) and re-run to re-download."
                )
            raise

    def transcribe(self, audio_16k) -> str:
        segments, _info = self._model.transcribe(
            audio_16k,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
        )
        parts: list[str] = []
        for seg in segments:
            text = getattr(seg, "text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return " ".join(parts).strip()


@dataclass(frozen=True)
class Args:
    wav_dir: Path
    out_root: Path
    spk_id: str
    list_path: Path | None
    prefer_list: bool
    list_required_text: bool
    backend: str
    whisper_model: str
    device: str
    device_index: int
    compute_type: str
    language: str
    train_ratio: float
    seed: int
    instruct: str
    max_sec: float
    overwrite: bool
    limit: int
    use_wetext: bool
    whisper_batch_size: int
    whisper_beam_size: int
    vad_filter: bool
    faster_whisper_download_root: str
    faster_whisper_local_files_only: bool


def _write_kaldi_dir(out_dir: Path, items: list[dict], spk_id: str, instruct: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_scp_lines: list[str] = []
    text_lines: list[str] = []
    utt2spk_lines: list[str] = []
    instruct_lines: list[str] = []
    utts: list[str] = []

    for it in items:
        utt = it["utt"]
        wav = it["wav"]
        text = it["text"]
        wav_scp_lines.append(f"{utt} {wav}")
        text_lines.append(f"{utt} {text}")
        utt2spk_lines.append(f"{utt} {spk_id}")
        instruct_lines.append(f"{utt} {instruct}")
        utts.append(utt)

    spk2utt_line = f"{spk_id} " + " ".join(utts) if utts else f"{spk_id}"

    _write_text_utf8(out_dir / "wav.scp", "\n".join(wav_scp_lines) + ("\n" if wav_scp_lines else ""))
    _write_text_utf8(out_dir / "text", "\n".join(text_lines) + ("\n" if text_lines else ""))
    _write_text_utf8(out_dir / "utt2spk", "\n".join(utt2spk_lines) + ("\n" if utt2spk_lines else ""))
    _write_text_utf8(out_dir / "spk2utt", spk2utt_line + "\n")
    _write_text_utf8(out_dir / "instruct", "\n".join(instruct_lines) + ("\n" if instruct_lines else ""))


def run(args: Args) -> int:
    args.out_root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_root / "metadata.jsonl"
    if args.overwrite and metadata_path.exists():
        metadata_path.unlink()

    existing_utts = set()
    if not args.overwrite:
        existing_utts = _load_existing_utts(metadata_path)

    wavs = list(_iter_wavs(args.wav_dir))
    if args.limit and args.limit > 0:
        wavs = wavs[: args.limit]
    if not wavs:
        print(f"[prepare_xuan] No .wav found in {args.wav_dir}")
        return 2

    # Optional: load text annotations from `.list` (audio|speaker|lang|text).
    # We map by basename and stem so list entries can be absolute paths or different extensions.
    text_by_basename: dict[str, str] = {}
    text_by_stem: dict[str, str] = {}
    list_path: Path | None = args.list_path
    if list_path is None and args.prefer_list:
        list_path = find_same_name_list(args.wav_dir) or find_same_name_list(args.wav_dir.parent)
    if list_path is not None and list_path.exists():
        try:
            rows = parse_list(list_path)
            for row in rows:
                if not row.audio:
                    continue
                if row.text is None or not str(row.text).strip():
                    # Keep "missing text" out of the map; we'll fallback to ASR when needed.
                    continue
                base = Path(row.audio).name
                stem = Path(row.audio).stem
                text_by_basename.setdefault(base, str(row.text).strip())
                text_by_stem.setdefault(stem, str(row.text).strip())
            print(f"[prepare_xuan] list text enabled: {list_path} (rows={len(rows)} mapped={len(text_by_stem)})")
        except Exception as exc:
            raise SystemExit(f"[prepare_xuan] failed to parse list: {list_path}: {exc}") from exc

    text_normalizer = _TextNormalizer(use_wetext=args.use_wetext)

    # Lazily construct ASR transcriber only when we actually need ASR.
    # This allows "list text only" runs to avoid installing ASR extras.
    transcriber = None

    def get_transcriber():
        nonlocal transcriber
        if transcriber is not None:
            return transcriber
        if args.backend == "openai-whisper":
            transcriber = _WhisperTranscriber(
                whisper_model=args.whisper_model,
                device=args.device,
                language=args.language,
                batch_size=args.whisper_batch_size,
                beam_size=args.whisper_beam_size,
            )
        elif args.backend == "faster-whisper":
            transcriber = _FasterWhisperTranscriber(
                model_name=args.whisper_model,
                device=args.device,
                device_index=args.device_index,
                compute_type=args.compute_type,
                language=args.language,
                beam_size=args.whisper_beam_size,
                vad_filter=args.vad_filter,
                download_root=(args.faster_whisper_download_root or None),
                local_files_only=bool(args.faster_whisper_local_files_only),
            )
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
        return transcriber

    # Build transcription list
    records: list[dict] = []
    for wav_path in tqdm(wavs, desc="[prepare_xuan] transcribe", unit="wav"):
        utt = _utt_from_path(wav_path)
        if utt in existing_utts:
            continue
        dur = _audio_duration_sec(wav_path)
        if args.max_sec > 0 and dur > args.max_sec:
            print(f"[prepare_xuan] Skip {wav_path.name}: duration {dur:.2f}s > {args.max_sec:.2f}s")
            continue

        text_src = "asr"
        text_raw = ""

        # Prefer list text if available.
        if text_by_basename or text_by_stem:
            text_raw = text_by_basename.get(wav_path.name) or text_by_stem.get(wav_path.stem) or ""
            if text_raw.strip():
                text_src = "list"
            elif args.list_required_text:
                # Explicitly fallback to ASR if list is enabled but lacks a usable text for this item.
                text_src = "asr"
                text_raw = ""

        if text_src == "asr":
            try:
                audio_16k, sr = _load_audio_16k_mono(wav_path)
                if sr != 16000:
                    print(f"[prepare_xuan] WARN: {wav_path.name} resample failed to 16k (sr={sr})")
            except Exception as exc:
                print(f"[prepare_xuan] Failed to load {wav_path}: {exc}")
                continue

            try:
                text_raw = get_transcriber().transcribe(audio_16k)
            except Exception as exc:
                print(f"[prepare_xuan] Whisper failed for {wav_path.name}: {exc}")
                continue

        text = text_normalizer.normalize(text_raw)
        if not text:
            print(f"[prepare_xuan] Skip {wav_path.name}: empty text after normalization")
            continue

        rec = {
            "utt": utt,
            "wav": str(wav_path.resolve()),
            "spk": args.spk_id,
            "text_raw": text_raw,
            "text": text,
            "text_src": text_src,
        }
        _append_jsonl_utf8(metadata_path, rec)
        records.append(rec)

    if not records and not metadata_path.exists():
        print("[prepare_xuan] No records produced.")
        return 3

    # Load metadata (so split is based on the full set, including previous runs)
    # De-duplicate by utt to avoid accidental duplicates when resuming.
    all_records_by_utt: dict[str, dict] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("utt") and obj.get("wav") and obj.get("text"):
                    all_records_by_utt[str(obj["utt"])] = obj

    all_records = [all_records_by_utt[k] for k in sorted(all_records_by_utt.keys())]

    # Stable deterministic split by utt hash (adding new records won't reshuffle old ones)
    def is_train(utt: str) -> bool:
        h = hashlib.md5(f"{args.seed}:{utt}".encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        return bucket < args.train_ratio

    train_records = [r for r in all_records if is_train(str(r["utt"]))]
    dev_records = [r for r in all_records if not is_train(str(r["utt"]))]
    if len(all_records) >= 2 and not dev_records:
        dev_records = [train_records.pop()]

    _write_kaldi_dir(args.out_root / "train", train_records, args.spk_id, args.instruct)
    _write_kaldi_dir(args.out_root / "dev", dev_records, args.spk_id, args.instruct)

    print(f"[prepare_xuan] Total={len(all_records)} train={len(train_records)} dev={len(dev_records)}")
    print(f"[prepare_xuan] Wrote {metadata_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare xuan dataset for CosyVoice3 SFT (Whisper + kaldi + instruct).")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing .wav files.")
    parser.add_argument("--out_root", type=str, default="data/xuan_sft", help="Output root directory under repo.")
    parser.add_argument("--spk_id", type=str, default="xuan", help="Speaker id for utt2spk/spk2utt.")
    parser.add_argument("--list", type=str, default="", help="Optional .list annotation file (audio|speaker|lang|text).")
    parser.add_argument(
        "--prefer-list",
        action="store_true",
        default=True,
        help="Prefer same-name <wav_dir>/<wav_dir>.list or <parent>/<parent>.list if present (default: enabled).",
    )
    parser.add_argument(
        "--no-prefer-list",
        action="store_false",
        dest="prefer_list",
        help="Disable auto-detecting same-name list file.",
    )
    parser.add_argument(
        "--list-required-text",
        action="store_true",
        default=True,
        help="When list is enabled, require non-empty text; fallback to ASR if missing (default: enabled).",
    )
    parser.add_argument(
        "--no-list-required-text",
        action="store_false",
        dest="list_required_text",
        help="If list is enabled and text is missing, do not fallback to ASR (will likely skip item).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai-whisper", "faster-whisper"],
        default="openai-whisper",
        help="ASR backend. faster-whisper is recommended on 6GB GPUs (INT8 + optional VAD).",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="medium",
        help=(
            "Model name. For openai-whisper: tiny/base/small/medium/large/large-v3. "
            "For faster-whisper: large-v3 recommended (downloads CTranslate2 model)."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu", help="Whisper device: cpu or cuda.")
    parser.add_argument("--device_index", type=int, default=0, help="GPU device index for faster-whisper.")
    parser.add_argument(
        "--compute_type",
        type=str,
        default="",
        help="faster-whisper compute_type, e.g. int8_float16/float16/int8. Default auto based on device.",
    )
    parser.add_argument("--language", type=str, default="zh", help="Language code for Whisper.")
    parser.add_argument("--train_ratio", type=float, default=0.98, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=1986, help="Random seed for splitting.")
    parser.add_argument("--instruct", type=str, default=INSTRUCT_DEFAULT, help="Instruct string (CosyVoice3LM requires it).")
    parser.add_argument("--max_sec", type=float, default=30.0, help="Skip audio longer than this (<=0 to disable).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite metadata.jsonl (disables resume).")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N wavs (0 means no limit).")
    parser.add_argument("--no_wetext", action="store_true", help="Disable wetext normalization (faster; less robust TN).")
    parser.add_argument("--whisper_batch_size", type=int, default=1, help="Whisper decoding batch_size.")
    parser.add_argument("--whisper_beam_size", type=int, default=1, help="Whisper beam search size (1 = greedy).")
    parser.add_argument("--vad_filter", action="store_true", help="Enable VAD filtering (faster-whisper only).")
    parser.add_argument(
        "--faster_whisper_download_root",
        type=str,
        default="",
        help="Cache dir for faster-whisper CTranslate2 models (default: Hugging Face cache).",
    )
    parser.add_argument(
        "--faster_whisper_local_files_only",
        action="store_true",
        help="Do not download faster-whisper models; only use local cache.",
    )
    ns = parser.parse_args()

    compute_type = str(ns.compute_type).strip()
    if not compute_type:
        compute_type = "int8_float16" if str(ns.device).lower() == "cuda" else "int8"

    args = Args(
        wav_dir=Path(ns.wav_dir),
        out_root=Path(ns.out_root),
        spk_id=str(ns.spk_id),
        list_path=(Path(ns.list).expanduser() if str(ns.list).strip() else None),
        prefer_list=bool(ns.prefer_list),
        list_required_text=bool(ns.list_required_text),
        backend=str(ns.backend),
        whisper_model=str(ns.whisper_model),
        device=str(ns.device),
        device_index=int(ns.device_index),
        compute_type=compute_type,
        language=str(ns.language),
        train_ratio=float(ns.train_ratio),
        seed=int(ns.seed),
        instruct=str(ns.instruct),
        max_sec=float(ns.max_sec),
        overwrite=bool(ns.overwrite),
        limit=int(ns.limit),
        use_wetext=not bool(ns.no_wetext),
        whisper_batch_size=int(ns.whisper_batch_size),
        whisper_beam_size=int(ns.whisper_beam_size),
        vad_filter=bool(ns.vad_filter),
        faster_whisper_download_root=str(ns.faster_whisper_download_root),
        faster_whisper_local_files_only=bool(ns.faster_whisper_local_files_only),
    )

    if not args.wav_dir.exists():
        print(f"[prepare_xuan] wav_dir not found: {args.wav_dir}")
        return 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
