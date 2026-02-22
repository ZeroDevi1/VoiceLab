import argparse
from pathlib import Path

from voicelab_bootstrap import ensure_sys_path


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

def _load_state_dict_maybe(path: Path) -> dict | None:
    if not path.exists():
        return None
    import torch

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        return None
    # Some training checkpoints mix weights with metadata.
    obj.pop("epoch", None)
    obj.pop("step", None)
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CosyVoice3 SFT inference and save wav outputs.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--spk_id", type=str, default="xuan")
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="You are a helpful assistant.<|endofprompt|>",
        help="CosyVoice3 requires <|endofprompt|> in prompt_text (recommended). Set empty string to disable.",
    )
    parser.add_argument(
        "--llm_ckpt",
        type=str,
        default="",
        help=(
            "Optional path to an LLM checkpoint (epoch_*_whole.pt / llm_avg.pt) to hot-load before inference. "
            "This lets you audition the best epoch without re-assembling model_dir."
        ),
    )
    parser.add_argument(
        "--flow_ckpt",
        type=str,
        default="",
        help=(
            "Optional path to a Flow checkpoint (epoch_*_whole.pt / flow_avg.pt / base flow.pt) to hot-load before inference. "
            "Useful for A/B testing: keep SFT LLM but fall back to the base flow for better clarity."
        ),
    )
    parser.add_argument("--out_dir", type=str, default="out_wav")
    parser.add_argument("--stream", action="store_true", help="Use chunk streaming inference.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Do not split text by punctuation/sentences; synthesize as one segment for better continuity.",
    )
    parser.add_argument(
        "--text_frontend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable text frontend normalization (wetext). Disable to avoid downloads / for quick tests.",
    )
    parser.add_argument("--concat", action="store_true", help="Also save concatenated full.wav (may use lots of RAM).")
    args = parser.parse_args()

    ensure_sys_path()

    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
    import torchaudio  # noqa: E402
    import torch  # noqa: E402

    text_path = Path(args.text_file)
    if not text_path.exists():
        print(f"[infer] text_file not found: {text_path}")
        return 2
    text = _read_text_file(text_path)
    if not text:
        print("[infer] text is empty")
        return 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cosyvoice = AutoModel(model_dir=args.model_dir)

    flow_ckpt = str(args.flow_ckpt).strip()
    if flow_ckpt:
        state = _load_state_dict_maybe(Path(flow_ckpt))
        if state is None:
            print(f"[infer] failed to load flow_ckpt as state_dict: {flow_ckpt}")
            return 8
        missing, unexpected = cosyvoice.model.flow.load_state_dict(state, strict=False)
        print(f"[infer] hot-loaded flow_ckpt: {flow_ckpt}")
        if missing:
            print(f"[infer] WARN: missing_keys={len(missing)}")
        if unexpected:
            print(f"[infer] WARN: unexpected_keys={len(unexpected)}")

    llm_ckpt = str(args.llm_ckpt).strip()
    if llm_ckpt:
        state = _load_state_dict_maybe(Path(llm_ckpt))
        if state is None:
            print(f"[infer] failed to load llm_ckpt as state_dict: {llm_ckpt}")
            return 6
        missing, unexpected = cosyvoice.model.llm.load_state_dict(state, strict=False)
        print(f"[infer] hot-loaded llm_ckpt: {llm_ckpt}")
        if missing:
            print(f"[infer] WARN: missing_keys={len(missing)}")
        if unexpected:
            print(f"[infer] WARN: unexpected_keys={len(unexpected)}")

    is_cosyvoice3 = getattr(getattr(cosyvoice.model, "llm", None), "__class__", type("x", (), {})).__name__ == "CosyVoice3LM"
    prompt_text_raw = str(args.prompt_text).strip()
    if is_cosyvoice3 and not prompt_text_raw:
        print("[infer] CosyVoice3 requires prompt_text containing <|endofprompt|>.")
        print("[infer] Pass --prompt_text 'You are a helpful assistant.<|endofprompt|>' (recommended).")
        return 5

    prompt_text_token = None
    prompt_text_token_len = None
    if is_cosyvoice3 and prompt_text_raw:
        prompt_text_norm = cosyvoice.frontend.text_normalize(prompt_text_raw, split=False, text_frontend=bool(args.text_frontend))
        prompt_text_token, prompt_text_token_len = cosyvoice.frontend._extract_text_token(prompt_text_norm)

    # NOTE: `frontend.text_normalize(..., split=False)` returns a *string* (not a list).
    # Iterating over it would synthesize per-character, producing lots of tiny chunks and a garbled full.wav.
    split = not bool(args.no_split)
    norm = cosyvoice.frontend.text_normalize(text, split=split, text_frontend=bool(args.text_frontend))
    if isinstance(norm, str):
        segments = [norm]
    elif isinstance(norm, (list, tuple)):
        segments = list(norm)
    else:
        segments = [norm]

    chunks = []
    idx = 0
    for seg in segments:
        model_input = cosyvoice.frontend.frontend_sft(seg, args.spk_id)
        if prompt_text_token is not None and prompt_text_token_len is not None:
            model_input["prompt_text"] = prompt_text_token
            model_input["prompt_text_len"] = prompt_text_token_len
        # In non-stream mode, cosyvoice yields a single full waveform for this segment.
        # In stream mode, it yields multiple non-overlapping waveform pieces which we need to concatenate.
        wav_pieces: list[torch.Tensor] = []
        for out in cosyvoice.model.tts(**model_input, stream=bool(args.stream), speed=float(args.speed)):
            wav = out["tts_speech"]
            if not isinstance(wav, torch.Tensor):
                print(f"[infer] unexpected tts_speech type: {type(wav)}")
                return 4
            wav_pieces.append(wav)
            if bool(args.stream):
                piece_path = out_dir / f"piece_{idx:04d}.wav"
                torchaudio.save(str(piece_path), wav, cosyvoice.sample_rate)
                print(f"[infer] wrote {piece_path}")
                idx += 1

        if not wav_pieces:
            print("[infer] no audio produced")
            return 7

        seg_wav = wav_pieces[0] if len(wav_pieces) == 1 else torch.cat(wav_pieces, dim=1)
        chunk_path = out_dir / f"chunk_{len(chunks):04d}.wav"
        torchaudio.save(str(chunk_path), seg_wav, cosyvoice.sample_rate)
        print(f"[infer] wrote {chunk_path}")
        if args.concat:
            chunks.append(seg_wav)

    if args.concat and chunks:
        full = torch.cat(chunks, dim=1)
        full_path = out_dir / "full.wav"
        torchaudio.save(str(full_path), full, cosyvoice.sample_rate)
        print(f"[infer] wrote {full_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
