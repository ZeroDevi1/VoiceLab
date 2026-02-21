import argparse
from pathlib import Path

from voicelab_bootstrap import ensure_sys_path


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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
    parser.add_argument("--out_dir", type=str, default="out_wav")
    parser.add_argument("--stream", action="store_true", help="Use chunk streaming inference.")
    parser.add_argument("--speed", type=float, default=1.0)
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
    is_cosyvoice3 = getattr(getattr(cosyvoice.model, "llm", None), "__class__", type("x", (), {})).__name__ == "CosyVoice3LM"
    prompt_text_raw = str(args.prompt_text).strip()
    if is_cosyvoice3 and not prompt_text_raw:
        print("[infer] CosyVoice3 requires prompt_text containing <|endofprompt|>.")
        print("[infer] Pass --prompt_text 'You are a helpful assistant.<|endofprompt|>' (recommended).")
        return 5

    prompt_text_token = None
    prompt_text_token_len = None
    if is_cosyvoice3 and prompt_text_raw:
        prompt_text_norm = cosyvoice.frontend.text_normalize(prompt_text_raw, split=False, text_frontend=True)
        prompt_text_token, prompt_text_token_len = cosyvoice.frontend._extract_text_token(prompt_text_norm)

    chunks = []
    idx = 0
    for seg in cosyvoice.frontend.text_normalize(text, split=True, text_frontend=True):
        model_input = cosyvoice.frontend.frontend_sft(seg, args.spk_id)
        if prompt_text_token is not None and prompt_text_token_len is not None:
            model_input["prompt_text"] = prompt_text_token
            model_input["prompt_text_len"] = prompt_text_token_len
        for out in cosyvoice.model.tts(**model_input, stream=bool(args.stream), speed=float(args.speed)):
            wav = out["tts_speech"]
            if not isinstance(wav, torch.Tensor):
                print(f"[infer] unexpected tts_speech type: {type(wav)}")
                return 4
            chunk_path = out_dir / f"chunk_{idx:04d}.wav"
            torchaudio.save(str(chunk_path), wav, cosyvoice.sample_rate)
            print(f"[infer] wrote {chunk_path}")
            if args.concat:
                chunks.append(wav)
            idx += 1

    if args.concat and chunks:
        full = torch.cat(chunks, dim=1)
        full_path = out_dir / "full.wav"
        torchaudio.save(str(full_path), full, cosyvoice.sample_rate)
        print(f"[infer] wrote {full_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
