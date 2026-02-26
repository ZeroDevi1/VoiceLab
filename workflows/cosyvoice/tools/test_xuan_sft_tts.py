import argparse
import os
import re
from pathlib import Path

from voicelab_bootstrap import ensure_sys_path

DEFAULT_TEXT = (
    "马头有大！马头来啦！我跟你们说马头不仅要小组第一马头还要去msi。"
    "挖槽把狼母格温全ban了，马头笑了，哈哈哈哈哈 哈哈哈哈 "
    "挖槽三打四了，马头被单杀了 马头怎么不去死呢马头挖槽马头在卡视野 "
    "哎呦挖槽咬住了！把别人当杀币"
)

PROMPT_NEUTRAL = "You are a helpful assistant.<|endofprompt|>"
PROMPT_HYPE = "马头有大！马头来啦！<|endofprompt|>"


def _workflow_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_llm_ckpt_dir() -> Path:
    return _workflow_root() / "exp" / "xuan_sft" / "llm" / "torch_ddp"


def _pick_epoch_ckpt(ckpt_dir: Path, epoch: int) -> Path | None:
    p = ckpt_dir / f"epoch_{epoch}_whole.pt"
    return p if p.exists() else None


def _pick_latest_epoch_ckpt(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    best_epoch = -1
    best_path: Path | None = None
    for p in ckpt_dir.glob("epoch_*_whole.pt"):
        m = re.match(r"^epoch_(\\d+)_whole\\.pt$", p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = p
    return best_path


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


def _prompt_text_from_args(prompt_text_arg: str | None, preset: str) -> str:
    if prompt_text_arg is None:
        if preset == "neutral":
            return PROMPT_NEUTRAL
        if preset == "hype":
            return PROMPT_HYPE
        if preset == "none":
            return ""
        raise ValueError(f"unknown prompt_preset={preset!r}")
    return str(prompt_text_arg)


def _prompt_tokens_for_cosyvoice3(cosyvoice, prompt_text_raw: str, text_frontend: bool):
    if not prompt_text_raw.strip():
        return None, None
    prompt_text_norm = cosyvoice.frontend.text_normalize(prompt_text_raw, split=False, text_frontend=text_frontend)
    return cosyvoice.frontend._extract_text_token(prompt_text_norm)


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    return s or "x"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a quick SFT TTS sample for speaker 'xuan'.")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft")
    parser.add_argument("--spk_id", type=str, default="xuan")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument(
        "--prompt_preset",
        type=str,
        choices=["neutral", "hype", "none"],
        default="hype",
        help=(
            "Prompt preset for CosyVoice3 instruct mode. "
            "Use 'hype' to inject emotion, 'neutral' for stable baseline, 'none' to disable."
        ),
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help=(
            "Override prompt_text. CosyVoice3 requires '<|endofprompt|>' to appear in prompt_text. "
            "If omitted, uses --prompt_preset."
        ),
    )
    parser.add_argument("--out_wav", type=str, default="out_wav/xuan_test.wav")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--speed", type=float, default=1.1)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate a small comparison set (neutral/hype prompts × speed 1.0/1.1) into --out_wav directory.",
    )
    parser.add_argument(
        "--text_frontend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enable text frontend normalization (may download wetext on first run).",
    )
    parser.add_argument(
        "--llm_ckpt",
        type=str,
        default="",
        help=(
            "Optional: path to an LLM checkpoint to hot-load for inference. "
            "Examples: exp/xuan_sft/llm/torch_ddp/epoch_6_whole.pt, exp/xuan_sft/llm/torch_ddp/llm_avg.pt. "
            "If empty, will prefer epoch_6_whole.pt when present, otherwise auto-pick the latest epoch_*_whole.pt."
        ),
    )
    args = parser.parse_args()

    ensure_sys_path()

    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
    import torchaudio  # noqa: E402
    import torch  # noqa: E402

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"[test_tts] model_dir not found: {model_dir}")
        return 2

    text = str(args.text).strip()
    if not text:
        print("[test_tts] text is empty")
        return 3

    cosyvoice = AutoModel(model_dir=str(model_dir))
    if args.spk_id not in cosyvoice.list_available_spks():
        print(f"[test_tts] spk_id not found in spk2info.pt: {args.spk_id}")
        print(f"[test_tts] available_spks={cosyvoice.list_available_spks()}")
        print("[test_tts] SFT inference does NOT require a reference wav, but it DOES require spk2info.pt to include spk_id.")
        return 4

    is_cosyvoice3 = getattr(getattr(cosyvoice.model, "llm", None), "__class__", type("x", (), {})).__name__ == "CosyVoice3LM"
    try:
        prompt_text_raw = _prompt_text_from_args(args.prompt_text, str(args.prompt_preset))
    except ValueError as e:
        print(f"[test_tts] {e}")
        return 7
    prompt_text_raw = str(prompt_text_raw)
    if is_cosyvoice3:
        if not prompt_text_raw.strip():
            print("[test_tts] CosyVoice3 requires prompt_text containing '<|endofprompt|>'.")
            print("[test_tts] Use --prompt_preset neutral/hype or pass --prompt_text explicitly.")
            return 7
        if "<|endofprompt|>" not in prompt_text_raw:
            print("[test_tts] CosyVoice3 requires '<|endofprompt|>' to appear in prompt_text.")
            print("[test_tts] Example: --prompt_text '马头有大！马头来啦！<|endofprompt|>'")
            return 7

    llm_ckpt_arg = str(args.llm_ckpt).strip()
    llm_ckpt: Path | None
    if llm_ckpt_arg:
        llm_ckpt = Path(llm_ckpt_arg)
        # Convenience: allow passing just "epoch_6_whole.pt".
        if not llm_ckpt.exists() and (os.sep not in llm_ckpt_arg and "/" not in llm_ckpt_arg and "\\" not in llm_ckpt_arg):
            cand = _default_llm_ckpt_dir() / llm_ckpt_arg
            if cand.exists():
                llm_ckpt = cand
    else:
        ckpt_dir = _default_llm_ckpt_dir()
        llm_ckpt = _pick_epoch_ckpt(ckpt_dir, 6) or _pick_latest_epoch_ckpt(ckpt_dir)

    if llm_ckpt is not None:
        state = _load_state_dict_maybe(llm_ckpt)
        if state is None:
            print(f"[test_tts] failed to load llm_ckpt as state_dict: {llm_ckpt}")
            return 5
        missing, unexpected = cosyvoice.model.llm.load_state_dict(state, strict=False)
        print(f"[test_tts] hot-loaded llm_ckpt: {llm_ckpt}")
        if missing:
            print(f"[test_tts] WARN: missing_keys={len(missing)}")
        if unexpected:
            print(f"[test_tts] WARN: unexpected_keys={len(unexpected)}")

    out_path = Path(args.out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_one(*, prompt_text: str, speed: float, out_file: Path) -> int:
        prompt_text_token = None
        prompt_text_token_len = None
        if is_cosyvoice3:
            prompt_text_token, prompt_text_token_len = _prompt_tokens_for_cosyvoice3(
                cosyvoice, prompt_text_raw=prompt_text, text_frontend=bool(args.text_frontend)
            )

        chunks: list[torch.Tensor] = []
        for seg in cosyvoice.frontend.text_normalize(text, split=True, text_frontend=bool(args.text_frontend)):
            model_input = cosyvoice.frontend.frontend_sft(seg, spk_id=str(args.spk_id))
            if prompt_text_token is not None and prompt_text_token_len is not None:
                model_input["prompt_text"] = prompt_text_token
                model_input["prompt_text_len"] = prompt_text_token_len
            for out in cosyvoice.model.tts(**model_input, stream=bool(args.stream), speed=float(speed)):
                wav = out["tts_speech"]
                chunks.append(wav)

        if not chunks:
            print("[test_tts] no audio produced")
            return 6

        full = torch.cat(chunks, dim=1)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_file), full, cosyvoice.sample_rate)
        print(f"[test_tts] wrote {out_file} (sr={cosyvoice.sample_rate}, sec={full.shape[1] / cosyvoice.sample_rate:.2f})")
        return 0

    if args.compare:
        out_dir = out_path.parent
        prefix = out_path.stem
        ckpt_tag = _slug(llm_ckpt.name if llm_ckpt is not None else "base")
        combos = [
            ("neutral", PROMPT_NEUTRAL, 1.0),
            ("neutral", PROMPT_NEUTRAL, 1.1),
            ("hype", PROMPT_HYPE, 1.0),
            ("hype", PROMPT_HYPE, 1.1),
        ]
        if not is_cosyvoice3:
            combos = [("noprompt", "", float(args.speed))]

        rc = 0
        for prompt_tag, prompt_text, speed in combos:
            out_file = out_dir / f"{prefix}_{ckpt_tag}_{prompt_tag}_speed{speed:.2f}.wav"
            this_rc = run_one(prompt_text=prompt_text if is_cosyvoice3 else "", speed=speed, out_file=out_file)
            rc = rc or this_rc
        return rc

    rc = run_one(prompt_text=prompt_text_raw, speed=float(args.speed), out_file=out_path)
    if rc != 0:
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
