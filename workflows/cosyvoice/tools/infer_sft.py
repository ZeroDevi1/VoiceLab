import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from voicelab_bootstrap import ensure_sys_path


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _sha1_utf8(s: str) -> str:
    # Used only for run metadata / A/B comparison. Do not use for security.
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


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


@dataclass(frozen=True)
class _ResolvedText:
    text: str
    text_input: str  # "file" | "inline"
    text_file: str  # always present for run.json compatibility; "" for inline input


class _TextResolutionError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Speaker SFT inference (CosyVoice3) and save wav outputs."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument(
        "--spk_id",
        type=str,
        default="",
        help="Speaker id for SFT inference (must exist in <model_dir>/spk2info.pt). Required.",
    )

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file to synthesize.",
    )
    text_group.add_argument(
        "--text", type=str, default=None, help="Inline UTF-8 text to synthesize."
    )

    parser.add_argument(
        "--prompt_text",
        type=str,
        default="<|endofprompt|>",
        help=(
            "Prompt text for CosyVoice3 instruct mode. CosyVoice3 requires '<|endofprompt|>' to appear in either "
            "the prompt_text or the spoken text. Default is the minimal marker-only prompt to reduce leakage. "
            "Pass an empty string to disable prompt injection (then you must include '<|endofprompt|>' in --text/--text_file)."
        ),
    )
    parser.add_argument(
        "--prompt_strategy",
        choices=["inject", "guide_prefix"],
        default="inject",
        help=(
            "How to use prompt_text: inject=pass via model_input['prompt_text'] (default; may leak unpredictably), "
            "guide_prefix=prepend prompt (without '<|endofprompt|>') to the *spoken* text of each segment, "
            "and inject only '<|endofprompt|>' to satisfy CosyVoice3."
        ),
    )
    parser.add_argument(
        "--guide_sep",
        type=str,
        default=" ",
        help="Separator inserted between guide prefix and the real segment text when --prompt_strategy guide_prefix.",
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
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_wav",
        help="Output directory for run.json + chunk_0000.wav (and piece_*.wav when --stream).",
    )
    parser.add_argument(
        "--out_wav",
        type=str,
        default="",
        help=(
            "Optional output wav filepath (e.g. out_wav/demo/chunk_0000.wav). "
            "If set, the script will write the single-segment wav to this exact path, "
            "and will still write run.json (and piece_*.wav when --stream) into its parent directory."
        ),
    )
    parser.add_argument(
        "--stream", action="store_true", help="Use chunk streaming inference."
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--seed", type=int, default=1986, help="Random seed for reproducible A/B tests."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM sampling temperature (implemented by scaling log-probs before nucleus/RAS).",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.6, help="LLM sampling top_p for nucleus/RAS."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="LLM sampling top_k for nucleus/RAS."
    )
    parser.add_argument(
        "--win_size", type=int, default=10, help="LLM RAS repetition window size."
    )
    parser.add_argument(
        "--tau_r",
        type=float,
        default=1.0,
        help="LLM RAS repetition threshold (higher = less random_sampling).",
    )
    parser.add_argument(
        "--text_frontend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable text frontend normalization (wetext). Disable to avoid downloads / for quick tests.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_text(args: argparse.Namespace) -> _ResolvedText:
    if getattr(args, "text_file", None) is not None:
        text_path = Path(str(args.text_file))
        if not text_path.exists():
            raise _TextResolutionError(2, f"[infer] text_file not found: {text_path}")
        text = _read_text_file(text_path)
        if not text:
            raise _TextResolutionError(3, "[infer] text is empty")
        return _ResolvedText(
            text=text, text_input="file", text_file=str(args.text_file)
        )

    if getattr(args, "text", None) is not None:
        text = str(args.text).strip()
        if not text:
            raise _TextResolutionError(3, "[infer] text is empty")
        return _ResolvedText(text=text, text_input="inline", text_file="")

    # Should be unreachable due to mutually_exclusive_group(required=True).
    raise _TextResolutionError(3, "[infer] text is empty")


def resolve_output(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """
    Returns:
      - out_dir: directory to write run.json + chunk/piece wavs
      - out_wav: explicit output wav path if provided (else None -> use out_dir/chunk_0000.wav)
    """
    out_wav_raw = str(getattr(args, "out_wav", "") or "").strip()
    if out_wav_raw:
        out_wav = Path(out_wav_raw)
        if out_wav.exists() and out_wav.is_dir():
            raise _TextResolutionError(
                17, f"[infer] out_wav points to a directory, expected a file: {out_wav}"
            )
        out_dir = out_wav.parent
        return out_dir, out_wav

    out_dir = Path(str(getattr(args, "out_dir", "out_wav")))
    return out_dir, None


def _strip_endofprompt(s: str) -> str:
    # CosyVoice3 uses a special marker token; it's not meant to be spoken.
    return str(s or "").replace("<|endofprompt|>", "").strip()


def _extract_prompt_tokens(cosyvoice, *, prompt_text_raw: str, text_frontend: bool):
    """
    CosyVoice3 expects prompt_text tokens (already normalized/extracted by frontend).
    Returns (token, token_len) or (None, None) when prompt_text_raw is empty.
    """
    prompt_text_raw = str(prompt_text_raw or "").strip()
    if not prompt_text_raw:
        return None, None
    prompt_text_norm = cosyvoice.frontend.text_normalize(
        prompt_text_raw, split=False, text_frontend=bool(text_frontend)
    )
    return cosyvoice.frontend._extract_text_token(prompt_text_norm)


def _write_run_metadata(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    text: str,
    prompt_text: str,
    text_input: str,
    text_file: str,
    prompt_strategy: str,
    prompt_inject_text: str,
    guide_text: str,
) -> None:
    # Keep this small and deterministic so it can be used for A/B listening manifests.
    payload = {
        "model_dir": str(args.model_dir),
        "spk_id": str(args.spk_id),
        "text_input": str(text_input),
        "text_file": str(text_file),
        "text_sha1": _sha1_utf8(text),
        "text_chars": int(len(text)),
        "split_mode": "none",
        "prompt_strategy": str(prompt_strategy),
        "prompt_sha1": _sha1_utf8(prompt_text) if prompt_text else "",
        "prompt_chars": int(len(prompt_text)),
        "prompt_inject_sha1": _sha1_utf8(prompt_inject_text)
        if prompt_inject_text
        else "",
        "prompt_inject_chars": int(len(prompt_inject_text)),
        "guide_sha1": _sha1_utf8(guide_text) if guide_text else "",
        "guide_chars": int(len(guide_text)),
        "llm_ckpt": str(args.llm_ckpt),
        "flow_ckpt": str(args.flow_ckpt),
        "speed": float(args.speed),
        "stream": bool(args.stream),
        "text_frontend": bool(args.text_frontend),
        "seed": int(args.seed),
        "llm_sampling_temperature": float(args.temperature),
        "llm_sampling_top_p": float(args.top_p),
        "llm_sampling_top_k": int(args.top_k),
        "llm_sampling_win_size": int(args.win_size),
        "llm_sampling_tau_r": float(args.tau_r),
    }
    (out_dir / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _set_seed(seed: int) -> None:
    # CosyVoice sampling uses torch.multinomial (torch RNG) and sometimes python random.
    # Setting all three makes A/B tests meaningful.
    import random

    import numpy as np  # type: ignore
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        resolved = resolve_text(args)
    except _TextResolutionError as e:
        print(e.message)
        return int(e.code)

    try:
        out_dir, out_wav_path = resolve_output(args)
    except _TextResolutionError as e:
        print(e.message)
        return int(e.code)

    prompt_text_raw = str(getattr(args, "prompt_text", "") or "")
    # Preserve an intentional empty string, but normalize whitespace-only to empty.
    prompt_text_raw = prompt_text_raw.strip()
    prompt_strategy = (
        str(getattr(args, "prompt_strategy", "inject") or "inject").strip().lower()
    )
    guide_sep = str(getattr(args, "guide_sep", " ") or " ")

    guide_text = ""
    prompt_inject_text = prompt_text_raw
    if prompt_strategy == "guide_prefix":
        guide_text = _strip_endofprompt(prompt_text_raw)
        # In guide_prefix mode we always inject only the marker to satisfy CosyVoice3,
        # and rely on the spoken guide_text to drive emotion (user can cut it in post).
        prompt_inject_text = "<|endofprompt|>"
        if not guide_text:
            print(
                "[infer] WARN: prompt_strategy=guide_prefix but guide_text is empty (nothing will be prepended)."
            )

    spk_id = str(args.spk_id).strip()
    if not spk_id:
        print("[infer] spk_id is required.")
        print("[infer] Example: --spk_id dream")
        return 15

    if int(args.seed) < 0:
        print("[infer] seed must be >= 0")
        return 9
    if float(args.temperature) <= 0:
        print("[infer] temperature must be > 0")
        return 10
    if not (0 < float(args.top_p) <= 1.0):
        print("[infer] top_p must be in (0, 1]")
        return 11
    if int(args.top_k) <= 0:
        print("[infer] top_k must be > 0")
        return 12
    if int(args.win_size) <= 0:
        print("[infer] win_size must be > 0")
        return 13
    if float(args.tau_r) < 0:
        print("[infer] tau_r must be >= 0")
        return 14

    out_dir.mkdir(parents=True, exist_ok=True)

    # Heavy deps import after parse/validation so `--help` and unit tests stay light.
    ensure_sys_path()

    import torch  # noqa: E402
    import torchaudio  # noqa: E402
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

    cosyvoice = AutoModel(model_dir=args.model_dir)
    _set_seed(int(args.seed))

    # Validate speaker id exists in spk2info.pt (SFT inference requires it).
    list_spks = getattr(cosyvoice, "list_available_spks", None)
    if callable(list_spks):
        available = list_spks()
        if spk_id not in available:
            print(f"[infer] spk_id not found in spk2info.pt: {spk_id}")
            print(f"[infer] available_spks={available}")
            return 16

    # Take control of the LLM sampling behavior without editing cosyvoice3.yaml.
    # Most CosyVoice configs use ras_sampling; we wrap it to apply temperature as log-prob scaling.
    try:
        from cosyvoice.utils.common import ras_sampling  # type: ignore

        llm = getattr(cosyvoice.model, "llm", None)
        if llm is not None and hasattr(llm, "sampling"):

            def _sampling(weighted_scores, decoded_tokens, sampling):  # type: ignore[no-redef]
                if float(args.temperature) != 1.0:
                    weighted_scores = weighted_scores / float(args.temperature)
                return ras_sampling(
                    weighted_scores,
                    decoded_tokens,
                    sampling,
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    win_size=int(args.win_size),
                    tau_r=float(args.tau_r),
                )

            llm.sampling = _sampling  # type: ignore[attr-defined]
            print(
                "[infer] sampling override:"
                f" temp={float(args.temperature):.3f}"
                f" top_p={float(args.top_p):.3f}"
                f" top_k={int(args.top_k)}"
                f" win_size={int(args.win_size)}"
                f" tau_r={float(args.tau_r):.3f}"
                f" seed={int(args.seed)}"
            )
    except Exception as e:
        print(
            f"[infer] WARN: failed to override sampling params (using model default): {e}"
        )

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

    is_cosyvoice3 = (
        getattr(
            getattr(cosyvoice.model, "llm", None), "__class__", type("x", (), {})
        ).__name__
        == "CosyVoice3LM"
    )
    if is_cosyvoice3:
        # CosyVoice3 requires '<|endofprompt|>' to appear in either the prompt_text or the spoken text.
        if prompt_inject_text:
            if "<|endofprompt|>" not in prompt_inject_text:
                # Make this robust: auto-append the required token so users don't get a late assertion crash.
                prompt_inject_text = prompt_inject_text + "<|endofprompt|>"
                print(
                    "[infer] WARN: auto-appended '<|endofprompt|>' to prompt_text (CosyVoice3 requirement)."
                )
        else:
            if "<|endofprompt|>" not in resolved.text:
                print(
                    "[infer] CosyVoice3 requires '<|endofprompt|>' in text or prompt_text."
                )
                print(
                    "[infer] Pass --prompt_text '<|endofprompt|>' (default) or include the marker in --text/--text_file."
                )
                return 5

    # Write run metadata only after we have validated spk_id/prompt, but without storing any raw text content.
    _write_run_metadata(
        out_dir,
        args=args,
        text=resolved.text,
        prompt_text=prompt_text_raw,
        text_input=resolved.text_input,
        text_file=resolved.text_file,
        prompt_strategy=prompt_strategy,
        prompt_inject_text=prompt_inject_text,
        guide_text=guide_text,
    )

    prompt_token = None
    prompt_token_len = None
    if is_cosyvoice3 and prompt_inject_text:
        prompt_token, prompt_token_len = _extract_prompt_tokens(
            cosyvoice,
            prompt_text_raw=prompt_inject_text,
            text_frontend=bool(args.text_frontend),
        )

    # Never auto-split: users decide how to segment long texts.
    norm = cosyvoice.frontend.text_normalize(
        resolved.text, split=False, text_frontend=bool(args.text_frontend)
    )
    if isinstance(norm, str):
        seg_norm = norm
    elif isinstance(norm, (list, tuple)):
        # Unexpected, but make it deterministic and still synthesize as one segment.
        seg_norm = "".join(str(x) for x in norm)
        print(
            "[infer] WARN: text_normalize returned a list/tuple with split=False; coercing to a single segment."
        )
    else:
        seg_norm = str(norm)
        print(f"[infer] WARN: text_normalize returned {type(norm)}; coercing to str.")

    segments = [seg_norm]

    idx = 0  # piece index (stream mode)
    for seg_i, seg in enumerate(segments):
        seg_text = seg
        if prompt_strategy == "guide_prefix" and guide_text:
            seg_text = f"{guide_text}{guide_sep}{seg_text}"

        model_input = cosyvoice.frontend.frontend_sft(seg_text, spk_id)
        if is_cosyvoice3 and prompt_token is not None and prompt_token_len is not None:
            model_input["prompt_text"] = prompt_token
            model_input["prompt_text_len"] = prompt_token_len
        # In non-stream mode, cosyvoice yields a single full waveform for this segment.
        # In stream mode, it yields multiple non-overlapping waveform pieces which we need to concatenate.
        wav_pieces: list[torch.Tensor] = []
        for out in cosyvoice.model.tts(
            **model_input, stream=bool(args.stream), speed=float(args.speed)
        ):
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

        seg_wav = (
            wav_pieces[0] if len(wav_pieces) == 1 else torch.cat(wav_pieces, dim=1)
        )
        out_path = out_wav_path or (out_dir / "chunk_0000.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), seg_wav, cosyvoice.sample_rate)
        print(f"[infer] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
