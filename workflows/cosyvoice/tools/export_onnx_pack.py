#!/usr/bin/env python3
"""
Export a "CosyVoice ONNX Pack" (V1) for ChaosSeed pure-Rust runtime (tract-onnx).

This script is intentionally *offline tooling*:
- It may use PyTorch / Transformers to export ONNX.
- The produced pack can be consumed by ChaosSeed runtime without Python / onnxruntime.

Example:
  uv run python tools/export_onnx_pack.py \
    --model_dir pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
    --spk_id dream \
    --llm_ckpt  exp/dream_sft/llm/torch_ddp/epoch_5_whole.pt \
    --flow_ckpt exp/dream_sft/flow/torch_ddp/flow_avg.pt \
    --out_dir   export_packs/dream_sft_pack_v1 \
    --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from voicelab_bootstrap import ensure_sys_path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    obj.pop("epoch", None)
    obj.pop("step", None)
    return obj


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export CosyVoice3 SFT models into an ONNX pack (V1)."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="CosyVoice3 model dir (contains cosyvoice3.yaml/llm.pt/flow.pt/hift.pt).",
    )
    p.add_argument(
        "--out_dir", type=str, required=True, help="Output directory for the pack."
    )
    p.add_argument(
        "--spk_id",
        type=str,
        default="",
        help="Optional: sanity check that spk_id exists in spk2info.pt.",
    )
    p.add_argument(
        "--llm_ckpt",
        type=str,
        default="",
        help="Optional: hot-load LLM checkpoint state_dict before export.",
    )
    p.add_argument(
        "--flow_ckpt",
        type=str,
        default="",
        help="Optional: hot-load Flow checkpoint state_dict before export.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Export device.",
    )
    # torch.onnx.export（legacy exporter）最高稳定支持到 opset 17。
    # opset 18+ 建议走 torch.onnx.dynamo_export（但仍是 preview）。
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (legacy exporter: <=17 recommended).",
    )
    p.add_argument(
        "--skip_onnx",
        action="store_true",
        help="Only export pack.json/tokenizer/spk2info (no ONNX).",
    )
    p.add_argument(
        "--no_constant_folding",
        action="store_true",
        help=(
            "Disable ONNX constant folding to speed up export (often helps Flow/HiFT). "
            "May produce a slightly larger graph; runtime behavior should stay the same."
        ),
    )
    p.add_argument(
        "--flow_dummy_token_len",
        type=int,
        default=256,
        help="Dummy speech token length for exporting Flow (larger is safer for long texts).",
    )
    return p


@contextlib.contextmanager
def _patch_sdpa_for_onnx_export():
    """
    修复 torch.onnx.export（legacy exporter）在导出 SDPA（scaled_dot_product_attention）时的兼容性问题。

    现象：Qwen2 在 Transformers 的某些默认路径会调用 F.scaled_dot_product_attention，
    torch==2.3.x 的 ONNX symbolic 可能会把 Python float scale 当作 Constant(Tensor) 写入，进而崩溃。

    方案：导出期间 monkey-patch 成纯 matmul/softmax 实现，让导出图只包含基础算子，
    同时更利于 tract-onnx 这类纯 Rust runtime 的支持。
    """

    import torch
    import torch.nn.functional as F

    orig = getattr(F, "scaled_dot_product_attention", None)
    if orig is None:
        yield
        return

    def _sdpa_fallback(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        # 推理/导出场景：不支持 dropout（应为 0）。
        if dropout_p and float(dropout_p) != 0.0:
            raise RuntimeError(
                "[export] SDPA fallback does not support dropout during ONNX export."
            )

        scores = torch.matmul(query, key.transpose(-2, -1))
        d = int(query.size(-1))
        s = float(scale) if scale is not None else (1.0 / (float(d) ** 0.5))
        scores = scores * s

        if is_causal:
            q_len = int(scores.size(-2))
            k_len = int(scores.size(-1))
            causal = torch.ones(
                (q_len, k_len), dtype=torch.bool, device=scores.device
            ).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))

        if attn_mask is not None:
            # SDPA 支持 bool mask（False=mask）或 float mask（加性）。
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, value)

    F.scaled_dot_product_attention = _sdpa_fallback  # type: ignore[assignment]
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig  # type: ignore[assignment]


@contextlib.contextmanager
def _patch_stft_for_onnx_export():
    """
    使用 Conv1d 数学等价替换 torch.stft，绕过 legacy ONNX exporter 对复数/aten::stft 的限制。

    说明：
    - 该补丁只在导出期间启用，目标是让导出图只包含基础算子（Conv/Pad/Trig 等），更利于 opset 17
      以及 tract-onnx 这类纯 Rust runtime 的兼容性。
    - 为避免 ONNX 图中出现 complex dtype，这里会强制返回与 `return_complex=False` 等价的实数表示：
      [..., 2] 的 (real, imag) 对。
    """

    import math

    import torch
    import torch.nn.functional as F

    orig_stft = torch.stft
    warned = {"force_real": False}

    def stft_fallback(
        input,  # noqa: A002 - keep torch.stft signature
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=None,
    ):
        # 只覆盖推理/导出常见路径；遇到非常规输入/参数则回退原实现，避免静默导出错误图。
        if not isinstance(input, torch.Tensor):
            return orig_stft(
                input,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
            )

        if input.is_complex():
            raise RuntimeError(
                "[export] stft fallback: complex input is not supported."
            )

        # ONNX legacy exporter 对 complex dtype 支持很差；导出时强制走 real/imag 对表示。
        if return_complex is None or bool(return_complex) is True:
            if not warned["force_real"]:
                print(
                    "[export] WARN: torch.stft(return_complex=True/None) -> forcing return_complex=False for ONNX export."
                )
                warned["force_real"] = True
        return_complex = False

        try:
            n_fft_i = int(n_fft)
        except Exception:
            return orig_stft(
                input,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
            )

        hop_i = int(hop_length) if hop_length is not None else (n_fft_i // 4)
        win_i = int(win_length) if win_length is not None else n_fft_i

        x = input
        input_was_1d = False
        if x.dim() == 1:
            input_was_1d = True
            x = x.unsqueeze(0)
        if x.dim() != 2:
            # 罕见形状：回退原实现（避免导出错误图）。
            return orig_stft(
                input,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
            )

        if center:
            pad = n_fft_i // 2
            # F.pad 的 1D 反射/复制填充需要 (N, C, L)
            x = F.pad(x.unsqueeze(1), (pad, pad), mode=str(pad_mode)).squeeze(1)

        device = x.device
        # 用 float32 构建基底，最后再 cast 到输入 dtype，避免 float16 下精度过差。
        k = torch.arange(n_fft_i, dtype=torch.float32, device=device).unsqueeze(
            1
        )  # [K,1]
        n = torch.arange(n_fft_i, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # [1,N]
        ang = 2.0 * math.pi * (k * n) / float(n_fft_i)  # [K,N]
        basis_real = torch.cos(-ang)
        basis_imag = torch.sin(-ang)

        if window is not None:
            if not isinstance(window, torch.Tensor):
                # window 不是 Tensor 时回退，避免 dtype/device 对不上。
                return orig_stft(
                    input,
                    n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=center,
                    pad_mode=pad_mode,
                    normalized=normalized,
                    onesided=onesided,
                    return_complex=return_complex,
                )
            if window.is_complex():
                raise RuntimeError(
                    "[export] stft fallback: complex window is not supported."
                )
            w = window.to(device=device, dtype=torch.float32)
            if int(w.numel()) != int(win_i):
                # 参数不匹配时直接回退，避免静默行为变化。
                return orig_stft(
                    input,
                    n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=center,
                    pad_mode=pad_mode,
                    normalized=normalized,
                    onesided=onesided,
                    return_complex=return_complex,
                )
            if win_i != n_fft_i:
                pad_left = (n_fft_i - win_i) // 2
                pad_right = n_fft_i - win_i - pad_left
                w = F.pad(w, (pad_left, pad_right))
            basis_real = basis_real * w.unsqueeze(0)
            basis_imag = basis_imag * w.unsqueeze(0)

        if normalized:
            scale = 1.0 / (float(n_fft_i) ** 0.5)
            basis_real = basis_real * scale
            basis_imag = basis_imag * scale

        if onesided:
            freqs = n_fft_i // 2 + 1
            basis_real = basis_real[:freqs]
            basis_imag = basis_imag[:freqs]
        else:
            freqs = n_fft_i

        # 合并实部/虚部作为卷积核：[out_channels, in_channels, kernel_size]
        basis = torch.cat([basis_real.unsqueeze(1), basis_imag.unsqueeze(1)], dim=0).to(
            dtype=x.dtype
        )
        y = F.conv1d(x.unsqueeze(1), basis, stride=hop_i)

        # 还原为 torch.stft(return_complex=False) 的输出布局：[B, F, Frames, 2]
        B = int(x.size(0))
        frames = int(y.size(-1))
        y = y.view(B, 2, int(freqs), frames).permute(0, 2, 3, 1).contiguous()
        if input_was_1d:
            y = y.squeeze(0)
        return y

    torch.stft = stft_fallback  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.stft = orig_stft  # type: ignore[assignment]


@dataclass(frozen=True)
class _LlmIoNames:
    prefill_inputs: List[str]
    prefill_outputs: List[str]
    decode_inputs: List[str]
    decode_outputs: List[str]


def _make_llm_io_names(num_layers: int) -> _LlmIoNames:
    past = []
    new_past = []
    for i in range(num_layers):
        past.extend([f"past_{i}_key", f"past_{i}_value"])
        new_past.extend([f"present_{i}_key", f"present_{i}_value"])
    return _LlmIoNames(
        prefill_inputs=["input_ids"],
        prefill_outputs=["logits", *past],
        decode_inputs=["token_id", *past],
        decode_outputs=["logits", *new_past],
    )


def _pack_past(flat: Tuple[Any, ...]) -> Tuple[Tuple[Any, Any], ...]:
    assert len(flat) % 2 == 0
    return tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))


def _unpack_past(past: Iterable[Tuple[Any, Any]]) -> Tuple[Any, ...]:
    out = []
    for k, v in past:
        out.append(k)
        out.append(v)
    return tuple(out)


def _past_len(cache_k: Any) -> int:
    # HF typically: [B, H, T, D] or [B, T, H, D] depending on model.
    # We only need T; use a tolerant heuristic.
    if hasattr(cache_k, "dim") and cache_k.dim() >= 3:
        # Prefer axis=2 when 4D, else use the second-to-last.
        if cache_k.dim() == 4:
            return int(cache_k.size(2))
        return int(cache_k.size(-2))
    raise ValueError(f"unexpected cache key shape: {getattr(cache_k, 'shape', None)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = Path(str(args.model_dir)).expanduser().resolve()
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_sys_path()

    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"[export] device={device}")

    cosyvoice = AutoModel(model_dir=str(model_dir))
    # cosyvoice.model is a small wrapper class (not an nn.Module). Move sub-modules explicitly.
    cosyvoice.model.llm.to(device).eval()  # type: ignore[attr-defined]
    cosyvoice.model.flow.to(device).eval()  # type: ignore[attr-defined]
    cosyvoice.model.hift.to(device).eval()  # type: ignore[attr-defined]

    if str(args.spk_id).strip():
        spk_id = str(args.spk_id).strip()
        list_spks = getattr(cosyvoice, "list_available_spks", None)
        if callable(list_spks):
            available = list_spks()
            if spk_id not in available:
                raise SystemExit(
                    f"[export] spk_id not found in spk2info.pt: {spk_id} (available={available})"
                )

    # Optional hot-load checkpoints (state_dict) just like infer_sft.py.
    llm_ckpt = (
        Path(str(args.llm_ckpt)).expanduser().resolve()
        if str(args.llm_ckpt).strip()
        else None
    )
    if llm_ckpt is not None:
        state = _load_state_dict_maybe(llm_ckpt)
        if state is None:
            raise SystemExit(
                f"[export] failed to load llm_ckpt as state_dict: {llm_ckpt}"
            )
        missing, unexpected = cosyvoice.model.llm.load_state_dict(state, strict=False)  # type: ignore[attr-defined]
        print(f"[export] hot-loaded llm_ckpt: {llm_ckpt}")
        if missing:
            print(f"[export] WARN: llm missing_keys={len(missing)}")
        if unexpected:
            print(f"[export] WARN: llm unexpected_keys={len(unexpected)}")

    flow_ckpt = (
        Path(str(args.flow_ckpt)).expanduser().resolve()
        if str(args.flow_ckpt).strip()
        else None
    )
    if flow_ckpt is not None:
        state = _load_state_dict_maybe(flow_ckpt)
        if state is None:
            raise SystemExit(
                f"[export] failed to load flow_ckpt as state_dict: {flow_ckpt}"
            )
        missing, unexpected = cosyvoice.model.flow.load_state_dict(state, strict=False)  # type: ignore[attr-defined]
        print(f"[export] hot-loaded flow_ckpt: {flow_ckpt}")
        if missing:
            print(f"[export] WARN: flow missing_keys={len(missing)}")
        if unexpected:
            print(f"[export] WARN: flow unexpected_keys={len(unexpected)}")

    # -----------------------------
    # Export tokenizer.json (+ endOfPromptTokenId)
    # -----------------------------
    from transformers import AutoTokenizer

    blank = model_dir / "CosyVoice-BlankEN"
    tok = AutoTokenizer.from_pretrained(str(blank), use_fast=True)
    eop_raw = tok.convert_tokens_to_ids("<|endofprompt|>")
    if eop_raw is None:
        # CosyVoice3 uses a custom marker token that is not part of vanilla Qwen2 tokenizer.
        # CosyVoice expects it to be appended after the existing special tokens, and upstream
        # hardcodes the id as 151646 for Qwen2-0.5B vocab. We do NOT hardcode it here; we add
        # the token and record the resulting id into pack.json.
        _ = tok.add_special_tokens({"additional_special_tokens": ["<|endofprompt|>"]})
        eop_raw = tok.convert_tokens_to_ids("<|endofprompt|>")
    if eop_raw is None:
        raise SystemExit("[export] failed to register <|endofprompt|> into tokenizer.")
    eop_id = int(eop_raw)
    # Save tokenizer.json in HF fast-tokenizer format for Rust `tokenizers` crate.
    tok_json = out_dir / "tokenizer.json"
    try:
        tok.backend_tokenizer.save(str(tok_json))
    except Exception as exc:
        raise SystemExit(f"[export] failed to save tokenizer.json: {exc}")
    print(f"[export] wrote {tok_json} (endOfPromptTokenId={eop_id})")

    # -----------------------------
    # Export spk2info.json (only embeddings)
    # -----------------------------
    spk2info_pt = model_dir / "spk2info.pt"
    if not spk2info_pt.exists():
        raise SystemExit(f"[export] missing spk2info.pt: {spk2info_pt}")
    spk2info_obj = torch.load(str(spk2info_pt), map_location="cpu", weights_only=True)
    if not isinstance(spk2info_obj, dict):
        raise SystemExit(f"[export] unexpected spk2info.pt type: {type(spk2info_obj)}")
    spk2info_json: dict[str, dict[str, list[float]]] = {}
    for k, v in spk2info_obj.items():
        if not isinstance(v, dict) or "embedding" not in v:
            continue
        emb = v["embedding"]
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().to(torch.float32).flatten().tolist()
        else:
            emb = list(emb)
        spk2info_json[str(k)] = {"embedding": [float(x) for x in emb]}

    if not spk2info_json:
        raise SystemExit("[export] spk2info.pt did not contain any embeddings.")
    # Infer embedding dim.
    any_emb = next(iter(spk2info_json.values()))["embedding"]
    spk_embed_dim = int(len(any_emb))
    spk2info_path = out_dir / "spk2info.json"
    spk2info_path.write_text(
        json.dumps(spk2info_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[export] wrote {spk2info_path} (spkEmbedDim={spk_embed_dim}, spks={len(spk2info_json)})"
    )

    # -----------------------------
    # Build pack.json
    # -----------------------------
    llm = cosyvoice.model.llm  # type: ignore[attr-defined]
    flow = cosyvoice.model.flow  # type: ignore[attr-defined]
    hift = cosyvoice.model.hift  # type: ignore[attr-defined]

    # 说明：
    # - Rust 端的推理停止条件默认是「采样到 stop token」。
    # - 我们用 pack.json 的 `stopTokenStart` 表示：token_id >= stopTokenStart 视为 stop token。
    #
    # 但不同 CosyVoice3 导出/权重可能有两种常见约定：
    # 1) decoder 的 vocab = speech_token_size + N（N>=1），其中 [speech_token_size..] 是 stop / special tokens
    # 2) decoder 的 vocab = speech_token_size（没有额外 stop range），此时通常「最后一个 token」是 stop
    #
    # 若 pack.json 误把 stopTokenStart 写成 speech_token_size，而 decoder vocab 又恰好等于 speech_token_size，
    # 则 Rust 端永远采样不到 stop token，会一路跑到 max_len，结果：推理很慢 + 音频容易全是“电音/噪声”。
    speech_token_size = int(getattr(llm, "speech_token_size"))
    decoder_vocab_size: int | None = None
    try:
        dec = getattr(llm, "llm_decoder", None)
        if dec is not None and hasattr(dec, "out_features"):
            decoder_vocab_size = int(getattr(dec, "out_features"))
        elif dec is not None and hasattr(dec, "weight") and hasattr(dec.weight, "shape"):
            decoder_vocab_size = int(dec.weight.shape[0])
    except Exception:
        decoder_vocab_size = None

    stop_token_start = speech_token_size
    if decoder_vocab_size is not None:
        if decoder_vocab_size == speech_token_size:
            # 约定 #2：vocab 内没有额外 stop range；把最后一个 id 当 stop。
            # speech tokens: [0 .. speech_token_size-2]
            # stop token: speech_token_size-1
            speech_token_size = max(1, speech_token_size - 1)
            stop_token_start = speech_token_size
            print(
                "[export] detected decoder_vocab_size == speech_token_size; "
                f"treating last token as stop (speechTokenSize={speech_token_size}, stopTokenStart={stop_token_start})"
            )
        elif decoder_vocab_size < speech_token_size:
            # 理论上不应该发生：decoder vocab 小于 speech_token_size 代表权重/导出不一致。
            print(
                "[export] WARN: decoder_vocab_size < speech_token_size; "
                f"decoder_vocab_size={decoder_vocab_size} speech_token_size={speech_token_size}. "
                "Keeping stopTokenStart as speech_token_size, but the pack may be invalid."
            )
        else:
            # 约定 #1：存在 stop range
            stop_token_start = speech_token_size
            print(
                "[export] detected decoder_vocab_size > speech_token_size; "
                f"stopTokenStart={stop_token_start} decoder_vocab_size={decoder_vocab_size}"
            )
    token_mel_ratio = int(getattr(flow, "token_mel_ratio", 2))
    sample_rate = int(getattr(cosyvoice, "sample_rate", 24000))

    num_layers = int(
        getattr(getattr(llm, "llm").model.config, "num_hidden_layers")
    )  # Qwen2Encoder.model.config
    io_names = _make_llm_io_names(num_layers)

    pack_json = {
        "packVersion": 1,
        "sampleRate": sample_rate,
        "speechTokenSize": speech_token_size,
        "stopTokenStart": stop_token_start,
        "endOfPromptTokenId": eop_id,
        "spkEmbedDim": spk_embed_dim,
        "tokenMelRatio": token_mel_ratio,
        "tokenizerAddSpecialTokens": True,
        "textNormalize": "basic",
        "llm": {
            "minTokenTextRatio": 2.0,
            "maxTokenTextRatio": 20.0,
            "prefillIo": {
                "inputs": io_names.prefill_inputs,
                "outputs": io_names.prefill_outputs,
            },
            "decodeIo": {
                "inputs": io_names.decode_inputs,
                "outputs": io_names.decode_outputs,
            },
        },
        "flowIo": {"inputs": ["speech_tokens", "spk_embedding"], "outputs": ["mel"]},
        "hiftIo": {"inputs": ["mel"], "outputs": ["wav"]},
        "files": {
            "tokenizerJson": "tokenizer.json",
            "spk2infoJson": "spk2info.json",
            "llmPrefillOnnx": "llm_prefill.onnx",
            "llmDecodeOnnx": "llm_decode.onnx",
            "flowInferOnnx": "flow_infer.onnx",
            "hiftInferOnnx": "hift_infer.onnx",
        },
    }
    (out_dir / "pack.json").write_text(
        json.dumps(pack_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[export] wrote {out_dir / 'pack.json'}")

    if args.skip_onnx:
        sha = {
            "pack.json": _sha256_file(out_dir / "pack.json"),
            "tokenizer.json": _sha256_file(tok_json),
            "spk2info.json": _sha256_file(spk2info_path),
        }
        (out_dir / "sha256.json").write_text(
            json.dumps(sha, indent=2) + "\n", encoding="utf-8"
        )
        print("[export] --skip_onnx set; done.")
        return 0

    # -----------------------------
    # Export ONNX graphs
    # -----------------------------

    class _LlmPrefill(torch.nn.Module):
        def __init__(self, llm_mod: Any):
            super().__init__()
            self._qwen = llm_mod.llm.model.model  # Qwen2Model
            self._embed = llm_mod.llm.model.model.embed_tokens
            self._speech_emb = llm_mod.speech_embedding
            self._decoder = llm_mod.llm_decoder
            self._sos = int(llm_mod.sos)
            self._task = int(llm_mod.task_id)

        def forward(self, input_ids):
            # input_ids: [1, T] int64
            text_emb = self._embed(input_ids)
            sos_emb = (
                self._speech_emb.weight[self._sos].reshape(1, 1, -1).to(text_emb.dtype)
            )
            task_emb = (
                self._speech_emb.weight[self._task].reshape(1, 1, -1).to(text_emb.dtype)
            )
            x = torch.cat([sos_emb, text_emb, task_emb], dim=1)
            attn = torch.ones((1, x.size(1)), dtype=torch.bool, device=x.device)
            out = self._qwen(
                inputs_embeds=x, attention_mask=attn, use_cache=True, return_dict=True
            )
            h = out.last_hidden_state[:, -1:, :]
            logits = self._decoder(h).squeeze(1)  # [1, V]
            past = out.past_key_values
            # Transformers>=4.48 引入 Cache 对象；为保持导出 IO 仍是传统 (k,v) tuple，这里转回 legacy cache。
            if hasattr(past, "to_legacy_cache"):
                past = past.to_legacy_cache()
            return (logits, *(_unpack_past(past)))

    class _LlmDecode(torch.nn.Module):
        def __init__(self, llm_mod: Any, num_layers: int):
            super().__init__()
            self._qwen = llm_mod.llm.model.model  # Qwen2Model
            self._speech_emb = llm_mod.speech_embedding
            self._decoder = llm_mod.llm_decoder
            self._num_layers = int(num_layers)

        def forward(self, token_id, *past_flat):
            # token_id: [1, 1] int64 (speech token id)
            cache = _pack_past(past_flat)
            if len(cache) != self._num_layers:
                raise RuntimeError(
                    f"bad cache len={len(cache)} expected={self._num_layers}"
                )
            x = self._speech_emb(token_id)
            # 注意：不能用 `_past_len(...)` 来构造固定长度的 attention_mask。
            # 否则 ONNX tracing 会把 dummy past_len 固化成常量（例如 past_len=3 -> t=4），
            # 运行时遇到真实 past_len（例如 70）就会触发 4 by 70 之类的广播错误。
            # 这里从 KV cache 的 shape 动态派生 mask 长度：[B, past_len] + [B, 1] => [B, past_len+1]
            k0 = cache[0][0]  # [B, H, past_len, D]
            past_mask = torch.ones_like(k0[:, 0, :, 0], dtype=torch.bool)  # [B, past_len]
            cur_mask = torch.ones_like(token_id, dtype=torch.bool)  # [B, 1]
            attn = torch.cat([past_mask, cur_mask], dim=1)  # [B, past_len+1]

            # Transformers>=4.48: past_key_values 需要 Cache 对象；这里从 legacy cache 构建 DynamicCache。
            cache_obj: Any = cache
            try:
                from transformers.cache_utils import DynamicCache  # type: ignore

                cache_obj = DynamicCache.from_legacy_cache(cache)
            except Exception:
                cache_obj = cache
            out = self._qwen(
                inputs_embeds=x,
                attention_mask=attn,
                use_cache=True,
                past_key_values=cache_obj,
                return_dict=True,
            )
            h = out.last_hidden_state  # [1, 1, H]
            logits = self._decoder(h).squeeze(1)  # [1, V]
            past = out.past_key_values
            if hasattr(past, "to_legacy_cache"):
                past = past.to_legacy_cache()
            return (logits, *(_unpack_past(past)))

    class _FlowInfer(torch.nn.Module):
        def __init__(self, flow_mod: Any):
            super().__init__()
            self._flow = flow_mod

        def forward(self, speech_tokens, spk_embedding):
            # speech_tokens: [1, T] int64 -> int32
            token = speech_tokens.to(dtype=torch.int32)
            # 注意：ONNX 导出时不能用 token.size(1) 这种会在 tracing 期被“固化”的写法。
            # 否则 token_len 会被静态固定成 dummy_speech 的长度，运行时遇到长文本（例如 T=135）
            # 会触发 Mul/MatMul/Broadcast 等算子的 10 vs 135 广播错误。
            token_len = torch.onnx.operators.shape_as_tensor(token)[1:2].to(
                device=token.device, dtype=torch.int32
            )
            prompt_token = torch.zeros((1, 0), dtype=torch.int32, device=token.device)
            prompt_token_len = torch.tensor([0], dtype=torch.int32, device=token.device)
            prompt_feat = torch.zeros(
                (1, 0, 80), dtype=torch.float32, device=token.device
            )
            prompt_feat_len = torch.tensor([0], dtype=torch.int32, device=token.device)
            mel, _ = self._flow.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=spk_embedding.to(dtype=torch.float32),
                streaming=False,
                finalize=True,
            )
            return mel

    class _HiFTInfer(torch.nn.Module):
        def __init__(self, hift_mod: Any):
            super().__init__()
            self._hift = hift_mod

        def forward(self, mel):
            wav, _ = self._hift.inference(
                speech_feat=mel.to(dtype=torch.float32), finalize=True
            )
            return wav

    # Put model modules on device + eval + float32 for export stability.
    llm.to(device).eval()
    flow.to(device).eval()
    hift.to(device).eval()
    llm = llm.to(torch.float32)
    flow = flow.to(torch.float32)
    hift = hift.to(torch.float32)

    llm_prefill = _LlmPrefill(llm).to(device).eval()
    llm_decode = _LlmDecode(llm, num_layers).to(device).eval()
    flow_infer = _FlowInfer(flow).to(device).eval()
    hift_infer = _HiFTInfer(hift).to(device).eval()

    llm_prefill_path = out_dir / "llm_prefill.onnx"
    llm_decode_path = out_dir / "llm_decode.onnx"
    flow_path = out_dir / "flow_infer.onnx"
    hift_path = out_dir / "hift_infer.onnx"

    opset = int(args.opset)
    if opset > 17:
        print(
            f"[export] WARN: torch.onnx.export legacy exporter may not support opset {opset}; clamping to 17."
        )
        opset = 17
    print(f"[export] exporting ONNX (opset={opset}) ...")
    do_constant_folding = not bool(getattr(args, "no_constant_folding", False))
    if not do_constant_folding:
        print("[export] constant folding: disabled (--no_constant_folding)")

    # 导出期间的兼容性补丁（legacy ONNX exporter）：
    # - SDPA：帮助 Qwen2（LLM）以及 Flow 内可能用到 SDPA 的 DiT 模块顺利导出。
    # - STFT：避免图中出现 complex/aten::stft，保持 opset 17 + tract-onnx 友好。
    with _patch_sdpa_for_onnx_export(), _patch_stft_for_onnx_export():
        # LLM prefill
        t0 = time.time()
        dummy_ids = torch.tensor(
            [[eop_id]], dtype=torch.int64, device=device
        )  # minimal, still includes eop marker
        torch.onnx.export(
            llm_prefill,
            (dummy_ids,),
            str(llm_prefill_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=do_constant_folding,
            input_names=io_names.prefill_inputs,
            output_names=io_names.prefill_outputs,
            dynamic_axes={
                "input_ids": {1: "text_len"},
                **{name: {2: "past_len"} for name in io_names.prefill_outputs[1:]},
            },
        )
        print(f"[export] wrote {llm_prefill_path} ({time.time() - t0:.1f}s)")

        # LLM decode
        t0 = time.time()
        with torch.no_grad():
            pre_out = llm_prefill(dummy_ids)
        # pre_out = (logits, *past_flat)
        past_flat = tuple(pre_out[1:])
        dummy_tok = torch.zeros((1, 1), dtype=torch.int64, device=device)
        torch.onnx.export(
            llm_decode,
            (dummy_tok, *past_flat),
            str(llm_decode_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=do_constant_folding,
            input_names=io_names.decode_inputs,
            output_names=io_names.decode_outputs,
            dynamic_axes={
                "token_id": {1: "one"},
                **{name: {2: "past_len"} for name in io_names.decode_inputs[1:]},
                **{
                    name: {2: "past_len_plus_1"} for name in io_names.decode_outputs[1:]
                },
            },
        )
        print(f"[export] wrote {llm_decode_path} ({time.time() - t0:.1f}s)")

        # Flow
        print("[export] exporting Flow (this can take a long time) ...")
        t0 = time.time()
        dummy_speech = torch.zeros(
            (1, int(args.flow_dummy_token_len)), dtype=torch.int64, device=device
        )
        dummy_emb = torch.zeros((1, spk_embed_dim), dtype=torch.float32, device=device)
        torch.onnx.export(
            flow_infer,
            (dummy_speech, dummy_emb),
            str(flow_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=do_constant_folding,
            input_names=["speech_tokens", "spk_embedding"],
            output_names=["mel"],
            dynamic_axes={"speech_tokens": {1: "token_len"}, "mel": {2: "mel_len"}},
        )
        print(f"[export] wrote {flow_path} ({time.time() - t0:.1f}s)")

        # HiFT
        print("[export] exporting HiFT ...")
        t0 = time.time()
        dummy_mel = torch.zeros((1, 80, 20), dtype=torch.float32, device=device)
        torch.onnx.export(
            hift_infer,
            (dummy_mel,),
            str(hift_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=do_constant_folding,
            input_names=["mel"],
            output_names=["wav"],
            dynamic_axes={"mel": {2: "mel_len"}, "wav": {1: "wav_len"}},
        )
        print(f"[export] wrote {hift_path} ({time.time() - t0:.1f}s)")

    sha = {
        "pack.json": _sha256_file(out_dir / "pack.json"),
        "tokenizer.json": _sha256_file(tok_json),
        "spk2info.json": _sha256_file(spk2info_path),
        "llm_prefill.onnx": _sha256_file(llm_prefill_path),
        "llm_decode.onnx": _sha256_file(llm_decode_path),
        "flow_infer.onnx": _sha256_file(flow_path),
        "hift_infer.onnx": _sha256_file(hift_path),
    }
    (out_dir / "sha256.json").write_text(
        json.dumps(sha, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[export] wrote {out_dir / 'sha256.json'}")
    print("[export] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
