# 在 WSL2 Ubuntu 24.04 用 uv 训练 CosyVoice 3「Dream 说话人 SFT」（LLM+Flow）全流程记录

记录日期：2026-02-22  
项目目录：`~/AntiGravityProjects/VoiceLab`  
workflow：`~/AntiGravityProjects/VoiceLab/workflows/cosyvoice`  
数据集（原始）：`/mnt/c/AIGC/数据集/Dream`（本次实际为 28 条 mp3，最长约 24.6s）

> 目标：训练出一个可复用的 **Dream 专属 CosyVoice3 SFT 模型目录**，并可对任意文本合成 Dream 声线音频。  
> 范围：LLM + Flow 都训练（但推理时也支持回退使用 base flow 做对比）。

---

## 0) 环境信息

- OS：WSL2 Ubuntu 24.04
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU（可见 `nvidia-smi`）
- `uv`：0.9.26

---

## 1) 路径与变量约定

在终端里统一使用：

```bash
export VOICELAB_DIR=~/AntiGravityProjects/VoiceLab
export WORKFLOW_DIR="$VOICELAB_DIR/workflows/cosyvoice"
export COSYVOICE_VENDOR_DIR="$VOICELAB_DIR/vendor/CosyVoice"

export DREAM_MP3_DIR="/mnt/c/AIGC/数据集/Dream"
export DREAM_SPK_ID="dream"

export DREAM_WAV_DIR="$WORKFLOW_DIR/data/dream_wav_24k_trim"
export DREAM_SFT_ROOT="$WORKFLOW_DIR/data/dream_sft"
export DREAM_MODEL_DIR="$WORKFLOW_DIR/pretrained_models/Fun-CosyVoice3-0.5B-dream-sft"
```

---

## 2) vendor 与子模块

本仓库 `vendor/CosyVoice` 是软链接到本机 CosyVoice 仓库：

```bash
ls -la "$VOICELAB_DIR/vendor"
```

CosyVoice3 的 YAML 配置会依赖 `third_party/Matcha-TTS`，需要初始化子模块：

```bash
cd "$COSYVOICE_VENDOR_DIR"
git submodule update --init --recursive
```

---

## 3) workflow Python 环境（Python 3.10）

`workflows/cosyvoice/pyproject.toml` 要求 `Python>=3.10,<3.11`，因此使用 uv 安装并同步：

```bash
cd "$WORKFLOW_DIR"
uv python install 3.10
uv sync --python 3.10 --extra asr
```

### 3.1 openai-whisper 安装坑（pkg_resources）

`openai-whisper==20231117` 在隔离构建时可能报：

```
ModuleNotFoundError: No module named 'pkg_resources'
```

解决方式：对 `openai-whisper` 关闭 build isolation，让它在当前 venv（已有 `setuptools==69.5.1`）里构建：

```bash
cd "$WORKFLOW_DIR"
uv pip install --python .venv/bin/python "openai-whisper==20231117" --no-build-isolation

# 让 sync 也能稳定跑（可重复执行，不会破坏环境）
uv sync --python 3.10 --extra asr --no-build-isolation-package openai-whisper
```

---

## 4) 数据预处理（mp3 -> wav，24k mono，去首尾静音）

目标：把训练用音频放到 Linux 文件系统（更快），并做基本静音裁剪。

```bash
mkdir -p "$DREAM_WAV_DIR"

dur() { ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1" 2>/dev/null || echo 0; }

for f in "$DREAM_MP3_DIR"/*.mp3; do
  base="$(basename "$f" .mp3)"
  tmp="$DREAM_WAV_DIR/${base}.tmp.wav"
  out="$DREAM_WAV_DIR/${base}.wav"

  # 24k/mono/PCM16
  ffmpeg -y -hide_banner -loglevel error -i "$f" -ac 1 -ar 24000 -c:a pcm_s16le "$tmp"

  # 去首尾静音；如果裁剪过度，则回退为未裁剪版本
  if ! sox "$tmp" "$out" silence 1 0.15 -35d reverse silence 1 0.15 -35d reverse; then
    cp -f "$tmp" "$out"
  else
    d_tmp="$(dur "$tmp")"
    d_out="$(dur "$out")"
    # 经验规则：原始 >1.5s 但裁剪后 <1.5s，多半是剪过头 -> 回退
    if python3 -c "import sys; dt=float(sys.argv[1] or 0); do=float(sys.argv[2] or 0); sys.exit(0 if (dt>1.5 and do<1.5) else 1)" "$d_tmp" "$d_out"; then
      cp -f "$tmp" "$out"
    fi
  fi

  rm -f "$tmp"
done
```

验收：

```bash
ls -1 "$DREAM_WAV_DIR"/*.wav | wc -l
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$DREAM_WAV_DIR/xxx.wav"
```

---

## 5) 生成 SFT 数据（faster-whisper 转写 -> kaldi 目录）

用 `tools/prepare_xuan_sft_dataset.py`（该脚本会产出 `metadata.jsonl` + `train/dev` 的 kaldi 文件）：

```bash
cd "$WORKFLOW_DIR"

#（可选）国内网络加速 Hugging Face 相关下载
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

uv run python tools/prepare_xuan_sft_dataset.py \
  --wav_dir "$DREAM_WAV_DIR" \
  --out_root "$DREAM_SFT_ROOT" \
  --spk_id "$DREAM_SPK_ID" \
  --backend faster-whisper \
  --whisper_model large-v3 \
  --device cuda \
  --compute_type int8_float16 \
  --vad_filter \
  --language zh \
  --max_sec 30
```

产物（示例）：

- `data/dream_sft/metadata.jsonl`
- `data/dream_sft/train/{wav.scp,text,utt2spk,spk2utt,instruct}`
- `data/dream_sft/dev/{...}`

验收：

```bash
wc -l data/dream_sft/metadata.jsonl
head -n 5 data/dream_sft/metadata.jsonl
```

> 质量建议：如果合成“说不清楚/错词多”，优先抽查 `metadata.jsonl` 的 `text` 是否对齐音频；必要时手工修正 `train/text` / `dev/text`。

---

## 6) 提取 embedding（campplus.onnx）

```bash
cd "$WORKFLOW_DIR"
uv run python tools/extract_embedding.py --dir data/dream_sft/train --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
uv run python tools/extract_embedding.py --dir data/dream_sft/dev   --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
```

验收（必须有 `dream`）：

```bash
uv run python - <<'PY'
import torch
spk2 = torch.load("data/dream_sft/train/spk2embedding.pt", map_location="cpu")
print("keys=", list(spk2.keys()))
print("dim=", len(next(iter(spk2.values()))))
PY
```

---

## 7) 生成 parquet + data.list

小数据建议切分更细（本次用每片 10 条）：

```bash
cd "$WORKFLOW_DIR"
mkdir -p data/dream_sft/train/parquet data/dream_sft/dev/parquet

uv run python tools/make_parquet_list.py \
  --num_utts_per_parquet 10 --num_processes 4 \
  --src_dir data/dream_sft/train --des_dir data/dream_sft/train/parquet

uv run python tools/make_parquet_list.py \
  --num_utts_per_parquet 10 --num_processes 1 \
  --src_dir data/dream_sft/dev --des_dir data/dream_sft/dev/parquet

wc -l data/dream_sft/train/parquet/data.list
wc -l data/dream_sft/dev/parquet/data.list
```

---

## 8) 训练（LLM + Flow）

训练前环境变量（保证 vendor + Matcha 可导入）：

```bash
cd "$WORKFLOW_DIR"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="$WORKFLOW_DIR:$COSYVOICE_VENDOR_DIR:$COSYVOICE_VENDOR_DIR/third_party/Matcha-TTS:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

# ORT CUDA 依赖在 WSL 里可能缺库（本次就遇到 libcublasLt.so.11），训练/推理仍可继续：
export COSYVOICE_ORT_FORCE_CPU=1
```

### 8.1 训练 LLM

```bash
cd "$WORKFLOW_DIR"
uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xuan_sft.yaml \
  --train_data data/dream_sft/train/parquet/data.list \
  --cv_data    data/dream_sft/dev/parquet/data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir exp/dream_sft/llm/torch_ddp \
  --tensorboard_dir tensorboard/dream_sft/llm/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 2 \
  --use_amp
```

产物：

- `exp/dream_sft/llm/torch_ddp/epoch_*_whole.pt`

### 8.2 训练 Flow

```bash
cd "$WORKFLOW_DIR"
uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xuan_sft.yaml \
  --train_data data/dream_sft/train/parquet/data.list \
  --cv_data    data/dream_sft/dev/parquet/data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model flow \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --model_dir exp/dream_sft/flow/torch_ddp \
  --tensorboard_dir tensorboard/dream_sft/flow/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 2 \
  --use_amp
```

> 经验：小数据训练 Flow 很容易“清晰度变差/发音怪”，后面推理阶段会提供 “SFT LLM + base flow” 的对比手段。

---

## 9) 平均 checkpoint（val_best）+ 组装 Dream 模型目录 + 生成 spk2info

平均模型：

```bash
cd "$WORKFLOW_DIR"
uv run python "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/average_model.py" \
  --dst_model exp/dream_sft/llm/torch_ddp/llm_avg.pt \
  --src_path exp/dream_sft/llm/torch_ddp \
  --num 3 --val_best

uv run python "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/average_model.py" \
  --dst_model exp/dream_sft/flow/torch_ddp/flow_avg.pt \
  --src_path exp/dream_sft/flow/torch_ddp \
  --num 5 --val_best
```

组装模型目录（复制 base 模型并覆盖 llm/flow）：

```bash
cd "$WORKFLOW_DIR"
uv run python tools/assemble_xuan_sft_model_dir.py --overwrite \
  --base_model_dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out_model_dir  pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
  --llm_pt  exp/dream_sft/llm/torch_ddp/llm_avg.pt \
  --flow_pt exp/dream_sft/flow/torch_ddp/flow_avg.pt
```

生成 `spk2info.pt`（SFT 推理必需）：

```bash
cd "$WORKFLOW_DIR"
uv run python tools/make_spk2info_from_spk2embedding.py \
  --spk2embedding_pt data/dream_sft/train/spk2embedding.pt \
  --spk_id dream \
  --out_spk2info_pt pretrained_models/Fun-CosyVoice3-0.5B-dream-sft/spk2info.pt
```

最终模型目录：

- `pretrained_models/Fun-CosyVoice3-0.5B-dream-sft/`
  - `llm.pt`
  - `flow.pt`
  - `spk2info.pt`
  - 以及 base 的 `hift.pt` / tokenizer / onnx 等

---

## 10) 推理（合成音频）

文本文件（UTF-8）：

```bash
cd "$WORKFLOW_DIR"
mkdir -p out_wav/dream_quote
cat > out_wav/dream_quote/input.txt <<'TXT'
我喜欢你，无论岁月拿你怎样。我想陪着你，走过花甲、踏过珠黄，到达一如既往。
TXT
```

### 10.1 直接用组装后的 dream-sft 合成

```bash
cd "$WORKFLOW_DIR"
uv run python tools/infer_sft.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
  --spk_id dream \
  --text_file out_wav/dream_quote/input.txt \
  --out_dir out_wav/dream_quote \
  --speed 1.0 \
  --prompt_text "你太入迷了。<|endofprompt|>"
```

输出：

- `out_wav/dream_quote/chunk_0000.wav`

### 10.2 用“训练过程中分数最高（CV loss 最低）”的 LLM 做试听

这一步无需重新组装 `model_dir`，直接热加载某个 epoch：

```bash
cd "$WORKFLOW_DIR"
uv run python tools/infer_sft.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
  --spk_id dream \
  --text_file out_wav/dream_quote/input.txt \
  --out_dir out_wav/dream_quote_bestllm_fixed \
  --llm_ckpt exp/dream_sft/llm/torch_ddp/epoch_5_whole.pt \
  --prompt_text "你太入迷了。<|endofprompt|>"
```

输出：

- `out_wav/dream_quote_bestllm_fixed/chunk_0000.wav`

### 10.3 常见补救：保留 SFT LLM，但回退 base flow（更清晰更稳）

```bash
cd "$WORKFLOW_DIR"
uv run python tools/infer_sft.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
  --spk_id dream \
  --text_file out_wav/dream_quote/input.txt \
  --out_dir out_wav/dream_quote_bestllm_baseflow \
  --llm_ckpt  exp/dream_sft/llm/torch_ddp/epoch_5_whole.pt \
  --flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --prompt_text "你太入迷了。<|endofprompt|>"
```

输出：

- `out_wav/dream_quote_bestllm_baseflow/chunk_0000.wav`

### 10.4 经验基线命令（“基准口味”：base flow + best LLM + prompt_short + sampling06k10 + speed=0.95）

> 注意：此处**不记录任何具体推理文本**；请自行把要合成的内容写入 `<TEXT_FILE>`（UTF-8）。

```bash
cd "$WORKFLOW_DIR"

#（可选）避免 ORT CUDA provider 缺库导致的不稳定噪声
export COSYVOICE_ORT_FORCE_CPU=1

uv run python tools/infer_sft.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
  --spk_id dream \
  --text_file <TEXT_FILE> \
  --out_dir <OUT_DIR> \
  --llm_ckpt  exp/dream_sft/llm/torch_ddp/epoch_5_whole.pt \
  --flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --prompt_text "你太入迷了。<|endofprompt|>" \
  --speed 0.95 \
  --seed 1986 \
  --temperature 1.0 \
  --top_p 0.6 \
  --top_k 10 \
  --win_size 10 \
  --tau_r 1.0
```

### 10.5（推荐）手工分段 + `guide_prefix`：每段前面都加“引导 prompt”（后期手工剪掉引导段）

> 适用：你想要“多段合成 + 情绪一致”，但又不希望脚本自动拆分导致情绪漂移。  
> `tools/infer_sft.py` **永远不做自动拆分**；请你自行把长文本切成多个小段（多个 txt 文件或多次命令），每段单独推理。

```bash
cd "$WORKFLOW_DIR"
uv run python tools/infer_sft.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
  --spk_id dream \
  --text_file <SEGMENT_TEXT_FILE> \
  --out_dir <SEGMENT_OUT_DIR> \
  --llm_ckpt  exp/dream_sft/llm/torch_ddp/epoch_5_whole.pt \
  --flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --prompt_text "<PROMPT_TEXT><|endofprompt|>" \
  --prompt_strategy guide_prefix \
  --guide_sep "。 " \
  --speed 1.02 \
  --seed 1986 \
  --temperature 1.0 \
  --top_p 0.75 \
  --top_k 20 \
  --win_size 10 \
  --tau_r 1.0
```

说明：
- 每次运行只输出一个音频文件：`chunk_0000.wav`（位于 `<SEGMENT_OUT_DIR>`）。
- 每段的开头都会朗读一遍“引导 prompt”（来自 `--prompt_text`，脚本会自动去掉其中的 `<|endofprompt|>`），你可以后期把引导段裁掉再自行拼接。

### 10.6（可选）直接用 `--text` 传入文本 + 用 `--out_wav` 精确指定输出文件名（run.json 不落盘原文）

```bash
cd "$WORKFLOW_DIR"
uv run python tools/infer_sft.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-dream-sft \
  --spk_id dream \
  --text "<TEXT>" \
  --out_wav <OUT_WAV> \
  --llm_ckpt  exp/dream_sft/llm/torch_ddp/epoch_5_whole.pt \
  --flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --prompt_text "<PROMPT_TEXT><|endofprompt|>" \
  --speed 0.95 \
  --seed 1986 \
  --temperature 1.0 \
  --top_p 0.6 \
  --top_k 10 \
  --win_size 10 \
  --tau_r 1.0
```

说明：
- 传入 `--out_wav` 会把本次单段推理的输出音频写到该路径（文件名任意，不再存在 “full.wav 拼接” 的概念）。
- `run.json`（以及 `--stream` 时的 `piece_*.wav`）会写入 `--out_wav` 的父目录（例如 `<OUT_WAV>` 的父目录）。

---

## 11) 本次排障与代码改动记录（重要）

### 11.1 `tools/voicelab_bootstrap.py` 路径 bug

问题：`voicelab_root()` 计算错误，导致 `ensure_sys_path()` 指向了错误的 `vendor/CosyVoice`，推理时报：

```
ModuleNotFoundError: No module named 'cosyvoice'
```

修复：`workflows/cosyvoice/tools/voicelab_bootstrap.py` 中把 `voicelab_root()` 修正为 `workflow_root().parents[1]`。

### 11.2 `tools/infer_sft.py` 的逐字符合成 bug（split=False 返回值误迭代）

问题：上游 `frontend.text_normalize(..., split=False)` 返回的是 **string**，如果直接 `for seg in ...` 会变成“按字符迭代”，导致：

- 生成大量 `chunk_00xx.wav`
- 合成输出呈现“逐字拼接”的错乱听感（像“说不明白”）

修复：`workflows/cosyvoice/tools/infer_sft.py` 将 split=False 的返回值包装成单元素 list，并强制单段推理输出（避免任何自动拆分）。

同时新增能力：

- `--llm_ckpt`：热加载某个 LLM epoch（试听 best epoch）
- `--flow_ckpt`：热加载 Flow（用于 “SFT LLM + base flow” 对比）
- `--text_frontend/--no-text_frontend`：控制 wetext 归一化
- **永远不自动拆分文本**：推理始终以单段进行，避免分句导致情绪漂移（分段由用户自行处理）

### 11.3 ORT CUDA provider 缺库

现象（推理/某些组件初始化时）：

```
Failed to load library libonnxruntime_providers_cuda.so ... libcublasLt.so.11 not found
```

说明：这会让 onnxruntime CUDA EP 创建失败，但会回退到 CPU EP；PyTorch 训练仍在 GPU 上跑。本次通过：

```bash
export COSYVOICE_ORT_FORCE_CPU=1
```

保证流程稳定可跑。

---

## 12) 结果与下一步建议（基于本次现象）

- 如果出现“合成不自然/不连贯/吐字怪”，优先做两件事：
  1) 保持“单段推理”（`tools/infer_sft.py` 默认行为；如需分段请手工分段）
  2) 用 `--flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt` 回退 base flow（小数据训 flow 容易把清晰度训坏）
- 如果仍然“说不清楚”，大概率是训练语料文本质量/数量不足：
  - 增加更干净的语音（去噪/去混响/去 BGM）
  - 手工校对转写文本（尤其是专有名词、口癖、连读）
