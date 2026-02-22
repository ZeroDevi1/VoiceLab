# 在 WSL2 Ubuntu 24.04 用 uv 训练 CosyVoice 3「XingTong 说话人 SFT」（LLM+Flow）— 参数化 Runbook

记录日期：2026-02-22  
项目目录：`~/AntiGravityProjects/VoiceLab`  
workflow：`~/AntiGravityProjects/VoiceLab/workflows/cosyvoice`  
数据集（原始）：`/mnt/c/AIGC/数据集/XingTong`（实测约 **1563** 条 `wav`，44.1kHz mono，时长约 2–16.8s）

> 目标：训练出一个可复用的 **XingTong 专属 CosyVoice3 SFT 模型目录**，并可对任意文本合成 XingTong 声线音频。  
> 范围：LLM + Flow 都训练；推理阶段默认先 A/B 使用 **base flow** 做清晰度保底。
>
> 说明：本文**只记录参数与命令模板**，不包含任何具体推理文本内容。

---

## 0) 统一变量（全流程固定，避免路径/ID 漂移）

在终端里：

```bash
export VOICELAB_DIR=~/AntiGravityProjects/VoiceLab
export WORKFLOW_DIR="$VOICELAB_DIR/workflows/cosyvoice"
export COSYVOICE_VENDOR_DIR="$VOICELAB_DIR/vendor/CosyVoice"

export XINGTONG_SRC_WAV_DIR="/mnt/c/AIGC/数据集/XingTong"
export XINGTONG_SPK_ID="XingTong"

# 预处理后的训练用 wav（放 Linux 文件系统里更快）
export XINGTONG_WAV_DIR="$WORKFLOW_DIR/data/xingtong_wav_24k_trim"

# CosyVoice SFT 数据根目录
export XINGTONG_SFT_ROOT="$WORKFLOW_DIR/data/xingtong_sft"

# 输出的可复用模型目录
export XINGTONG_MODEL_DIR="$WORKFLOW_DIR/pretrained_models/Fun-CosyVoice3-0.5B-XingTong-sft"
```

---

## 1) 环境准备（uv + workflow venv）

```bash
cd "$WORKFLOW_DIR"
uv python install 3.10
uv sync --python 3.10 --extra asr
```

（必要时）初始化 CosyVoice 子模块（Matcha-TTS）：

```bash
cd "$COSYVOICE_VENDOR_DIR"
git submodule update --init --recursive
```

---

## 2) 数据预处理（原始 wav -> 24k/mono/PCM16 + 去首尾静音）

### 2.1 串行 bash（最稳、依赖最少）

```bash
mkdir -p "$XINGTONG_WAV_DIR"

dur() { ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1" 2>/dev/null || echo 0; }

for f in "$XINGTONG_SRC_WAV_DIR"/*.wav; do
  base="$(basename "$f" .wav)"
  tmp="$XINGTONG_WAV_DIR/${base}.tmp.wav"
  out="$XINGTONG_WAV_DIR/${base}.wav"

  # 转 wav（24k/mono/PCM16）
  ffmpeg -y -hide_banner -loglevel error -i "$f" -ac 1 -ar 24000 -c:a pcm_s16le "$tmp"

  # 去首尾静音（阈值：0.15s / -35d）
  # 若裁剪过度（原始>1.5s 但输出<1.5s）则回退到未裁剪版本
  if ! sox "$tmp" "$out" silence 1 0.15 -35d reverse silence 1 0.15 -35d reverse; then
    cp -f "$tmp" "$out"
  else
    d_tmp="$(dur "$tmp")"
    d_out="$(dur "$out")"
    if python3 -c "import sys; dt=float(sys.argv[1] or 0); do=float(sys.argv[2] or 0); sys.exit(0 if (dt>1.5 and do<1.5) else 1)" "$d_tmp" "$d_out"; then
      cp -f "$tmp" "$out"
    fi
  fi

  rm -f "$tmp"
done
```

### 2.2 GNU parallel（可选，更快）

> 仅在你已安装 `parallel` 且熟悉时使用。

---

## 3) 生成 SFT 数据（Whisper 转写 -> kaldi 目录）

目标产物：
- `data/xingtong_sft/metadata.jsonl`
- `data/xingtong_sft/train/{wav.scp,text,utt2spk,spk2utt,instruct}`
- `data/xingtong_sft/dev/{...}`

推荐命令（faster-whisper large-v3 + VAD）：

```bash
cd "$WORKFLOW_DIR"

uv run python tools/prepare_xuan_sft_dataset.py \
  --wav_dir "$XINGTONG_WAV_DIR" \
  --out_root "$XINGTONG_SFT_ROOT" \
  --spk_id "$XINGTONG_SPK_ID" \
  --backend faster-whisper \
  --whisper_model large-v3 \
  --device cuda \
  --compute_type int8_float16 \
  --vad_filter \
  --language zh \
  --max_sec 30 \
  --train_ratio 0.98 \
  --seed 1986
```

质检建议（强烈建议做 5–10 条抽查）：
- `metadata.jsonl` 的 `text` 与音频大致匹配
- 错词/漏词会直接影响 LLM 对齐上限

---

## 4) 提取说话人 embedding（campplus.onnx）

```bash
cd "$WORKFLOW_DIR"
uv run python tools/extract_embedding.py --dir "$XINGTONG_SFT_ROOT/train" --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
uv run python tools/extract_embedding.py --dir "$XINGTONG_SFT_ROOT/dev"   --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
```

验收：`spk2embedding.pt` 中必须包含 key：`XingTong`。

---

## 5) 生成 parquet + data.list

对 1563 条数据，推荐每片 200 条、并发按 CPU 调整：

```bash
cd "$WORKFLOW_DIR"
mkdir -p "$XINGTONG_SFT_ROOT/train/parquet" "$XINGTONG_SFT_ROOT/dev/parquet"

uv run python tools/make_parquet_list.py \
  --num_utts_per_parquet 200 --num_processes 8 \
  --src_dir "$XINGTONG_SFT_ROOT/train" --des_dir "$XINGTONG_SFT_ROOT/train/parquet"

uv run python tools/make_parquet_list.py \
  --num_utts_per_parquet 200 --num_processes 8 \
  --src_dir "$XINGTONG_SFT_ROOT/dev" --des_dir "$XINGTONG_SFT_ROOT/dev/parquet"
```

---

## 6) 训练（LLM + Flow）

建议环境变量（稳妥配置）：

```bash
cd "$WORKFLOW_DIR"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="$WORKFLOW_DIR:$COSYVOICE_VENDOR_DIR:$COSYVOICE_VENDOR_DIR/third_party/Matcha-TTS:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export COSYVOICE_ORT_FORCE_CPU=1
```

### 6.0 TensorBoard（只看本次 XingTong 训练 / Loss WebUI）

> 关键点：**不要把 logdir 指向父目录**（例如 `tensorboard/`），而是指向本次 run 的唯一目录；这样就不会混入历史（包括 xuan）的曲线。

推荐做法：每次开训先生成一个 `RUN_TAG`，并把它复用到“恢复训练”里。

```bash
cd "$WORKFLOW_DIR"

# 生成本次 run 的唯一标签（也可以手动指定一个固定值，方便你断点恢复时复用）
export RUN_TAG="$(date +%Y%m%d_%H%M%S)"

# LLM / Flow 分开一套 TB 目录，互不干扰
export TB_LLM_DIR="tensorboard/xingtong_sft/llm/torch_ddp/${RUN_TAG}"
export TB_FLOW_DIR="tensorboard/xingtong_sft/flow/torch_ddp/${RUN_TAG}"
mkdir -p "$TB_LLM_DIR" "$TB_FLOW_DIR"

# 单独开一个终端跑（WebUI）
uv run tensorboard --logdir "$TB_LLM_DIR" --port 6006 --bind_all

# 浏览器访问（WSL 本机一般直接用 Windows 浏览器打开）
# http://localhost:6006
```

如果要看 Flow，则把 `--logdir` 换成 `$TB_FLOW_DIR`（或起第二个 tensorboard 端口）。

#### 6.0.1 常见问题：TensorBoard 显示 `No Runs` / 看不到 loss

出现 `No Runs` 基本只有一种原因：你传给 `--logdir` 的目录里**没有** `events.out.tfevents.*` 文件（通常是变量没设置或指错目录）。

快速排查（确认 event 文件存在）：

```bash
cd "$WORKFLOW_DIR"
find tensorboard/xingtong_sft -type f -name "events.out.tfevents*" | tail
```

只看“本次最新 run”的最稳命令（自动选择最新目录）：

```bash
cd "$WORKFLOW_DIR"

# LLM（取最近一次启动写入的 tensorboard run 目录）
TB_DIR="$(ls -dt tensorboard/xingtong_sft/llm/torch_ddp/* | head -n 1)"
echo "TB_DIR=$TB_DIR"
uv run tensorboard --logdir "$TB_DIR" --port 6006 --bind_all --reload_interval 5
```

> 备注：loss 标量点只有在训练跑过 `train_conf.log_interval`（本项目常见为每 100 step 写一次）之后才会出现；刚启动时曲线为空是正常的。

### 6.1 训练 LLM

开始训练（建议把日志落盘，便于估算速度与排障）：

```bash
cd "$WORKFLOW_DIR"

export MODEL_LLM_DIR="exp/xingtong_sft/llm/torch_ddp"
mkdir -p "$MODEL_LLM_DIR"

: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${TB_LLM_DIR:=tensorboard/xingtong_sft/llm/torch_ddp/${RUN_TAG}}"
mkdir -p "$TB_LLM_DIR"

uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xingtong_sft_llm.yaml \
  --train_data "$XINGTONG_SFT_ROOT/train/parquet/data.list" \
  --cv_data    "$XINGTONG_SFT_ROOT/dev/parquet/data.list" \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir "$MODEL_LLM_DIR" \
  --tensorboard_dir "$TB_LLM_DIR" \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 2 \
  --use_amp 2>&1 | tee "$MODEL_LLM_DIR/train_${RUN_TAG}.log"
```

监控（是否按 30 分钟落“安全 checkpoint”）：

```bash
watch -n 60 'ls -lt exp/xingtong_sft/llm/torch_ddp/epoch_*_step_*.pt 2>/dev/null | head'
```

> 说明：`save_interval_sec`（默认 1800 秒）是在 YAML 的 `train_conf.save_interval_sec` 里控制的；**改了 YAML 要重启训练进程才会生效**。  
> 这些 `epoch_<E>_step_<S>.pt` 是“可断点续跑”的轻量 checkpoint（`model_only`，优化器状态会重置）。

恢复训练（从最近一次安全 checkpoint 继续）：

```bash
cd "$WORKFLOW_DIR"

export MODEL_LLM_DIR="exp/xingtong_sft/llm/torch_ddp"
export CKPT_LLM="$(ls -t ${MODEL_LLM_DIR}/epoch_*_step_*.pt | head -n 1)"
echo "resume from: $CKPT_LLM"

: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${TB_LLM_DIR:=tensorboard/xingtong_sft/llm/torch_ddp/${RUN_TAG}}"
mkdir -p "$TB_LLM_DIR"

uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xingtong_sft_llm.yaml \
  --train_data "$XINGTONG_SFT_ROOT/train/parquet/data.list" \
  --cv_data    "$XINGTONG_SFT_ROOT/dev/parquet/data.list" \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint "$CKPT_LLM" \
  --model_dir "$MODEL_LLM_DIR" \
  --tensorboard_dir "$TB_LLM_DIR" \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 2 \
  --use_amp 2>&1 | tee -a "$MODEL_LLM_DIR/train_${RUN_TAG}.log"
```

### 6.2 训练 Flow

开始训练：

```bash
cd "$WORKFLOW_DIR"

export MODEL_FLOW_DIR="exp/xingtong_sft/flow/torch_ddp"
mkdir -p "$MODEL_FLOW_DIR"

: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${TB_FLOW_DIR:=tensorboard/xingtong_sft/flow/torch_ddp/${RUN_TAG}}"
mkdir -p "$TB_FLOW_DIR"

uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xingtong_sft_flow.yaml \
  --train_data "$XINGTONG_SFT_ROOT/train/parquet/data.list" \
  --cv_data    "$XINGTONG_SFT_ROOT/dev/parquet/data.list" \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model flow \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --model_dir "$MODEL_FLOW_DIR" \
  --tensorboard_dir "$TB_FLOW_DIR" \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 2 \
  --use_amp 2>&1 | tee "$MODEL_FLOW_DIR/train_${RUN_TAG}.log"
```

恢复训练（从最近一次安全 checkpoint 继续）：

```bash
cd "$WORKFLOW_DIR"

export MODEL_FLOW_DIR="exp/xingtong_sft/flow/torch_ddp"
export CKPT_FLOW="$(ls -t ${MODEL_FLOW_DIR}/epoch_*_step_*.pt | head -n 1)"
echo "resume from: $CKPT_FLOW"

: "${RUN_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${TB_FLOW_DIR:=tensorboard/xingtong_sft/flow/torch_ddp/${RUN_TAG}}"
mkdir -p "$TB_FLOW_DIR"

uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xingtong_sft_flow.yaml \
  --train_data "$XINGTONG_SFT_ROOT/train/parquet/data.list" \
  --cv_data    "$XINGTONG_SFT_ROOT/dev/parquet/data.list" \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model flow \
  --checkpoint "$CKPT_FLOW" \
  --model_dir "$MODEL_FLOW_DIR" \
  --tensorboard_dir "$TB_FLOW_DIR" \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 2 \
  --use_amp 2>&1 | tee -a "$MODEL_FLOW_DIR/train_${RUN_TAG}.log"
```

验收：
- `exp/xingtong_sft/llm/torch_ddp/epoch_*_whole.pt`
- `exp/xingtong_sft/flow/torch_ddp/epoch_*_whole.pt`

---

## 7) 平均 checkpoint + 组装 XingTong 模型目录 + 生成 spk2info

```bash
cd "$WORKFLOW_DIR"

uv run python "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/average_model.py" \
  --dst_model exp/xingtong_sft/llm/torch_ddp/llm_avg.pt  --src_path exp/xingtong_sft/llm/torch_ddp  --num 3 --val_best

uv run python "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/average_model.py" \
  --dst_model exp/xingtong_sft/flow/torch_ddp/flow_avg.pt --src_path exp/xingtong_sft/flow/torch_ddp --num 5 --val_best

uv run python tools/assemble_xuan_sft_model_dir.py --overwrite \
  --base_model_dir pretrained_models/Fun-CosyVoice3-0.5B \
  --out_model_dir  "$XINGTONG_MODEL_DIR" \
  --llm_pt  exp/xingtong_sft/llm/torch_ddp/llm_avg.pt \
  --flow_pt exp/xingtong_sft/flow/torch_ddp/flow_avg.pt

uv run python tools/make_spk2info_from_spk2embedding.py \
  --spk2embedding_pt "$XINGTONG_SFT_ROOT/train/spk2embedding.pt" \
  --spk_id "$XINGTONG_SPK_ID" \
  --out_spk2info_pt "$XINGTONG_MODEL_DIR/spk2info.pt"
```

验收：
- `pretrained_models/Fun-CosyVoice3-0.5B-XingTong-sft/spk2info.pt` 存在且包含 `XingTong`

---

## 8) 推理（参数甜点位 + A/B base flow）

推理命令模板（只记录参数）：

```bash
cd "$WORKFLOW_DIR"
export COSYVOICE_ORT_FORCE_CPU=1

uv run python tools/infer_sft.py \
  --model_dir "$XINGTONG_MODEL_DIR" \
  --spk_id "$XINGTONG_SPK_ID" \
  --text_file <TEXT_FILE> \
  --out_dir  <OUT_DIR> \
  --llm_ckpt  exp/xingtong_sft/llm/torch_ddp/epoch_<N>_whole.pt \
  --flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --prompt_text "<SHORT_PROMPT_FROM_METADATA><|endofprompt|>" \
  --seed 1986 \
  --temperature 1.0 \
  --top_p 0.6 --top_k 10 --win_size 10 --tau_r 1.0 \
  --speed 0.95
```

统一试听命名（命名=参数）：

```bash
python3 tools/collect_listen_wavs.py \
  --src <OUT_ROOT> \
  --dst <LISTEN_DIR> \
  --mode copy \
  --spk_id "$XINGTONG_SPK_ID" \
  --include_params
```

说明：
- `tools/infer_sft.py` 每次推理只输出一个文件：`chunk_0000.wav`（位于 `<OUT_DIR>`）。
- `collect_listen_wavs.py` 默认收集 `chunk_0000.wav`；如你要收集旧的 `full.wav` 产物，请显式传 `--pattern '**/full*.wav'`。
