# 在 WSL2 Ubuntu 24.04 用 uv 训练 CosyVoice 3「说话人 SFT」（VoiceLab：上游 CosyVoice 作为 vendor，workflow 独立）

你当前约束：
- WSL2：Ubuntu 24.04
- VoiceLab 主项目：`~/AntiGravityProjects/VoiceLab`
- CosyVoice（上游 vendor）：`~/AntiGravityProjects/VoiceLab/vendor/CosyVoice`（本项目 git 忽略，可随时删/重拉）
- 本次 workflow：`~/AntiGravityProjects/VoiceLab/workflows/cosyvoice`
- 数据集路径：`/mnt/c/AIGC/数据集/xuan`

> 性能提示：这份文档假设你已把 **workflow 与训练产物**（`data/`、`exp/`、`pretrained_models/`、`tensorboard/` 等）放在 WSL 的 Linux 文件系统（即 `~/AntiGravityProjects/VoiceLab/workflows/cosyvoice`）。这样速度通常会明显快于把这些目录放在 `/mnt/c`。  
> 训练集原始 wav 仍保留在 `/mnt/c/AIGC/数据集/xuan` 属于正常做法（空间/管理方便），只是读取原始数据时仍会受到一点 `/mnt/c` IO 影响。

建议先在 WSL 终端里约定几个变量（后文命令会用到）：

```bash
export VOICELAB_DIR=~/AntiGravityProjects/VoiceLab
export WORKFLOW_DIR="$VOICELAB_DIR/workflows/cosyvoice"
export COSYVOICE_VENDOR_DIR="$VOICELAB_DIR/vendor/CosyVoice"
export XUAN_WAV_DIR="/mnt/c/AIGC/数据集/xuan"
```

---

## 0) WSL GPU 基础检查

在 WSL 终端里：

```bash
nvidia-smi
```

看到 GPU 信息才算 WSL GPU 正常。

---

## 1) 系统依赖（建议安装）

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg sox libsox-dev libsndfile1 unzip build-essential
```

---

## 2) uv + Python 3.10 + 依赖安装

### 2.0 拉取/更新上游 CosyVoice（vendor）

推荐在 VoiceLab 根目录用 `voicelab` 工具同步全部 vendor（CosyVoice / GPT-SoVITS / RVC）：

```bash
cd "$VOICELAB_DIR"
uv run -m voicelab vendor sync
```

如果你想手动管理（不推荐），也可以自己 `git clone` 到 `vendor/`。

如果你本机已经有一个 CosyVoice 仓库（例如 `~/AntiGravityProjects/CosyVoice`），也可以先用软链接快速接上（后续再换成干净 clone 也行）：

```bash
cd "$VOICELAB_DIR"
ln -s ~/AntiGravityProjects/CosyVoice vendor/CosyVoice
```

进入 workflow 目录（你的数据/产物/脚本都在这里）：

```bash
cd "$WORKFLOW_DIR"
```

### 2.1 CosyVoice 子模块（必须）

CosyVoice3 的配置会用到 `third_party/Matcha-TTS`（例如 `matcha.utils.audio.mel_spectrogram`）。如果不初始化子模块会直接报错。

```bash
cd "$COSYVOICE_VENDOR_DIR"
git submodule update --init --recursive
```

### 2.2 （国内网络推荐）Hugging Face 镜像与缓存位置

`faster-whisper` 下载 Whisper（CTranslate2）模型默认走 Hugging Face。国内网络建议先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
```

> 依赖坑：如果训练时报 `ModuleNotFoundError: No module named 'pkg_resources'`，通常是 `setuptools` 版本过新导致不再提供 `pkg_resources`。  
> 处理方式（推荐）：降级到旧版 `setuptools`（例如 69.x）：
>
> ```bash
> uv pip install --python .venv/bin/python "setuptools==69.5.1"
> ```

> 如果你看到 `Cannot load the target vocabulary from the model directory`，通常是 Hugging Face 缓存里模型文件下载不完整/损坏。  
> 处理方式：删除对应缓存目录后重新运行（例如 `rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3`）。

安装 uv（若已安装可跳过）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### 2.3 用 `uv sync` 初始化本 workflow 环境（推荐）

本 workflow 的 `pyproject.toml` 已锁定 `Python>=3.10,<3.11`，并内置了上游 CosyVoice 的依赖列表（含 `setuptools==69.5.1` 用于解决 `pkg_resources` 兼容问题）。  
此外，本目录的 `uv.toml` 预置了 PyTorch / ORT 的额外 index（用于安装 GPU wheels）。

初始化环境：

```bash
cd "$WORKFLOW_DIR"
uv sync
```

（可选）安装 ASR 依赖（faster-whisper）：

```bash
cd "$WORKFLOW_DIR"
uv sync --extra asr
```

> 如果你网络环境无法访问 `uv.toml` 里的 index，可自行编辑 `uv.toml`，或者在命令行里用 `uv sync --default-index ... --index ...` 覆盖。

验证 ORT provider（仅检查“声明可用”，不代表一定能成功加载 CUDA DLL；后面训练可用 CPU 回退兜底）：

```bash
cd "$WORKFLOW_DIR"
uv run python tools/diag_ort.py
```

---

## 3) （可选）ASR 用 faster-whisper（Large-v3 INT8）

`faster-whisper` 在 6GB 显存上跑 `large-v3` 的 INT8/INT8+FP16 通常更快更准（中文也更稳）。

安装方式：见上面 `uv sync --extra asr`。

---

## 4) 下载 CosyVoice3 预训练模型

下载到：`pretrained_models/Fun-CosyVoice3-0.5B`

```bash
cd "$WORKFLOW_DIR"
uv run python tools/download_pretrained_cosyvoice3.py --source modelscope
```

---

## 5) 数据准备（转写 -> kaldi -> embedding -> parquet）

### 5.1 Whisper 转写 + 生成 `data/xuan_sft`

（推荐）用 faster-whisper + VAD：

```bash
cd "$WORKFLOW_DIR"
uv run python tools/prepare_xuan_sft_dataset.py --wav_dir "$XUAN_WAV_DIR" \
  --backend faster-whisper --whisper_model large-v3 --device cuda --compute_type int8_float16 --vad_filter
```

> 说明：脚本默认启用 `wetext` 文本归一化，首次运行会下载 `wetext` 模型（走 ModelScope 缓存）。  
> 如果你想跳过这一步，追加 `--no_wetext`（会更快，但归一化鲁棒性略低）。

你也可以先试跑 10 条确认速度：

```bash
cd "$WORKFLOW_DIR"
uv run python tools/prepare_xuan_sft_dataset.py --wav_dir "$XUAN_WAV_DIR" \
  --backend faster-whisper --whisper_model large-v3 --device cuda --compute_type int8_float16 --vad_filter --limit 10
```

### 5.2 提取 embedding（campplus.onnx）

```bash
cd "$WORKFLOW_DIR"
uv run python tools/extract_embedding.py --dir data/xuan_sft/train --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
uv run python tools/extract_embedding.py --dir data/xuan_sft/dev   --onnx_path pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
```

> 迁移提示：如果你的 `data/xuan_sft/*/wav.scp` 是在 Windows 上生成的，里面可能是 `C:\\...` 路径。  
> 这几个工具（`extract_embedding.py` / `extract_speech_token.py` / `make_parquet_list.py`）会在 WSL 下自动把 `C:\\...` 转成 `/mnt/c/...`，无需手动改文件。

### 5.3 生成 parquet + data.list

```bash
cd "$WORKFLOW_DIR"
mkdir -p data/xuan_sft/train/parquet data/xuan_sft/dev/parquet
uv run python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 --src_dir data/xuan_sft/train --des_dir data/xuan_sft/train/parquet
uv run python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 4 --src_dir data/xuan_sft/dev   --des_dir data/xuan_sft/dev/parquet
```

> 性能建议（推荐按 `num_workers` 调分片）：
>
> - `--num_utts_per_parquet` 控制每个 parquet 里包含多少条样本；越小 => 分片越多。
> - 经验法则：让 `data.list` 里的分片数量 `>= num_workers * 2 ~ 4`，这样多 worker 才能均匀读不同文件，减少 IO 争用。
> - 对你这种几百条规模的数据，常用取值：
>   - `num_workers=4`：`--num_utts_per_parquet 50`（通常会得到 ~8 个分片）
>   - `num_workers=6`：`--num_utts_per_parquet 30~50`
>
> 你可以用下面命令确认分片数量（行数就是分片数）：
>
> ```bash
> wc -l data/xuan_sft/train/parquet/data.list
> ```

---

## 6) 训练（llm 必做，flow 可选）

### 6.0 重要环境变量

训练前建议设置：

```bash
cd "$WORKFLOW_DIR"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="$WORKFLOW_DIR:$COSYVOICE_VENDOR_DIR:$COSYVOICE_VENDOR_DIR/third_party/Matcha-TTS:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

#### ORT CUDA 依赖问题（强烈建议先用 CPU 兜底）

即使在 WSL2，`onnxruntime-gpu` 也可能因为缺少系统 CUDA/cuDNN 动态库而无法加载 CUDA EP。

为保证能开跑，建议先加：

```bash
export COSYVOICE_ORT_FORCE_CPU=1
```

这样训练仍然用 **PyTorch GPU**，只是 ORT 的那部分（在线 speech-token extractor）用 CPU。

### 6.1 训练 llm

```bash
cd "$WORKFLOW_DIR"
taskset -c 0-11 uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xuan_sft.yaml \
  --train_data data/xuan_sft/train/parquet/data.list \
  --cv_data    data/xuan_sft/dev/parquet/data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir exp/xuan_sft/llm/torch_ddp \
  --tensorboard_dir tensorboard/xuan_sft/llm/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 6 \
  --prefetch 2 \
  --use_amp
```

> DataLoader 参数说明与推荐：
>
> - `--prefetch` 对应 PyTorch 的 `prefetch_factor`，单位是「每个 worker 预取的 batch 数」。总预取量约等于 `num_workers * prefetch` 个 batch，设太大容易吃爆内存。
> - `--num_workers 0` 时 PyTorch 不支持 `prefetch_factor`；此时不要传 `--prefetch`（或把它设置为一个不会生效的值并确认你的 PyTorch 版本是否允许）。
> - 推荐起步：
>   - `num_workers=4`：`--prefetch 2`（卡数据再试 `--prefetch 4`）
>   - `num_workers=6`：`--prefetch 2`（卡数据再试 `--prefetch 4`）
>
> checkpoint 保存频率（可选）：
>
> - 默认情况下（本配置 `save_per_step: -1`），训练会在每个 epoch 结束时做一次 CV 并保存 `epoch_{N}_whole.pt`。
> - 如果你把 `configs/cosyvoice3_xuan_sft.yaml` 里的 `train_conf.save_per_step` 改成 `100`，就会每 100 个 step 触发一次 **CV + 保存**，生成类似 `epoch_15_step_100.pt` 的文件；会明显拖慢训练（因为多跑了很多次 CV），一般不建议开得太频繁。

### 6.2 （可选）训练 flow

```bash
cd "$WORKFLOW_DIR"
taskset -c 0-11 uv run torchrun --standalone --nproc_per_node=1 "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/train.py" \
  --train_engine torch_ddp \
  --config configs/cosyvoice3_xuan_sft.yaml \
  --train_data data/xuan_sft/train/parquet/data.list \
  --cv_data    data/xuan_sft/dev/parquet/data.list \
  --qwen_pretrain_path pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/Fun-CosyVoice3-0.5B \
  --model flow \
  --checkpoint pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --model_dir exp/xuan_sft/flow/torch_ddp \
  --tensorboard_dir tensorboard/xuan_sft/flow/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 6 \
  --prefetch 2 \
  --use_amp
```

---

## 7) 平均 checkpoint + 组装模型 + 生成 spk2info

```bash
cd "$WORKFLOW_DIR"
uv run python "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/average_model.py" --dst_model exp/xuan_sft/llm/torch_ddp/llm_avg.pt  --src_path exp/xuan_sft/llm/torch_ddp  --num 3 --val_best
uv run python "$COSYVOICE_VENDOR_DIR/cosyvoice/bin/average_model.py" --dst_model exp/xuan_sft/flow/torch_ddp/flow_avg.pt --src_path exp/xuan_sft/flow/torch_ddp --num 5 --val_best

uv run python tools/assemble_xuan_sft_model_dir.py --overwrite \
  --llm_pt exp/xuan_sft/llm/torch_ddp/llm_avg.pt

uv run python tools/make_spk2info_from_spk2embedding.py \
  --spk2embedding_pt data/xuan_sft/train/spk2embedding.pt \
  --spk_id xuan \
  --out_spk2info_pt pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft/spk2info.pt
```

> 如果你没训 `flow`，`flow_avg.pt` 那行可以跳过，组装时会保留 base 的 `flow.pt`。

---

## 8) 推理（SFT）

> 如果你提示 `model_dir not found`（例如 `pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft` 不存在），先完成 **第 7 步组装模型目录**（至少需要把 base 目录复制/组装出来，并生成 `spk2info.pt`）。

```bash
cd "$WORKFLOW_DIR"
echo "你好，这是一个推理测试。" > input.txt
uv run python tools/infer_sft.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft \
  --spk_id xuan \
  --text_file ./input.txt \
  --out_dir ./out_wav
```

> SFT 推理不需要“参考音频”。只要 `--model_dir` 里有 `spk2info.pt`，且包含 `spk_id=xuan` 的 embedding，就可以直接合成。
>
> 另外：CosyVoice3 的 LLM 推理需要 `prompt_text` 或正文中包含特殊 token `<|endofprompt|>`（训练时的 `instruct`）。`tools/infer_sft.py` 默认注入最小 marker：`<|endofprompt|>`（减少 prompt 泄露）；如需更强情绪引导可显式传 `--prompt_text \"...<|endofprompt|>\"`。

### 8.1（可选）一键生成固定测试句

```bash
cd "$WORKFLOW_DIR"
uv run python tools/test_xuan_sft_tts.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft \
  --spk_id xuan \
  --out_wav out_wav/xuan_test.wav
```

默认行为：
- 如果存在 `exp/xuan_sft/llm/torch_ddp/epoch_6_whole.pt`，会优先热加载它（更利于你“锁定 Epoch 6”做试听）。
- 否则会自动加载最新的 `exp/xuan_sft/llm/torch_ddp/epoch_*_whole.pt`（如果存在）。
- CosyVoice3 默认使用 `prompt_preset=hype`（`马头有大！马头来啦！<|endofprompt|>`）+ `speed=1.1`，更容易把“爆发力”拉起来；需要更稳的基线可用 `--prompt_preset neutral` + `--speed 1.0`。

你也可以手动指定某个 checkpoint，例如：

```bash
uv run python tools/test_xuan_sft_tts.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft \
  --spk_id xuan \
  --llm_ckpt exp/xuan_sft/llm/torch_ddp/epoch_6_whole.pt \
  --out_wav out_wav/xuan_test_epoch6.wav
```

（可选）一键对比 “neutral/hype × speed 1.0/1.1” 四组试听（输出到 `--out_wav` 所在目录）：

```bash
uv run python tools/test_xuan_sft_tts.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-xuan-sft \
  --spk_id xuan \
  --llm_ckpt exp/xuan_sft/llm/torch_ddp/epoch_6_whole.pt \
  --compare \
  --out_wav out_wav/xuan_compare.wav
```

> 如果你不想触发 `wetext` 下载（或只是想快速回归），可追加 `--no-text-frontend`（归一化会弱一些，但更快）。

---

## 附：faster-whisper 与 onnxruntime 冲突的一键处理（WSL）

如果你安装 faster-whisper 后发现 CosyVoice 相关 ORT 行为异常，可执行：

```bash
cd "$WORKFLOW_DIR"
bash "$COSYVOICE_VENDOR_DIR/tools/setup_asr_faster_whisper.sh"
```
