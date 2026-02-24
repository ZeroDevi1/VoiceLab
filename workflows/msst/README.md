# MSST workflow（一键音频处理链）

把 **MSST-WebUI** 的常用处理链（分离伴奏/干声、去和声、去混响、降噪）做成脚本化 workflow。

## 一次处理会输出什么？

一次处理会输出这些文件（后缀与 WebUI 习惯一致；默认输出格式为 `wav`）：

1. 伴奏：`*_other.wav`（inst_v1e）
2. 干声：`*_vocals.wav`（big_beta5e）
3. 干声和声：`*_vocals_other.wav`（karaoke）
4. 去和声干声：`*_vocals_karaoke.wav`（karaoke）
5. 去和声干声去混响：`*_vocals_karaoke_noreverb.wav`（dereverb）
6. 去和声干声降噪：`*_vocals_karaoke_noreverb_dry.wav`（denoise）

## 快速开始（WSL / Ubuntu 24.04）

1) 同步 vendor（会 clone/pull 上游仓库到 `../../vendor/`；`vendor/` 在本 repo 中被忽略）：

```bash
cd ~/AntiGravityProjects/VoiceLab
uv run -m voicelab vendor sync
```

2) 初始化 MSST workflow 环境（Python 3.10）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/msst
uv python install 3.10
uv python pin 3.10
uv sync
```

3) 初始化 runtime（不污染 vendor；会在 `workflows/msst/runtime/` 建立运行目录，并从 `/mnt/c/AIGC/MSST-WebUI/pretrain` 拷贝需要的模型）：

```bash
uv run python tools/msst_init_runtime.py
```

4) 一键处理：

```bash
uv run python tools/msst_process_chain.py --input /path/to/song.wav --output-dir /path/to/out_dir
```

不指定 `--output-dir` 时，默认输出在：`workflows/msst/out_wav/`

## 输入/输出路径怎么指定？

`tools/msst_process_chain.py` 关键参数：

- `--input`：输入音频文件或目录（目录模式会处理常见音频后缀）
- `--output-dir`：输出目录（默认：`workflows/msst/out_wav`）
- `--device`：`auto|cuda|cpu`（默认 `auto`）
- `--gpu-ids`：GPU id 列表（默认 `0`，例如 `0,1`）
- `--output-format`：`wav|flac|mp3`（默认 `wav`）
- `--wav-bit-depth`：`FLOAT|PCM_16|PCM_24`（默认 `FLOAT`）
- `--no-ffmpeg-preconvert`：默认会用 `ffmpeg` 预转换为 `wav + 44100Hz + stereo`；加此参数可禁用

示例（输入单文件，输出到指定目录）：

```bash
uv run python tools/msst_process_chain.py \
  --input '/mnt/c/AIGC/音乐/栞 - MyGO!!!!!.flac' \
  --output-dir '/mnt/c/AIGC/音乐/栞'
```

## 处理链与模型参数（对应 `tools/msst_process_chain.py`）

说明：
- 每一步都使用 `runtime/configs/**` 里的 **默认配置**（这些配置来自 `vendor/MSST-WebUI/configs_backup`，避免被 WebUI 保存配置污染）
- `use_tta=False`
- 默认输出 `wav`，subtype 默认 `FLOAT`
- 默认会把输入预转换为 `44100Hz + stereo wav`（更稳定、与默认 config 对齐）

### Step0：伴奏（Instrumental / Other）

- 模型：`inst_v1e.ckpt`
- 配置：`runtime/configs/vocal_models/inst_v1e.ckpt.yaml`
- 默认参数：`sr=44100, chunk_size=485100, overlap(num_overlap)=4, batch_size=1`
- 输入：预转换后的 `*.wav`
- 输出：`*_other.wav`（仅保存 `other`）

### Step1：干声（Lead Vocals）

- 模型：`big_beta5e.ckpt`
- 配置：`runtime/configs/vocal_models/big_beta5e.ckpt.yaml`
- 默认参数：`sr=44100, chunk_size=485100, overlap(num_overlap)=2, batch_size=1`
- 输入：预转换后的 `*.wav`
- 输出：`*_vocals.wav`（仅保存 `vocals`）

### Step2：去和声/和声分离（Karaoke + Backing）

- 模型：`model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt`
- 配置：`runtime/configs/vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt.yaml`
- 默认参数：`sr=44100, chunk_size=352800, overlap(num_overlap)=4, batch_size=1`
- 输入：Step1 的 `*_vocals.wav`
- 输出：
  - `*_vocals_karaoke.wav`（仅主唱/去和声干声）
  - `*_vocals_other.wav`（和声/Backing）

### Step3：去混响（De-Reverb）

- 模型：`dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt`
- 配置：`runtime/configs/single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt.yaml`
- 默认参数：`sr=44100, chunk_size=352800, overlap(num_overlap)=4, batch_size=1`
- 输入：Step2 的 `*_vocals_karaoke.wav`
- 输出：`*_vocals_karaoke_noreverb.wav`（仅保存 `noreverb`）

### Step4：降噪（De-Noise）

- 模型：`denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`
- 配置：`runtime/configs/single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt.yaml`
- 默认参数：`sr=44100, chunk_size=352800, overlap(num_overlap)=4, batch_size=1`
- 输入：Step3 的 `*_vocals_karaoke_noreverb.wav`
- 输出：`*_vocals_karaoke_noreverb_dry.wav`（仅保存 `dry`）

## Runtime / 模型来源（对应 `tools/msst_init_runtime.py`）

runtime 目录结构（核心）：

- `workflows/msst/runtime/inference|modules|utils`：从 `vendor/MSST-WebUI` 软链
- `workflows/msst/runtime/configs`：从 `vendor/MSST-WebUI/configs_backup` 复制（“干净默认值”）
- `workflows/msst/runtime/pretrain/**`：模型文件（大文件）

模型拷贝的默认来源：

- `/mnt/c/AIGC/MSST-WebUI/pretrain`（可通过 `--assets-src` 或环境变量 `MSST_ASSETS_SRC_DIR` 覆盖）

你也可以通过环境变量覆盖路径：

- `MSST_VENDOR_DIR`：上游 MSST-WebUI repo 路径（默认 `vendor/MSST-WebUI`）
- `MSST_RUNTIME_DIR`：runtime 路径（默认 `workflows/msst/runtime`）
- `MSST_ASSETS_SRC_DIR`：本地模型来源路径（默认 `/mnt/c/AIGC/MSST-WebUI/pretrain`）

## 说明

- 需要系统已安装 `ffmpeg`（用于输入预转换）。
- 如果新环境没有 `/mnt/c/AIGC/MSST-WebUI/pretrain`，可用：
  - `uv run python tools/msst_download_models.py`（默认走 `https://hf-mirror.com`，可用 `--hf-base` 覆盖）
