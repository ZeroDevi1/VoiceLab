# MSST workflow（一键音频处理链）

把 **MSST-WebUI** 的常用处理链（分离伴奏/干声、去和声、去混响、降噪）做成脚本化 workflow。

一次处理会输出这些文件（后缀与 WebUI 习惯一致）：

1. 伴奏：`*_other.wav`（inst_v1e）
2. 干声：`*_vocals.wav`（big_beta5e）
3. 干声和声：`*_vocals_other.wav`（karaoke_becruily）
4. 去和声干声：`*_vocals_karaoke.wav`（karaoke_becruily）
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
uv run python tools/msst_process_chain.py --input /path/to/song.wav
```

输出默认在：`workflows/msst/out_wav/`

## 说明

- 需要系统已安装 `ffmpeg`（用于输入预转换）。
- 如果新环境没有 `/mnt/c/AIGC/MSST-WebUI/pretrain`，可用：
  - `uv run python tools/msst_download_models.py --hf-base https://hf-mirror.com`

