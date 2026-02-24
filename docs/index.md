# VoiceLab Docs

VoiceLab 是一个语音相关工作区：把上游项目（CosyVoice / RVC / MSST / GPT-SoVITS）当作 `vendor/` 依赖管理，把你的训练流程与脚本放在 `workflows/`，避免“污染上游仓库”。

## 一键初始化（新环境推荐）

在一个全新的机器/WSL 环境中 clone 仓库后，执行一条命令完成：
- vendor 拉取（默认 CN profile：GitHub 镜像前缀 + hf-mirror）
- 各 workflow 的 `uv sync`
- 自动下载所需模型/资产（不依赖系统代理）
- 初始化各 workflow runtime

```bash
cd ~/AntiGravityProjects/VoiceLab
uv run -m voicelab bootstrap
```

常用参数：
- `--dry-run`：只打印将要执行的步骤
- `--assets-dir /path/to/cache`：指定共享缓存目录（默认 `~/.cache/voicelab/assets`）
- `--profile global`：不使用 CN 默认镜像

## Workflows

- CosyVoice（SFT）
  - `docs/workflows/cosyvoice/cosyvoice_xuan_sft_wsl_ubuntu2404.md`
  - `docs/workflows/cosyvoice/cosyvoice_sft_param_playbook_wsl_ubuntu2404.md`
- RVC（训练 + 推理）
  - `docs/workflows/rvc/rvc_xingtong_wsl_ubuntu2404.md`
  - `docs/workflows/rvc/rvc_xuan_wsl_ubuntu2404.md`

## Wiki

本仓库支持把 `docs/` 自动同步到 GitHub Wiki。说明见：`docs/WIKI_SYNC.md`

