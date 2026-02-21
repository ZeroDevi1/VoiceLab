# VoiceLab

一个用于 **语音相关项目** 的工作区，目标是把各个上游项目（CosyVoice / GPT-SoVITS / RVC）当作 `vendor/` 依赖管理，而把你自己的 **数据准备、训练配置、训练产物与脚本** 保持在 `workflows/` 中，避免“污染上游仓库”。

## 目录约定

- `vendor/`：放上游仓库（整个目录在本工作区 git 中忽略）
- `workflows/`：放你自己的训练流程、脚本、配置与文档（本工作区 git 只追踪这里的内容）

## 快速开始（CosyVoice）

> 说明：`uv sync` 只负责同步 Python 依赖与虚拟环境，不支持“自动执行 git clone/pull”。  
> 因此本项目提供 `voicelab` 小工具来同步 `vendor/`（仍然是 **零污染上游仓库**，因为 `vendor/` 整体不进 git）。

1) 同步 vendor（会 clone/pull 这三个上游仓库）：

```bash
cd ~/AntiGravityProjects/VoiceLab
uv run -m voicelab init
```

2) 初始化 CosyVoice workflow 环境：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/cosyvoice
uv sync
```

完整流程见：`workflows/cosyvoice/docs/cosyvoice_xuan_sft_wsl_ubuntu2404.md`
