# RVC workflow（星瞳训练 + 炫神迁移）

本目录存放 **RVC 的数据准备、训练、索引构建、推理脚本与文档**，并且不把任何内容提交回上游 RVC 仓库。

上游仓库放在 `../../vendor/`（整个 `vendor/` 目录被 git 忽略，需要时自行 clone/pull）。

## 快速开始（WSL / Ubuntu 24.04）

1) 同步 vendor（会 clone/pull 上游仓库；你当前工作区已存在 `vendor/Retrieval-based-Voice-Conversion-WebUI`）：

```bash
cd ~/AntiGravityProjects/VoiceLab
uv run -m voicelab vendor sync
```

2) 初始化 RVC workflow 环境（建议 Python 3.10；若本机没有，可用 uv 管理）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv python install 3.10
uv python pin 3.10
uv sync
```

3) 初始化 runtime（不污染 vendor；会在 `workflows/rvc/runtime/` 建立运行目录，并链接必需的 hubert/rmvpe/预训练权重）：

```bash
uv run python tools/rvc_init_runtime.py
```

完整流程见：`docs/rvc_xingtong_wsl_ubuntu2404.md`
