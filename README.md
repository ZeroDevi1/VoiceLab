# VoiceLab

一个用于 **语音相关项目** 的工作区，目标是把各个上游项目（CosyVoice / GPT-SoVITS / RVC）当作 `vendor/` 依赖管理，而把你自己的 **数据准备、训练配置、训练产物与脚本** 保持在 `workflows/` 中，避免“污染上游仓库”。

文档索引：`docs/index.md`

## 目录约定

- `vendor/`：放上游仓库（整个目录在本工作区 git 中忽略）
- `workflows/`：放你自己的训练流程、脚本、配置与文档（本工作区 git 只追踪这里的内容）

## 快速开始（CosyVoice）

> 说明：`uv sync` 只负责同步 Python 依赖与虚拟环境，不支持“自动执行 git clone/pull”。  
> 因此本项目提供 `voicelab` 小工具来同步 `vendor/`（仍然是 **零污染上游仓库**，因为 `vendor/` 整体不进 git）。

0) Clone 后约定仓库根目录（避免强依赖本机固定路径）：

```bash
git clone git@github.com:ZeroDevi1/VoiceLab.git
cd VoiceLab
export VOICELAB_DIR="$PWD"
```

> 如果你不是在仓库根目录执行，可用：
>
> ```bash
> export VOICELAB_DIR="$(git rev-parse --show-toplevel)"
> ```

1) 同步 vendor（会 clone/pull 上游仓库）：

```bash
cd "$VOICELAB_DIR"
uv run -m voicelab init
```

2) 初始化 CosyVoice workflow 环境：

```bash
cd "$VOICELAB_DIR/workflows/cosyvoice"
uv sync
```

完整流程见：`docs/workflows/cosyvoice/cosyvoice_xuan_sft_wsl_ubuntu2404.md`

新环境推荐一键初始化（vendor + uv sync + 模型下载 + runtime init）：

```bash
cd "$VOICELAB_DIR"
uv run -m voicelab bootstrap
```

### 全局共享缓存说明（训练/推理代码无需修改）

`bootstrap` 默认会把大模型/资产下载到全局共享缓存：
- `~/.cache/voicelab/assets`

并通过各 workflow 的 `*_init_runtime.py` 把这些文件 **软链到 `workflows/*/runtime/`**（RVC/MSST），或在 workflow 内创建固定相对路径的软链（CosyVoice 的 `workflows/cosyvoice/pretrained_models`）。

因此你后续照常执行训练/推理脚本即可：训练/推理代码仍然只读 `runtime/`（或 workflow 内相对路径），**不需要改代码、不需要额外传 `--assets-src`**。
