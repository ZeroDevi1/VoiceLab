# GPT-SoVITS：用 `.list` 准备数据 + 训练入口（WSL / Ubuntu 24.04）

本文档对应 `workflows/gpt_sovits/`：提供 VoiceLab 风格的 **数据准备** 与 **训练入口**，并且对同名 `.list` 有“优先使用”的一致行为。

> `.list` 统一格式见：`docs/datasets/list_annotations.md`

## 0) 前置条件

- 已同步 vendor：`vendor/GPT-SoVITS`
  - `uv run -m voicelab vendor sync`
- 已安装 GPT-SoVITS 依赖（建议在 `workflows/gpt_sovits/` 创建自己的环境后安装）：

```bash
cd "$VOICELAB_DIR/workflows/gpt_sovits"
# 参考：vendor/GPT-SoVITS/requirements.txt（含 --no-binary=opencc）
# 你可以在自己的 venv 里执行：
#   pip install -r "$VOICELAB_DIR/vendor/GPT-SoVITS/requirements.txt"
```

## 1) 数据集与 `.list`

推荐目录结构：

- 数据集目录（音频）：`/mnt/c/AIGC/数据集/<Speaker>` 或 stage 到 `VoiceLab/datasets/<Speaker>`
- 同名 `.list`：`<dataset>/<Speaker>.list`（或 `<dataset>/<speaker>.list`）
- 也可集中放到：`/mnt/c/AIGC/数据集/标注文件/<Speaker>.list`

## 2)（推荐）先 stage 到 ext4（更快）

```bash
cd "$VOICELAB_DIR/workflows/gpt_sovits"
uv run python tools/gpt_sovits_stage_dataset.py \
  --src "/mnt/c/AIGC/数据集/XingTong" \
  --dst "$VOICELAB_DIR/datasets/XingTong"
```

stage 完成后会尽量确保 dst 目录里有同名 `.list`（来自 src 或从“标注文件”目录补拷贝）。

## 3) 数据准备（提取文本/Hubert/语义 token）

只要数据集目录里能找到同名 `.list`，通常只需要一条命令：

```bash
cd "$VOICELAB_DIR/workflows/gpt_sovits"
uv run python tools/gpt_sovits_prepare_dataset.py \
  --dataset-dir "$VOICELAB_DIR/datasets/XingTong"
```

输出：
- 数据集产物：`workflows/gpt_sovits/data/<exp-name>/`
  - `2-name2text.txt`
  - `4-cnhubert/`
  - `5-wav32k/`
  - `6-name2semantic.tsv`
- 配置/辅助输出：`workflows/gpt_sovits/runtime/<exp-name>/configs/`

## 4) 训练（入口脚本）

### 4.1 训练 GPT（S1）

```bash
cd "$VOICELAB_DIR/workflows/gpt_sovits"
uv run python tools/gpt_sovits_train_s1.py --exp-name XingTong
```

### 4.2 训练 SoVITS（S2）

```bash
cd "$VOICELAB_DIR/workflows/gpt_sovits"
uv run python tools/gpt_sovits_train_s2.py --exp-name XingTong
```

> 说明：训练本身依赖 GPT-SoVITS 的完整模型/权重与环境，本 workflow 的目标是提供“稳定的入口 + 同名 `.list` 优先 + 本地数据优先”的一致行为。

