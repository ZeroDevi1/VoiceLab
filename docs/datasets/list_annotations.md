# `.list` 标注文件：格式、优先级与各 Workflow 的使用方式

本仓库把 GPT-SoVITS 常见的 `.list` 标注文件格式作为**统一的“音频 + 文本标注”载体**，并在不同 workflow 中按需使用字段。

## 1) 格式定义（统一兼容）

每行 4 列，以 `|` 分隔：

```
audio_path|speaker|lang|text
```

- `audio_path`：音频路径（可为绝对路径或相对路径）
- `speaker`：说话人名（可为空；某些流程只作为备注）
- `lang`：语言标记（常见：`ZH/zh/JA/EN/...`）
- `text`：文本标注（TTS/SFT 训练会用到；RVC 不会用到）

注：
- 空行与以 `#` 开头的注释行会被忽略。
- 若行内出现多余的 `|`，会被合并进 `text`（尽量不要在文本里写 `|`）。

## 2) 同名 `.list` 的优先级规则

当某个流程需要在“数据集目录”里自动寻找 `.list` 时，会使用“同名优先”规则：

1. `<dataset_dir>/<dataset_dir.name>.list`
2. `<dataset_dir>/<dataset_dir.name.lower()>.list`
3. 若目录下只有一个 `*.list`，则使用该文件

例如：
- `.../XingTong/XingTong.list`
- `.../Xuan/xuan.list`

## 3) Stage（复制到 ext4）时的 `.list` 处理

在 WSL 场景下，建议把 `/mnt/c/...` 的数据集复制到 ext4（例如 `VoiceLab/datasets/`）提升 I/O。

- `workflows/rvc/tools/rvc_stage_dataset.py`：stage 时会把同目录 `.list` 一并复制；若 dst 中没有同名 `.list`，会尝试从集中目录补拷贝：
  - 默认：`/mnt/c/AIGC/数据集/标注文件/<同名>.list`
- `workflows/gpt_sovits/tools/gpt_sovits_stage_dataset.py`：行为与上面一致

## 4) 各 Workflow 如何使用 `.list`

### 4.1 RVC（训练/预处理）

RVC 训练 **不使用文本标注**，因此：
- `.list` 的 `speaker/lang/text` 在 RVC 中仅作为备注
- RVC 只消费第 1 列 `audio_path`，并把它当作**音频白名单**：
  - 若数据集目录存在同名 `.list`（或你显式传 `--list`），则只预处理 `.list` 中列出的音频
  - 同时会创建一个仅包含音频文件的 `_dataset_input` 目录喂给上游 `preprocess.py`，避免上游误把 `.list` 当音频读取

### 4.2 CosyVoice SFT（数据准备）

CosyVoice SFT 是“音频-文本”监督训练，因此 `text` 不能忽略。

`workflows/cosyvoice/tools/prepare_xuan_sft_dataset.py` 支持：
- 若发现 `.list`，优先使用 `.list` 第 4 列 `text`
- 对于 `.list` 缺失文本的样本，再 fallback 使用 ASR（Whisper / faster-whisper）

映射规则（兼容绝对路径、扩展名不一致等）：
- 用 `.list` 的 `audio_path` 的 `basename` / `stem` 去匹配 `wav_dir/*.wav`

### 4.3 GPT-SoVITS（数据准备/训练）

GPT-SoVITS 训练会使用：
- 第 1 列音频（用于提特征/语义 token）
- 第 3 列语言（用于分词/清洗）
- 第 4 列文本（训练监督信号）

`workflows/gpt_sovits/tools/gpt_sovits_prepare_dataset.py` 会：
- 优先使用同名 `.list`
- 在 workflow 的 `data/<exp-name>/` 下生成 `_effective.list`（使用 basename 而非绝对路径）
- 生成 `_wav_input/`（把数据集音频 symlink/copy 成 list 里需要的文件名），从而即使 `.list` 里是 `/mnt/c/...` 绝对路径，也能走本地 ext4 数据集

