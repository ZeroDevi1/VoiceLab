# RVC：训练星瞳模型 + 迁移炫神音频（WSL / Ubuntu 24.04）

本文档对应 `workflows/rvc/` 工作流：**用 1500 条星瞳数据训练 RVC v2 + 48k + f0，并生成 .index，然后把指定炫神 mp3 迁移成星瞳声线**。

## 0. 前置条件
- WSL2 + NVIDIA GPU 可用（`nvidia-smi` 能看到显卡）
- `ffmpeg` 已安装（本机已是 `ffmpeg 6.x`）
- 本工作区已存在上游仓库：`vendor/Retrieval-based-Voice-Conversion-WebUI`

> 说明：RVC 依赖链对 Python 版本比较敏感。这里固定使用 **Python 3.10**（`requires-python = ">=3.10,<3.11"`）。

## 0.1 路径约定（避免强依赖本机固定目录）

本文档不要求你把仓库放在固定路径（例如 `~/AntiGravityProjects/VoiceLab`）。只要你在仓库根目录执行过：

```bash
export VOICELAB_DIR="$PWD"
```

如果你不确定当前是否在仓库根目录，可以用：

```bash
export VOICELAB_DIR="$(git rev-parse --show-toplevel)"
```

## 1. 初始化 Python 环境（uv）

```bash
cd "$VOICELAB_DIR/workflows/rvc"

# 没有本地 python3.10 的情况下，用 uv 下载并 pin
uv python install 3.10
uv python pin 3.10

# 安装依赖（第一次会下载 torch GPU wheels，时间较长）
uv sync
```

如果下载 `nvidia-*` 大包时出现网络超时（例如 `UV_HTTP_TIMEOUT` 默认 30s 不够），可以加大超时时间重试：

```bash
UV_HTTP_TIMEOUT=600 uv sync
```

如果出现“解压失败/IO error”且反复卡在同一个包，通常是缓存里有坏包，清掉对应缓存后再重试：

```bash
uv cache clean --force nvidia-cublas-cu12
UV_HTTP_TIMEOUT=600 uv sync
```

如果 `uv sync` 在安装 `fairseq` 等包时报编译错误，通常需要系统依赖（Ubuntu）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
```

## 2. 初始化 runtime（不污染 vendor）

RVC 上游脚本里大量使用相对路径（例如 `assets/hubert/hubert_base.pt`、`configs/inuse/*`、`./logs/...`）。
本工作流通过 `workflows/rvc/runtime/` 作为“运行根目录”来承载这些路径，从而：
- 不修改 `vendor/`（上游代码只做只读 symlink）
- 训练产物/索引全部落到 `workflows/rvc/runtime/`

执行：

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_init_runtime.py
```

默认会从你已有的 Windows 目录链接大模型资源：
- `/mnt/c/AIGC/RVC20240604Nvidia/assets/hubert/hubert_base.pt`
- `/mnt/c/AIGC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt`
- `/mnt/c/AIGC/RVC20240604Nvidia/assets/pretrained_v2/*`

如需改资源位置：

```bash
uv run python tools/rvc_init_runtime.py --assets-src /path/to/assets
```

## 2.1（强烈推荐）把数据集复制到 WSL 原生目录（避免 /mnt/c 跨系统 I/O 瓶颈）

WSL2 访问 Windows 文件系统（`/mnt/c/...`）会有明显的 9P I/O 开销。对于 1500 条小 wav，**预处理阶段**会被拖慢，GPU 也可能等数据导致利用率不满。

推荐把数据集复制到 WSL 的 ext4（例如 VoiceLab 下的 `datasets/`）：

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_stage_dataset.py \
  --src /mnt/c/AIGC/数据集/XingTong \
  --dst "$VOICELAB_DIR/datasets/XingTong"
```

> `.list` 标注：stage 时会把源目录内的同名 `*.list` 一并复制；如果源目录没有同名 `.list`，会尝试从 `/mnt/c/AIGC/数据集/标注文件/<同名>.list` 补拷贝到目标目录。
> 统一 `.list` 格式与规则见：`docs/datasets/list_annotations.md`

之后训练时直接指向 WSL 路径即可：

```bash
uv run python tools/rvc_train.py --dataset-dir "$VOICELAB_DIR/datasets/XingTong"
```

也可以让训练脚本自动复制（只影响 preprocess 阶段；后续训练 I/O 都在 `workflows/rvc/runtime/logs`）：

```bash
uv run python tools/rvc_train.py --stage-dataset
```

## 3. 训练星瞳模型（阶段 1：先跑通全链路 50 epoch）

数据集：
- `$VOICELAB_DIR/datasets/XingTong`（约 1500 条 wav，WSL 原生路径，推荐）

`.list` 行为（重要）：
- 如果数据集目录存在同名 `.list`（例如 `XingTong/XingTong.list`），训练时会**只预处理 `.list` 中列出的音频**（音频白名单）。
- RVC 不使用文本标注；`.list` 的 `speaker/lang/text` 对 RVC 仅作备注。

执行：

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_train.py \
  --dataset-dir "$VOICELAB_DIR/datasets/XingTong" \
  --exp-name xingtong_v2_48k_f0 \
  --total-epoch 50 \
  --batch-size 4 \
  --save-every-epoch 3
```

输出位置（重要）：
- 实验目录：`workflows/rvc/runtime/logs/xingtong_v2_48k_f0/`
- 最终模型：`workflows/rvc/runtime/assets/weights/xingtong_v2_48k_f0.pth`

### Checkpoint 保存频率（按“时间”折算）

`--save-every-epoch` 是“按 epoch 保存一次 checkpoint”。你当前日志显示：
- Epoch 1 用时约 **12 分 47 秒**
- Epoch 2 用时约 **10 分 18 秒**

按这个速度估算，每个 epoch ~10-13 分钟，则：
- `--save-every-epoch 3` 约 **30-40 分钟**保存一次（推荐：防断电/中断，容灾更稳）
- `--save-every-epoch 4` 约 **40-55 分钟**保存一次（更省磁盘，但风险更高）

> 注意：epoch 用时会随 batch size / CPU 负载变化。如果你用 `--batch-size 2`，epoch 通常会更慢一点，可把保存间隔从 3 调到 2~3。

显存不够（RTX3060 Laptop 6GB）时：
- `--batch-size 4` 降到 `--batch-size 2`

### 训练日志里大量 FutureWarning / UserWarning

例如：
- `torch.cuda.amp.autocast(args...) is deprecated ...`
- `torch.load(weights_only=False) ...`

这些通常只是 PyTorch 版本升级导致的弃用提示，不影响训练。如果你想让日志更干净，可以在训练命令加上：

```bash
uv run python tools/rvc_train.py --quiet-warnings ...
```

## 4. 训练星瞳模型（阶段 2：继续到 200 epoch）

RVC 上游训练脚本会在 `runtime/logs/xingtong_v2_48k_f0/` 内自动发现 `G_*.pth/D_*.pth` 并 resume。

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_train.py \
  --dataset-dir "$VOICELAB_DIR/datasets/XingTong" \
  --exp-name xingtong_v2_48k_f0 \
  --total-epoch 200 \
  --batch-size 4 \
  --save-every-epoch 3 \
  --skip-preprocess --skip-f0 --skip-feature --skip-filelist
```

> 说明：第二阶段通常不需要重新预处理/提取特征；因此建议加 `--skip-*` 直接训练续跑。

## 5. 构建特征索引（.index）

该索引用于推理时的检索增强（index_rate 拉高能显著降低“电音感/撕裂”）。

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_train_index.py --exp-name xingtong_v2_48k_f0
```

输出：
- `workflows/rvc/runtime/indices/added_IVF*_Flat_nprobe_1_xingtong_v2_48k_f0_v2.index`
- 软链短名：`workflows/rvc/runtime/indices/xingtong_v2_48k_f0.index`

## 6. 推理（普通人声 + 歌声：根据输入选择不同预设）

推理建议：
- 普通人声（Speech）：优先用 **Preset-Speech**（更像目标音色可切 **Preset-MoreTarget**）
- 歌声（Singing）：优先用 **Preset-MoreClean**（更干净、撕裂风险更低；必要时再调 `pitch`）

> 建议：如果你的输入音频是 `workflows/msst` 产出的 `*_vocals_karaoke_noreverb_dry.wav`，通常推理效果会更稳定。
>
> 设备提示：如需强制用 GPU，可在命令里加 `--device cuda:0`（或其它编号）；强制 CPU 则用 `--device cpu`。

### 6.1 推理参数预设（三档）

Preset-Speech（默认）
- `index-rate=0.8`
- 其余保持默认（例如：`filter-radius=3, rms-mix-rate=0.25, protect=0.33`）

Preset-MoreTarget（更像目标音色）
- `index-rate=0.9`
- 其余同 Preset-Speech

Preset-MoreClean（更干净、撕裂风险更低）
- `index-rate=0.65`
- `protect=0.4`
- 其余同 Preset-Speech（例如：`filter-radius=3, rms-mix-rate=0.25`）

pitch 调整规则（可执行建议）
- 男 -> 女：优先从 `+10 ~ +15` 试（常用 `+12`）
- 女 -> 男：优先从 `-3 ~ -6` 试
- 每次只改 `2~3` 半音做 A/B 对比，不要一次跳太大

### 6.2 普通人声（Speech）示例：炫神 -> 星瞳（Preset-Speech）

待转换音频：
- `/mnt/c/AIGC/炫神/马头有大！马头来了！.mp3`

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_infer_one.py \
  --exp-name xingtong_v2_48k_f0 \
  --model latest \
  --input "/mnt/c/AIGC/炫神/马头有大！马头来了！.mp3" \
  --output "$VOICELAB_DIR/workflows/rvc/out_wav/马头有大_to_xingtong_pitch12_preset_speech.wav" \
  --pitch 12 \
  --f0-method rmvpe \
  --index-rate 0.8
```

### 6.3 歌声（Singing）示例：栞 -> 星瞳（Preset-MoreClean）

示例输入（通常由 `workflows/msst` 产出）：
- `/mnt/c/AIGC/音乐/栞/栞 - MyGO!!!!!_vocals_karaoke_noreverb_dry.wav`

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_infer_one.py \
  --exp-name xingtong_v2_48k_f0 \
  --model latest \
  --input '/mnt/c/AIGC/音乐/栞/栞 - MyGO!!!!!_vocals_karaoke_noreverb_dry.wav' \
  --output "$VOICELAB_DIR/workflows/rvc/out_wav/shiori_mygo_to_xingtong_pitch0_preset_moreclean.wav" \
  --pitch 0 \
  --f0-method crepe \
  --index-rate 0.65 \
  --protect 0.4 \
  --stereo-mode pan
```

> 提示：
> - `--f0-method crepe` 更稳但更慢；想快可换成 `rmvpe`。
> - 歌声场景一般先从 `--pitch 0` 开始（不移调）；如果你明确需要“更女声化”，再小步加 `pitch`（例如 `+6/+9/+12`）。
> - 上面示例里未出现的参数，会使用 `tools/rvc_infer_one.py` 的默认值；需要时见 §7（全参数说明）。

> 说明：推理脚本默认会对输出做峰值归一化（避免“几乎没声音”或“爆音电流声”）。如需关闭可加 `--no-normalize`。

## 7. 推理参数全量说明（可选）

本节是 `tools/rvc_infer_one.py` 的参数速查（给“需要深度调参/排查问题”时用）。一般日常推理只用 §6 的少量参数即可。

### 输入/输出/模型选择

- `--exp-name`：实验名（决定默认权重 `<exp-name>.pth` 与索引 `<exp-name>.index`）
- `--model`：权重选择
  - `latest`：自动选择 `runtime/assets/weights/` 下最近修改的、以 `exp-name` 为前缀的权重
  - 具体文件名：例如 `xingtong_v2_48k_f0_e30_s123456.pth`
  - 绝对路径：例如 `/path/to/xxx.pth`
- `--input`：输入音频路径
- `--output`：输出音频路径

### “影响听感”的常用参数（最常改）

- `--pitch`：变调（半音；唱歌一般先从 `0` 开始）
- `--f0-method`：`rmvpe|fcpe|crepe|harvest|pm`
  - `rmvpe`：速度/质量平衡（默认）
  - `crepe`：更稳但更慢（唱歌常用）
- `--index-rate`：索引检索强度（更像目标音色通常会更高；太高可能带来“电音感/咬字撕裂”）
- `--protect`：保护清辅音/呼吸，降低撕裂（更干净通常把它调高）
- `--rms-mix-rate`：响度包络融合（默认 `0.25` 通常足够）
- `--filter-radius`：平滑（默认 `3`）
- `--resample-sr`：重采样输出采样率（默认 `0` 表示不额外重采样）

### 立体声处理（当输入是 stereo 时）

- `--stereo-mode`：`mono|pan|dual`
  - `mono`：下混到单声道（默认，最快）
  - `pan`：先按 mono 推理，再把输入的声像“作为增益包络”复用到输出（更适合人声）
  - `dual`：左右声道分别推理（最慢，且更容易出现左右不一致）

`--stereo-mode pan` 的高级参数（一般不用改）：
- `--pan-window-ms`（默认 `50`）
- `--pan-hop-ms`（默认 `10`）
- `--pan-strength`（默认 `1.0`）

### 输出与其它

- `--subtype`：WAV subtype（默认 `PCM_16`；一般不用改）
- `--no-normalize`：关闭峰值归一化（默认会归一化，避免“太小声/爆音”）
- `--device`：强制设备（例如 `cuda:0` / `cpu`；不传则使用上游 Config 默认）
- `--is-half`：是否启用 FP16（默认开启）

全参数命令模板（仅供参考；把你不需要的参数删掉即可）：

```bash
cd "$VOICELAB_DIR/workflows/rvc"
uv run python tools/rvc_infer_one.py \
  --exp-name xingtong_v2_48k_f0 \
  --model latest \
  --input "/path/to/input.wav" \
  --output "/path/to/output.wav" \
  --device cuda:0 \
  --pitch 12 \
  --f0-method rmvpe \
  --index-rate 0.8 \
  --filter-radius 3 \
  --resample-sr 0 \
  --rms-mix-rate 0.25 \
  --protect 0.33 \
  --stereo-mode pan \
  --pan-window-ms 50 \
  --pan-hop-ms 10 \
  --pan-strength 1.0 \
  --subtype PCM_16
```

## 8. 常见问题排查
- 找不到 `hubert_base.pt` / `rmvpe.pt`
  - 先确认 `tools/rvc_init_runtime.py` 运行成功
  - 或用 `--assets-src` 指向你真实的 RVC 资产目录
- 显存不足 / CUDA OOM
  - 训练：降 `--batch-size`
  - 提取：确保只用单卡（默认就是）
- index 构建内存不足
  - `tools/rvc_train_index.py` 默认是“流式/省内存”实现
  - 仍不够：调低 `--max-train-frames`（例如 100000）
- `ModuleNotFoundError: No module named 'pkg_resources'`（librosa 导入时报错）
  - 原因：较新的 `setuptools` 可能不再包含 `pkg_resources`
  - 解决：本 workflow 已在 `workflows/rvc/pyproject.toml` 固定 `setuptools==69.5.1`；执行 `uv sync` 让其生效即可

## 9. TensorBoard（可视化训练曲线）

在 `workflows/rvc/` 下运行（使用 uv 环境）：

```bash
cd "$VOICELAB_DIR/workflows/rvc"

# 使用 uv 环境运行 tensorboard，指向你的日志目录
uv run tensorboard --logdir=runtime/logs/xingtong_v2_48k_f0 --host 0.0.0.0 --port 6006
```

然后在浏览器打开：
- 本机：`http://127.0.0.1:6006`
- 局域网其它设备：`http://<你的WSL主机IP>:6006`
