# RVC：训练 xuan 模型（v2 + 48k + f0）+ 推理参数方案（WSL / Ubuntu 24.04）

本文档对应 `workflows/rvc/` 工作流：使用 **RVC v2 + 48k + f0(RMVPE)** 训练 `xuan` 声线模型，并训练 `.index` 检索索引，最后给出可复用的推理参数预设。

> 说明：本文默认你在 **WSL2 / Ubuntu 24.04**，并且能正常使用 NVIDIA GPU（`nvidia-smi` 可用）。

## 0. 数据集概览（已探测）

Windows 侧数据集：
- 源路径：`/mnt/c/AIGC/数据集/xuan`
- 文件数：`390` 条 `wav`
- 抽样属性：`pcm_s16le` / `44100 Hz` / `mono`
- 总时长：约 `32.12` 分钟（单条约 `2.04s ~ 11.26s`，均值约 `4.94s`）

> 你最初描述为“400 条”，但实际目录下探测到 `390` 个文件；建议后续以脚本统计结果为准。

## 1. 前置条件

- 你的 VoiceLab 工作区：`~/AntiGravityProjects/VoiceLab`
- 本工作区已存在上游仓库（在 `vendor/` 下）：
  - `vendor/Retrieval-based-Voice-Conversion-WebUI`（RVC 上游）
- 已安装：
  - `ffmpeg`
  - `uv`
- GPU 可用：`nvidia-smi` 能看到显卡

## 2. 初始化 RVC workflow Python 环境（uv）

RVC 依赖链对 Python 版本比较敏感，建议固定 **Python 3.10**（本 workflow 也在 `pyproject.toml` 里限制了 `>=3.10,<3.11`）。

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc

# 没有本地 python3.10 的情况下，用 uv 下载并 pin
uv python install 3.10
uv python pin 3.10

# 安装依赖
uv sync
```

如果下载/安装大包超时，可临时调大超时时间：

```bash
UV_HTTP_TIMEOUT=600 uv sync
```

## 3. 初始化 runtime（不污染 vendor）

RVC 上游脚本里大量使用相对路径（例如 `assets/hubert/hubert_base.pt`、`configs/inuse/*`、`./logs/...`）。
本工作流通过 `workflows/rvc/runtime/` 作为“运行根目录”来承载这些路径，从而：
- 不修改 `vendor/`（上游代码只做只读 symlink）
- 训练产物/索引全部落到 `workflows/rvc/runtime/`

执行：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
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

## 4.（强烈推荐）把数据集复制到 WSL 原生目录（避免 /mnt/c 跨系统 I/O 瓶颈）

WSL2 访问 Windows 文件系统（`/mnt/c/...`）会有明显的 9P I/O 开销。对于几百条小 wav，**预处理/特征提取阶段**容易被 I/O 拖慢。

推荐把数据集复制到 WSL 的 ext4（例如 VoiceLab 下的 `datasets/`）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_stage_dataset.py \
  --src "/mnt/c/AIGC/数据集/xuan" \
  --dst "$HOME/AntiGravityProjects/VoiceLab/datasets/xuan"
```

复制完成后，确认文件数一致：

```bash
find "$HOME/AntiGravityProjects/VoiceLab/datasets/xuan" -maxdepth 1 -type f -iname "*.wav" | wc -l
```

之后训练时直接指向 WSL 路径：

```bash
uv run python tools/rvc_train.py --dataset-dir "$HOME/AntiGravityProjects/VoiceLab/datasets/xuan"
```

## 5. 训练 xuan 模型（阶段 A：先跑通全链路 30 epoch）

统一实验名（建议固定，后续索引/推理都用它）：
- `xuan_v2_48k_f0`

执行：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_train.py \
  --dataset-dir "$HOME/AntiGravityProjects/VoiceLab/datasets/xuan" \
  --exp-name xuan_v2_48k_f0 \
  --total-epoch 30 \
  --batch-size 4 \
  --save-every-epoch 5 \
  --quiet-warnings
```

输出位置（重要）：
- 实验目录：`workflows/rvc/runtime/logs/xuan_v2_48k_f0/`
- 最终模型：`workflows/rvc/runtime/assets/weights/xuan_v2_48k_f0.pth`

显存不够（RTX3060 Laptop 6GB）时：
- 把 `--batch-size 4` 降到 `--batch-size 2`

## 6. 训练 xuan 模型（阶段 B：续跑到 200 epoch）

RVC 上游训练脚本会在 `runtime/logs/xuan_v2_48k_f0/` 内自动发现 `G_*.pth/D_*.pth` 并 resume。
第二阶段通常不需要重新预处理/提取特征/写 filelist；因此建议跳过这些步骤直接续跑训练。

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_train.py \
  --exp-name xuan_v2_48k_f0 \
  --total-epoch 200 \
  --batch-size 4 \
  --save-every-epoch 10 \
  --skip-preprocess --skip-f0 --skip-feature --skip-filelist \
  --quiet-warnings
```

## 7. 构建特征索引（.index）

该索引用于推理时的检索增强（`index_rate` 拉高通常能显著降低“电音感/撕裂”）。

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_train_index.py --exp-name xuan_v2_48k_f0
```

输出：
- `workflows/rvc/runtime/indices/added_IVF*_Flat_nprobe_1_xuan_v2_48k_f0_v2.index`
- 软链短名：`workflows/rvc/runtime/indices/xuan_v2_48k_f0.index`

## 8. 单文件推理（sanity check）

先用一段已存在的输入音频做 sanity check（示例用仓库里已有的 XingTong wav）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_infer_one.py \
  --exp-name xuan_v2_48k_f0 \
  --model latest \
  --input "$HOME/AntiGravityProjects/VoiceLab/datasets/XingTong/XingTong_445.wav" \
  --output "$HOME/AntiGravityProjects/VoiceLab/workflows/rvc/out_wav/xingtong_to_xuan_pitch0.wav" \
  --pitch 0 \
  --index-rate 0.8 \
  --filter-radius 3 \
  --rms-mix-rate 0.25 \
  --protect 0.33
```

### 8.x 训练中途停止时，如何用“最新权重”做推理

`tools/rvc_infer_one.py` 默认加载 `runtime/assets/weights/<exp-name>.pth`。
但如果你是“中途手动停止训练”，这个 `<exp-name>.pth` **可能还没生成**（因为它通常在训练结束时保存）。

推荐做法：训练时开启周期性导出“可推理权重”，然后推理时用 `--model latest` 自动挑最新的那份。

另外，本 workflow 的 `tools/rvc_train.py` 已做了增强：当你在训练时按 **Ctrl+C**，会在进程退出后自动把“最近一次保存的 epoch checkpoint”导出成 `runtime/assets/weights/<exp-name>.pth`（前提：你至少已经保存过一次 checkpoint）。

1) 训练时（每 10 epoch 保存一次 checkpoint，并同步导出 weights）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_train.py \
  --exp-name xuan_v2_48k_f0 \
  --total-epoch 200 \
  --save-every-epoch 10 \
  --save-every-weights 1 \
  --batch-size 4 \
  --skip-preprocess --skip-f0 --skip-feature --skip-filelist \
  --quiet-warnings
```

2) 推理时自动选择最新权重：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_infer_one.py \
  --exp-name xuan_v2_48k_f0 \
  --model latest \
  --input "/path/to/input.wav" \
  --output "/path/to/output.wav"
```

如果你想在训练停止后“手动”导出（不依赖自动导出），也可以执行：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_export_latest_weights.py --exp-name xuan_v2_48k_f0
```

## 8.1 使用一首音乐人声做测试（台风 - 蒋蒋 vocals）

测试音频（男声人声干声，约 164.55s）：
- `/mnt/c/AIGC/音乐/台风/台风 - 蒋蒋_vocals_karaoke_noreverb_dry.wav`

### 方案 A：直接用 `rvc_infer_one.py`（最通用）

> 注意：`--pitch` 会改变整首歌的音高（移调）。唱歌场景一般建议先从 `0` 开始，先保证“不移调”，再根据音色/撕裂情况做微调。

> 立体声说明：这首 vocals 是 **stereo**。建议加 `--stereo-mode pan`，用“时间变化的增益”复用声像，
> 可以尽量保留左右声道的 panning，同时避免因相位/反相导致的“空心/低频被抵消”问题。
> `--stereo-mode dual`（左右各跑一次）更慢，也更容易出现左右细节不一致导致的宽度/相位怪感。

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_infer_one.py \
  --exp-name xuan_v2_48k_f0 \
  --model latest \
  --input "/mnt/c/AIGC/音乐/台风/台风 - 蒋蒋_vocals_karaoke_noreverb_dry.wav" \
  --output "$HOME/AntiGravityProjects/VoiceLab/workflows/rvc/out_wav/taifeng_jj_to_xuan_pitch0.wav" \
  --pitch 0 \
  --index-rate 0.65 \
  --filter-radius 3 \
  --rms-mix-rate 0.25 \
  --protect 0.4 \
  --stereo-mode pan
```

### 方案 B：使用本 workflow 的“固定测试歌”脚本（推荐省事）

该脚本内置了输入路径与一套更偏“干净/少撕裂”的 preset，并默认 `pitch=0`（不移调）：
- `tools/rvc_infer_taifeng_to_xuan.py`

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_infer_taifeng_to_xuan.py --model latest
```

如果你想用同一套 preset，但指定别的输出/或试不同 pitch：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_infer_taifeng_to_xuan.py \
  --pitch 0 \
  --output "$HOME/AntiGravityProjects/VoiceLab/workflows/rvc/out_wav/taifeng_jj_to_xuan_pitch0_try2.wav"
```

## 9. 推理参数预设（三档）

### Preset-Speech（默认）
- `pitch=0`
- `index-rate=0.8`
- `filter-radius=3`
- `rms-mix-rate=0.25`
- `protect=0.33`

### Preset-MoreTarget（更像目标音色）
- `index-rate=0.9`
- 其余同 Preset-Speech

### Preset-MoreClean（更干净、撕裂风险更低）
- `index-rate=0.65`
- `protect=0.4`
- 其余同 Preset-Speech

### pitch 调整规则（可执行建议）
- 需要更“女声化”：`+6 ~ +12`
- 需要更“男声化”：`-3 ~ -6`
- 每次只改 `2~3` 半音做 A/B 对比，不要一次跳太大

## 10. 批量推理（模板）

示例：把一个目录下的 wav 批量转换到输出目录（逐文件指定 `--output`，避免相对路径混乱）。

```bash
IN_DIR="$HOME/in_wav"
OUT_DIR="$HOME/out_wav_xuan"
mkdir -p "$OUT_DIR"

cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
for f in "$IN_DIR"/*.wav; do
  bn="$(basename "$f" .wav)"
  uv run python tools/rvc_infer_one.py \
    --exp-name xuan_v2_48k_f0 \
    --input "$f" \
    --output "$OUT_DIR/${bn}_to_xuan.wav" \
    --pitch 0 \
    --index-rate 0.8 \
    --filter-radius 3 \
    --rms-mix-rate 0.25 \
    --protect 0.33
done
```

## 11. TensorBoard（可视化训练曲线）

RVC 上游训练脚本会在实验目录下写 TensorBoard event 文件：
- `workflows/rvc/runtime/logs/<exp-name>/events.out.tfevents.*`

你可以用 TensorBoard 观察训练过程（loss 曲线、梯度范数、以及 mel 频谱图等），用于：
- 判断是否欠拟合/过拟合（loss 是否持续下降、是否早早平台期）
- 对比不同 epoch 的听感（结合 `--save-every-weights 1` 或在固定 epoch 处推理 A/B）
- 发现异常（loss 爆炸、梯度异常、训练停滞等）

### 11.1 启动（推荐：只看一个实验）

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run tensorboard --logdir=runtime/logs/xuan_v2_48k_f0 --host 0.0.0.0 --port 6006
```

浏览器打开（WSL2 常见访问方式）：
- Windows 本机浏览器通常可直接用：`http://127.0.0.1:6006`（WSL2 端口转发）
- 如果你在局域网其它设备访问，或本机 `127.0.0.1` 不通：
  1) 获取 WSL IP：
     ```bash
     hostname -I | awk '{print $1}'
     ```
  2) 用：`http://<WSL_IP>:6006`

### 11.2 启动（看多个实验/对比）

把 `--logdir` 指向 `runtime/logs`，TensorBoard 会按子目录分 run 展示（适合对比多个实验名）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run tensorboard --logdir=runtime/logs --host 0.0.0.0 --port 6006
```

### 11.3 推荐重点关注的面板/指标

- Scalars
  - `loss/g/total`：生成器总损失（整体趋势看“是否还在学”）
  - `loss/d/total`：判别器总损失（是否过强/过弱）
  - `grad_norm_g` / `grad_norm_d`：梯度范数（异常飙升/归零都值得排查）
- Images
  - `slice/mel_org` vs `slice/mel_gen`：生成 mel 是否逐渐贴近真实 mel（仅作趋势参考）

> 备注：RVC 的 loss 数值本身不容易“跨实验绝对比较”，更建议看同一套配置下的趋势变化 + 最终听感。
