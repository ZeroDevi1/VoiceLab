# RVC：训练星瞳模型 + 迁移炫神音频（WSL / Ubuntu 24.04）

本文档对应 `workflows/rvc/` 工作流：**用 1500 条星瞳数据训练 RVC v2 + 48k + f0，并生成 .index，然后把指定炫神 mp3 迁移成星瞳声线**。

## 0. 前置条件
- WSL2 + NVIDIA GPU 可用（`nvidia-smi` 能看到显卡）
- `ffmpeg` 已安装（本机已是 `ffmpeg 6.x`）
- 本工作区已存在上游仓库：`vendor/Retrieval-based-Voice-Conversion-WebUI`

> 说明：RVC 依赖链对 Python 版本比较敏感。这里固定使用 **Python 3.10**（`requires-python = ">=3.10,<3.11"`）。

## 1. 初始化 Python 环境（uv）

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc

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

## 2.1（强烈推荐）把数据集复制到 WSL 原生目录（避免 /mnt/c 跨系统 I/O 瓶颈）

WSL2 访问 Windows 文件系统（`/mnt/c/...`）会有明显的 9P I/O 开销。对于 1500 条小 wav，**预处理阶段**会被拖慢，GPU 也可能等数据导致利用率不满。

推荐把数据集复制到 WSL 的 ext4（例如 VoiceLab 下的 `datasets/`）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_stage_dataset.py \
  --src /mnt/c/AIGC/数据集/XingTong \
  --dst ~/AntiGravityProjects/VoiceLab/datasets/XingTong
```

之后训练时直接指向 WSL 路径即可：

```bash
uv run python tools/rvc_train.py --dataset-dir ~/AntiGravityProjects/VoiceLab/datasets/XingTong
```

也可以让训练脚本自动复制（只影响 preprocess 阶段；后续训练 I/O 都在 `workflows/rvc/runtime/logs`）：

```bash
uv run python tools/rvc_train.py --stage-dataset
```

## 3. 训练星瞳模型（阶段 1：先跑通全链路 50 epoch）

数据集：
- `~/AntiGravityProjects/VoiceLab/datasets/XingTong`（约 1500 条 wav，WSL 原生路径，推荐）

执行：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_train.py \
  --dataset-dir ~/AntiGravityProjects/VoiceLab/datasets/XingTong \
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
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_train.py \
  --dataset-dir ~/AntiGravityProjects/VoiceLab/datasets/XingTong \
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
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_train_index.py --exp-name xingtong_v2_48k_f0
```

输出：
- `workflows/rvc/runtime/indices/added_IVF*_Flat_nprobe_1_xingtong_v2_48k_f0_v2.index`
- 软链短名：`workflows/rvc/runtime/indices/xingtong_v2_48k_f0.index`

## 6. 单文件推理：炫神 -> 星瞳

待转换音频：
- `/mnt/c/AIGC/炫神/马头有大！马头来了！.mp3`

执行（默认 RMVPE，默认 pitch=+12，默认 index_rate=0.8）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc
uv run python tools/rvc_infer_one.py \
  --exp-name xingtong_v2_48k_f0 \
  --pitch 12 \
  --index-rate 0.8
```

输出：
- `workflows/rvc/out_wav/马头有大_xingtong_pitch12.wav`

> 提示：推理脚本默认会对输出做峰值归一化（避免“几乎没声音”或“爆音电流声”）。如需关闭可加 `--no-normalize`。

如果你在训练中开启了 `--save-every-weights 1`，会在 `runtime/assets/weights/` 下生成按 epoch/step 命名的权重文件，
可用 `--model` 指定某个权重来做盲测对比：

```bash
uv run python tools/rvc_infer_one.py \
  --exp-name xingtong_v2_48k_f0 \
  --model xingtong_v2_48k_f0_e30_s123456.pth \
  --pitch 12 \
  --index-rate 0.8
```

## 7. 参数调优建议（只讲关键）
- `--pitch`（变调，半音）
  - 男 -> 女：优先在 `+10 ~ +15` 范围内试
  - 起点：`+12`
  - “花栗鼠”感强：降到 `+9 ~ +11`
  - “还是像男人捏嗓子”：升到 `+13 ~ +14`
- `--index-rate`
  - 星瞳数据足够纯净时：`0.7 ~ 0.9`
  - 想更像星瞳：提高
  - 想更保留源音细节/减少“过拟合口感”：降低
- `--protect`（保护清辅音/呼吸，防撕裂）
  - 默认 `0.33`；电音撕裂明显可略降（更强保护），但可能削弱索引效果

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
cd ~/AntiGravityProjects/VoiceLab/workflows/rvc

# 使用 uv 环境运行 tensorboard，指向你的日志目录
uv run tensorboard --logdir=runtime/logs/xingtong_v2_48k_f0 --host 0.0.0.0 --port 6006
```

然后在浏览器打开：
- 本机：`http://127.0.0.1:6006`
- 局域网其它设备：`http://<你的WSL主机IP>:6006`
