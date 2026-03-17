# VoiceLab 新系统迁移与环境初始化实施计划

本文档面向 VoiceLab 仓库的维护者，目标是把当前 workflow 中对本机环境、路径、模型缓存和上游参考代码的隐式依赖，收敛为一套可复用、可验证、可在新机器上快速落地的初始化方案。

## 1) 目标

- 让新机器以 `uv run -m voicelab bootstrap` 作为唯一推荐入口。
- 把当前分散在脚本、文档、默认参数中的本机强依赖项显式化。
- 统一 `vendor/`、workflow 虚拟环境、模型资产缓存、runtime 初始化的约定。
- 为后续补齐 `gpt_sovits` 的一键初始化预留统一框架。

## 2) 当前范围

本计划覆盖以下 workflow：

- `cosyvoice`
- `rvc`
- `msst`
- `gpt_sovits`（当前仅做最小可用接入计划）

本计划优先处理“初始化层”和“依赖声明层”，不直接改动训练算法与推理逻辑。

## 3) 当前问题归类

### 3.1 已经具备的基础能力

- 仓库已提供 `voicelab bootstrap`，可统一执行 vendor 同步、`uv sync`、模型下载与 runtime 初始化。
- `vendor/` 与 `workflows/` 已分层，避免直接污染上游仓库。
- 多数 workflow 已支持通过环境变量覆盖 vendor 路径、assets 路径与 runtime 路径。

### 3.2 仍然存在的本机强依赖

- 部分脚本默认路径仍带有明显的 WSL/本机历史目录约定，例如 `/mnt/c/AIGC/...`。
- 数据集标注目录默认值仍偏向当前个人目录结构，而不是仓库内统一约定。
- `uv.toml` 中的 CUDA/PyTorch index 对新机器驱动与网络环境有要求。
- `gpt_sovits` 尚未完整纳入统一 bootstrap。
- 新环境缺少系统依赖时，目前还没有统一的只读自检入口。

### 3.3 风险点

- 新机器上 `git`、`uv`、`ffmpeg`、`rsync`、GPU 驱动不齐时，报错点分散。
- Windows / WSL / Linux 对 symlink 的权限和行为不同，runtime 初始化可能出现兼容性问题。
- 旧机器资产目录可复用，但当前复用方式分散在各 workflow 说明里，不利于迁移。

## 4) 实施原则

- 默认值优先使用“仓库相对路径”或“共享缓存路径”，不使用个人机器历史目录。
- 本机专用路径只允许出现在文档示例、迁移案例和显式参数中，不作为业务默认值。
- CLI 参数优先于环境变量，环境变量优先于仓库默认值。
- 保持 `vendor/` 为只读上游代码区，所有可写产物进入 `workflows/*/runtime/`、`workflows/*/data/` 或共享缓存。
- 新环境推荐流程必须可通过一份文档和一条主命令完成。

## 5) 分阶段实施

### 阶段 A：依赖盘点与分级

目标：形成统一的迁移依赖视图。

任务：

- 盘点系统依赖：`git`、`uv`、`ffmpeg`、`rsync`、`nvidia-smi`。
- 盘点参考代码依赖：`vendor/CosyVoice`、`vendor/RVC`、`vendor/MSST-WebUI`、`vendor/GPT-SoVITS`。
- 盘点模型资产依赖：
  - CosyVoice3 pretrained
  - RVC 的 `hubert_base.pt`、`rmvpe.pt`、`pretrained_v2/*`
  - MSST 的 `pretrain/*`
  - GPT-SoVITS 的 pretrained 目录与训练前置模型
- 盘点路径依赖：vendor 路径、assets 路径、annotation 路径、stage 目录、runtime 目录。
- 为每项依赖打标：`强依赖 / 弱依赖 / 可选依赖`。

交付物：

- 一份迁移依赖矩阵
- 一份本机强绑定项清单

### 阶段 B：收敛硬编码路径

目标：移除初始化流程对个人机器目录的默认依赖。

任务：

- 清理代码默认参数中的 `/mnt/c/AIGC/...`、旧 Windows 盘符路径等历史值。
- 统一路径优先级：
  1. CLI 参数
  2. 环境变量
  3. 仓库默认值或共享缓存
- 将 annotation 目录统一为可配置项，并为新环境提供 repo-relative 或显式传参方案。
- 保留示例路径，但仅放在文档中，不能作为脚本默认行为。

验收标准：

- 在一个全新目录 clone 后，不依赖个人历史目录即可完成 bootstrap。
- 不设置任何个人路径时，脚本不会默认指向不存在的旧机器资产目录。

### 阶段 C：补强统一 bootstrap 入口

目标：让 `voicelab bootstrap` 成为真正的一站式入口。

任务：

- 明确 bootstrap 的阶段输出：
  - vendor sync
  - env sync
  - assets download / reuse
  - runtime init
- 补齐 `gpt_sovits` 的最小接入：
  - vendor 同步
  - workflow 环境初始化入口
  - pretrained 路径约定
  - 无法自动化的部分给出明确提示
- 统一 `cn` / `global` profile 的说明与行为。
- 继续保持共享缓存根目录可通过 `VOICELAB_ASSETS_DIR` 覆盖。

验收标准：

- `cosyvoice`、`rvc`、`msst` 能通过同一入口完成初始化。
- `gpt_sovits` 至少能通过同一入口完成“拉代码 + 准备环境入口”的最小接入。

### 阶段 D：增加环境自检入口

目标：在真正执行 bootstrap 前后，都能快速判断环境状态。

任务：

- 新增 `voicelab doctor` 或等价只读检查命令。
- 检查项建议包括：
  - `git` / `uv` / `ffmpeg` / `rsync`
  - Python 3.10 可用性
  - GPU 与 `nvidia-smi`
  - `vendor/*` 是否齐全
  - workflow `uv sync` 是否完成
  - 共享 assets 缓存是否存在
  - runtime 关键文件是否完整
- 每个检查项输出 `OK / WARN / FAIL`，并附修复建议。

验收标准：

- 用户在新环境只需执行 doctor，即可看出缺的是什么和下一步该做什么。

### 阶段 E：统一资产缓存与复用方案

目标：兼顾“纯新环境下载”和“复用旧机器缓存”两种迁移方式。

任务：

- 明确共享资产根目录约定：默认使用 `VOICELAB_ASSETS_DIR`。
- 统一说明 workflow 级覆盖变量的用途与优先级。
- 明确哪些资产支持自动下载，哪些资产需要手动准备。
- 为“复用旧机器缓存 / NAS / 外接盘”提供单独操作指引。

验收标准：

- 用户能在不改代码的情况下切换“重新下载”与“复用旧缓存”两种初始化模式。

### 阶段 F：文档重构

目标：把“新环境初始化”收敛成一条主线，而不是分散在多个 workflow 文档里。

任务：

- 在 `docs/` 下保留一份面向新机器的总入口文档。
- 统一推荐流程：
  1. 安装系统依赖
  2. clone 仓库
  3. 执行 `voicelab bootstrap`
  4. 执行 `voicelab doctor`
  5. 再进入各 workflow 文档
- 将旧机器历史路径、个性化目录结构降级为“迁移案例”或“附录”。
- 在 `docs/index.md` 中显式暴露迁移与初始化文档入口。

验收标准：

- 新用户不需要先读多个 workflow 文档，也能完成首次环境搭建。

### 阶段 G：验证与回归

目标：确保方案不仅在当前机器成立，也能在目标环境复现。

任务：

- 至少验证以下场景：
  - 全新 WSL + 无旧资产
  - 全新 WSL + 复用旧资产目录
  - 非 WSL 环境下的最小可用性
- 验证维度：
  - bootstrap 是否能跑通
  - doctor 是否能准确定位缺失项
  - runtime 初始化是否完整
  - 关键 workflow 是否能进入训练/推理前置状态
- 记录验证结果与已知限制。

验收标准：

- 文档中存在已验证环境矩阵与已知问题清单。

## 6) 优先级建议

### P0（优先落地）

- 清理初始化相关代码中的本机硬编码默认路径
- 设计并实现 `voicelab doctor`
- 补齐 `bootstrap` 对 `gpt_sovits` 的最小支持
- 重写新环境初始化主线文档

### P1（增强体验）

- 统一 assets/cache/env 的命名与优先级说明
- 为缓存复用、离线迁移补充独立说明
- 补充 bootstrap 过程中的阶段性输出与错误提示

### P2（兼容与优化）

- 增强 Windows 非 WSL 场景兼容性
- 优化 symlink/copy 策略与权限提示
- 进一步收敛各 workflow 的个性化初始化差异

## 7) 建议执行顺序

推荐按以下顺序推进：

1. 先产出依赖矩阵与强绑定项清单
2. 再改初始化路径解析与默认值策略
3. 再补强 `bootstrap` 与 `doctor`
4. 再处理 `gpt_sovits` 的最小接入
5. 最后统一文档并做多环境验证

## 8) 预期成果

完成本计划后，仓库应达到以下状态：

- 新机器的推荐初始化命令收敛为一条主命令
- 本机专用路径不再作为默认行为出现在核心初始化脚本中
- `vendor`、assets、runtime 的职责边界更清晰
- 新用户能先看总文档，再按需进入具体 workflow 文档
- 迁移旧资产与全新下载两种路径都具备明确操作说明

## 9) 不在本轮范围内

以下事项不作为本轮实施计划的硬目标：

- 改写上游 vendor 仓库内部实现
- 重构训练算法或推理算法本身
- 为所有 workflow 提供完全一致的训练命令抽象
- 引入容器化/镜像化方案替代当前 `uv + vendor` 模式
