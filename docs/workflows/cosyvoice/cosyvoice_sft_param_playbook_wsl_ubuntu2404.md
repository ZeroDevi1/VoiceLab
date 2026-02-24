# CosyVoice3 说话人 SFT 参数经验 Playbook（WSL2 Ubuntu 24.04）

记录日期：2026-02-22  
适用范围：VoiceLab `workflows/cosyvoice`（CosyVoice3 Speaker SFT：LLM + Flow）  
目标取向：**自然度 / 可懂度优先**，在小样本与工程不稳定条件下保持可复现的 A/B 对比。

> 说明：本文**只记录参数与工程策略**，不记录任何具体推理文本内容。

---

## 1) 核心结论（可直接照抄的默认参数）

### 1.1 推理基线（推荐先跑这个）

- **Flow**：优先 A/B 使用 **Base Flow**（清晰度与稳定性更强）
  - `--flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt`
- **LLM**：用你本次训练里“val_best / best epoch”的 LLM（或 `llm_avg.pt`）
  - `--llm_ckpt exp/<spk>_sft/llm/torch_ddp/epoch_<N>_whole.pt`
- **采样（甜点位 sampling06k10）**：更保守，降低韵律随机性/崩坏概率
  - `--temperature 1.0`
  - `--top_p 0.6`
  - `--top_k 10`
  - `--win_size 10`
  - `--tau_r 1.0`
- **复现**：固定随机种子（A/B 测试才有意义）
  - `--seed 1986`
- **速度（更口语/更“人味”）**：从稍慢开始做 A/B
  - 推荐从 `--speed 0.95` 起步
  - 可下探 `0.90` / `0.85`（越慢越可能更自然，但也更可能拖音，需要试听）
- **分段策略**：
  - `tools/infer_sft.py` **永远不做自动拆分**（始终单段推理）
  - 如需分段/分句：请你自己在外部把文本切成多份，分多次推理（每次一段），再自行后期拼接

### 1.2 工程稳定性（WSL 常见坑的固定处理）

- ORT CUDA provider 缺库/不稳定：**推理/训练都可先强制 CPU EP 兜底**
  - `export COSYVOICE_ORT_FORCE_CPU=1`
- 文本前端 wetext：
  - 默认开启（`--text_frontend`，脚本默认 True）
  - 若需要“完全离线/不下载/快速试跑”，可用 `--no-text_frontend`（但 TN 质量与鲁棒性可能下降）

---

## 2) 典型翻车现象与“参数化”修复思路

### 2.1 “chunk 爆炸 / 逐字符合成” -> split=False 返回值被误迭代（工程 bug）

根因（工程层面）：
- `frontend.text_normalize(..., split=False)` 在上游会返回 **string**
- 若把它当 iterable 去 `for seg in norm:` 会变成“逐字符合成”

工程修复要点：
- 当 `split=False` 时，必须把返回值包装成 `segments=[norm]`
- 非 stream 模式下应只产出一个文件：`chunk_0000.wav`（或用户用 `--out_wav` 指定的路径）

### 2.2 “电音念经 / 拖音吟唱 / 韵律崩坏” -> 小样本 TTS 的经典症状

高概率成因（策略层面）：
- **训练分布 vs 推理文本风格**不一致（短口语训练 vs 长文本推理）
- 小样本下 **Flow 更易过拟合/退化**（清晰度下降、金属电音、随机崩坏）
- LLM 采样随机性过强导致 duration/token 选择不稳定

参数化修复路径（按见效速度排序）：
1) 推理阶段先回退 **Base Flow**（保清晰度）
2) 采样收紧到 `top_p=0.6/top_k=10/tau_r=1.0`（减少“随机乱飘”）
3) 固定 `seed` 做 A/B（否则结论不可复现）
4) 合理分句（split=True）降低一次生成的注意力负担
5) 语速微调（0.95/0.90/0.85）提升“口语人味”

---

## 3) 推理脚本：统一参数入口 + run.json + 统一试听命名

### 3.1 `tools/infer_sft.py`（已概念化）

该脚本作为“通用 Speaker SFT 推理工具”使用，关键点：
- `--spk_id` 必须显式指定（不再默认某个具体人物）
- 运行时覆写 `cosyvoice.model.llm.sampling`，无需手改 `cosyvoice3.yaml`
- 写出 `run.json` 但**不落盘保存任何具体文本内容**（只保存 hash/长度/参数）

### 3.2 标准推理命令模板（只示例参数，不包含推理文本）

```bash
cd workflows/cosyvoice
export COSYVOICE_ORT_FORCE_CPU=1

uv run python tools/infer_sft.py \
  --model_dir <MODEL_DIR> \
  --spk_id <SPK_ID> \
  --text_file <TEXT_FILE> \
  --out_dir  <OUT_DIR> \
  --llm_ckpt  exp/<spk>_sft/llm/torch_ddp/epoch_<N>_whole.pt \
  --flow_ckpt pretrained_models/Fun-CosyVoice3-0.5B/flow.pt \
  --prompt_text "<SHORT_PROMPT_FROM_METADATA><|endofprompt|>" \
  --speed 0.95 \
  --seed 1986 \
  --temperature 1.0 \
  --top_p 0.6 --top_k 10 --win_size 10 --tau_r 1.0
```

### 3.3 统一试听文件夹（命名=参数）

收集同一轮实验产物到一个文件夹逐个试听对比：

```bash
python3 tools/collect_listen_wavs.py \
  --src <OUT_ROOT> \
  --dst <LISTEN_DIR> \
  --mode copy \
  --spk_id <SPK_ID> \
  --include_params
```

输出示例（文件名包含核心参数 tag）：
- `...__speed095__temp100__tp060__tk010__tr100__seed1986__nosplit__wetext1__chunk_0000.wav`

---

## 4) A/B 实验最小矩阵（避免无意义全排列）

固定项：
- Base Flow + Best LLM + sampling06k10 + seed 固定

只动 2 个变量就够定位问题：
1) `speed=0.95` / `0.90` / `0.85`
2) （可选）`prompt_text`/`prompt_strategy` 做情绪与泄露的平衡

验收标准（主观听感 + 工程输出）：
- 非 stream：只产出一个 `chunk_0000.wav`（或 `--out_wav` 指定路径）
- 不出现明显“电音/金属感/念经式拖音”
- 吐字可懂、断句基本自然
