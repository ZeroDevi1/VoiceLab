# GPT-SoVITS

本目录提供 VoiceLab 风格的 GPT-SoVITS workflow（数据准备 + 训练入口），并与本仓库统一的 `.list` 标注格式打通。

文档入口：
- `docs/workflows/gpt_sovits/gpt_sovits_prepare_train_wsl_ubuntu2404.md`
- `.list` 统一规则：`docs/datasets/list_annotations.md`

上游仓库位于：

```bash
cd "${VOICELAB_DIR:-$PWD}"
uv run -m voicelab vendor sync
```
