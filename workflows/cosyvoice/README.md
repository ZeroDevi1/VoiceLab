# CosyVoice workflow（xuan SFT）

本目录存放 **CosyVoice 的数据准备、训练配置、训练产物与脚本**，并且不把任何内容提交回上游 CosyVoice 仓库。

上游 CosyVoice 仓库放在 `../../vendor/CosyVoice`（整个 `vendor/` 目录被 git 忽略，需要时自行 clone/pull）。

使用说明见：`docs/cosyvoice_xuan_sft_wsl_ubuntu2404.md`

初始化环境（WSL 推荐 Python 3.10）：

```bash
cd ~/AntiGravityProjects/VoiceLab/workflows/cosyvoice
uv sync
```
