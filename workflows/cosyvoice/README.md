# CosyVoice workflow（xuan SFT）

本目录存放 **CosyVoice 的数据准备、训练配置、训练产物与脚本**，并且不把任何内容提交回上游 CosyVoice 仓库。

上游 CosyVoice 仓库放在 `../../vendor/CosyVoice`（整个 `vendor/` 目录被 git 忽略，需要时自行 clone/pull）。

使用说明见：`docs/workflows/cosyvoice/cosyvoice_xuan_sft_wsl_ubuntu2404.md`

初始化环境（WSL 推荐 Python 3.10）：

> 提示：本文档使用 `$VOICELAB_DIR` 指向 VoiceLab 仓库根目录（不要求固定路径）。如果你在仓库根目录，可先执行：
>
> ```bash
> export VOICELAB_DIR="$PWD"
> ```

```bash
cd "$VOICELAB_DIR/workflows/cosyvoice"
uv sync
```

## （可选）先用 MSST workflow 做音频净化

如果你希望把输入音频做成更干净的干声（去和声、去混响、降噪），可以先用：
`workflows/msst` 生成 `*_vocals_karaoke_noreverb_dry.wav` 再进入后续流程。
