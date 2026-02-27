#!/usr/bin/env bash
set -euo pipefail

# 只同步 RVC 训练相关产物到 Windows 目录（WSL: /mnt/c/...）。
# 适用场景：代码/依赖在 Windows 侧自行准备，只想把训练出来的权重/索引/日志拷过去。
#
# 默认 PROFILE=small：只同步“必要训练产物”，避免把巨大的中间特征/音频一起拷贝。
# - checkpoints: workflows/rvc/runtime/logs/**/G_*.pth D_*.pth
# - configs/filelist/logs: config.json filelist.txt *.log events.out.tfevents.*
# - exported weights: workflows/rvc/runtime/assets/weights/
# - indices: workflows/rvc/runtime/indices/
#
# PROFILE=full：把 logs/<model>/ 下的所有内容都同步（通常非常大）。
#
# 用法：
#   scripts/sync_rvc_training_to_windows.sh
#   DST=/mnt/c/Projects/AntiGravityProjects/VoiceLab scripts/sync_rvc_training_to_windows.sh
#   PROFILE=full scripts/sync_rvc_training_to_windows.sh
#   MODEL=xingtong_v2_48k_f0 scripts/sync_rvc_training_to_windows.sh   # 只同步某个模型目录

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"
DST="${DST:-/mnt/c/Projects/AntiGravityProjects/VoiceLab}"

PROFILE="${PROFILE:-small}" # small | full
MODEL="${MODEL:-}"          # 例如 xingtong_v2_48k_f0；为空表示全部

mkdir -p "${DST}"

rsync_common=(
  -avm
  --delete
  --copy-links
)

if [[ -n "${MODEL}" ]]; then
  model_glob="workflows/rvc/runtime/logs/${MODEL}"
else
  model_glob="workflows/rvc/runtime/logs"
fi

case "${PROFILE}" in
  small)
    rsync "${rsync_common[@]}" \
      --include='workflows/' \
      --include='workflows/rvc/' \
      --include='workflows/rvc/runtime/' \
      --include='workflows/rvc/runtime/assets/' \
      --include='workflows/rvc/runtime/assets/weights/***' \
      --include='workflows/rvc/runtime/indices/***' \
      --exclude='workflows/rvc/runtime/logs/mute' \
      --include="${model_glob}/" \
      --include="${model_glob}/**/" \
      --include="${model_glob}/**/config.json" \
      --include="${model_glob}/**/filelist.txt" \
      --include="${model_glob}/**/*log" \
      --include="${model_glob}/**/events.out.tfevents.*" \
      --include="${model_glob}/**/G_*.pth" \
      --include="${model_glob}/**/D_*.pth" \
      --include="${model_glob}/**/eval/***" \
      --exclude='*' \
      "${SRC}/" "${DST}/"
    ;;
  full)
    rsync "${rsync_common[@]}" \
      --include='workflows/' \
      --include='workflows/rvc/' \
      --include='workflows/rvc/runtime/' \
      --include='workflows/rvc/runtime/assets/' \
      --include='workflows/rvc/runtime/assets/weights/***' \
      --include='workflows/rvc/runtime/indices/***' \
      --exclude='workflows/rvc/runtime/logs/mute' \
      --include="${model_glob}/***" \
      --exclude='*' \
      "${SRC}/" "${DST}/"
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE} (expected: small|full)" >&2
    exit 2
    ;;
esac
