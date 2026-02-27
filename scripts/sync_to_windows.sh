#!/usr/bin/env bash
set -euo pipefail

# 将当前仓库同步到 Windows 原生目录（WSL 下的 /mnt/c/...）。
#
# 软链接说明：
# - 本仓库包含大量软链接，其中有些是“绝对路径”（例如指向 /home/... 或 /mnt/c/...）。
# - 复制到 Windows 后，如果你打算在“Windows 原生”环境（非 WSL）直接使用这些链接，
#   通常需要：Windows 开发者模式/管理员权限 + 用 Windows 风格路径重新创建链接。
# - 因此本脚本默认使用 LINK_MODE=deref：把软链接“解引用”为真实文件/目录复制，
#   避免 Windows 对 symlink 的权限/兼容性问题。
#
# 用法：
#   scripts/sync_to_windows.sh
#   LINK_MODE=preserve scripts/sync_to_windows.sh     # 尽量保留软链接（可能导致 Windows 下不可用）
#   DST=/mnt/c/Projects/AntiGravityProjects/VoiceLab scripts/sync_to_windows.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"
DST="${DST:-/mnt/c/Projects/AntiGravityProjects/VoiceLab}"

# deref: 解引用软链接并复制目标内容（Windows 最省心）
# preserve: 保留软链接（需要 Windows/文件系统能正确支持 symlink）
LINK_MODE="${LINK_MODE:-deref}" # deref | preserve

rsync_link_args=()
case "${LINK_MODE}" in
  deref) rsync_link_args+=(--copy-links) ;;
  preserve) rsync_link_args+=(--links) ;;
  *)
    echo "Unknown LINK_MODE=${LINK_MODE} (expected: deref|preserve)" >&2
    exit 2
    ;;
esac

mkdir -p "${DST}"

# 这些 exclude 对应“外部资产软链接”（通常指向 Windows 盘上的大模型/权重目录）。
# 默认不解引用复制它们，避免把一大堆文件同步进工程目录。
# 如果你希望把它们也打包到 Windows 工程目录里：删除对应的 --exclude。
rsync -avm --delete \
  "${rsync_link_args[@]}" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='**/.venv/' \
  --exclude='.wiki-export/' \
  --exclude='**/__pycache__/' \
  --exclude='**/.cache/' \
  --exclude='**/.pytest_cache/' \
  --exclude='**/.ipynb_checkpoints/' \
  --exclude='**/.mypy_cache/' \
  --exclude='**/.ruff_cache/' \
  --exclude='.vscode/' \
  --exclude='.idea/' \
  --exclude='.vs/' \
  --exclude='.DS_Store' \
  --exclude='Thumbs.db' \
  --exclude='datasets/' \
  --exclude='workflows/**/data/' \
  --exclude='workflows/**/exp/' \
  --exclude='workflows/**/pretrained_models/' \
  --exclude='workflows/**/tensorboard/' \
  --exclude='workflows/**/out_wav/' \
  --exclude='workflows/**/runtime/logs/' \
  --exclude='workflows/rvc/runtime/assets/pretrained' \
  --exclude='workflows/rvc/runtime/assets/pretrained_v2' \
  --exclude='workflows/rvc/runtime/assets/hubert/hubert_base.pt' \
  --exclude='workflows/rvc/runtime/assets/rmvpe/rmvpe.pt' \
  "${SRC}/" "${DST}/"
