#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/wiki_export.sh <out_dir>

Exports Markdown docs/ into a directory that can be mirrored into GitHub Wiki (*.wiki.git).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

out_dir="${1:-}"
if [[ -z "${out_dir}" ]]; then
  usage >&2
  exit 2
fi

if [[ "${out_dir}" == "/" || "${out_dir}" == "." || "${out_dir}" == ".." ]]; then
  echo "error: unsafe out_dir: ${out_dir}" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
  echo "error: must be run inside a git repository" >&2
  exit 2
fi

cd "${repo_root}"

rm -rf "${out_dir}"
mkdir -p "${out_dir}"

copy() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "error: missing source markdown: ${src}" >&2
    exit 1
  fi
  cp -f "${src}" "${out_dir}/${dst}"
  chmod 0644 "${out_dir}/${dst}"
}

# Home (GitHub Wiki uses Home.md as the landing page)
copy "docs/index.md" "Home.md"

# Curated export with Chinese page names for better readability in Wiki sidebar.
#
# Format: "source_path|Wiki Page Name (file name without .md)"
pages=(
  "docs/WIKI_SYNC.md|Wiki 同步说明"
  "docs/workflows/rvc/rvc_xingtong_wsl_ubuntu2404.md|RVC 星瞳（WSL Ubuntu24.04）"
  "docs/workflows/rvc/rvc_xuan_wsl_ubuntu2404.md|RVC xuan（WSL Ubuntu24.04）"
  "docs/workflows/cosyvoice/cosyvoice_xuan_sft_wsl_ubuntu2404.md|CosyVoice xuan SFT（WSL Ubuntu24.04）"
  "docs/workflows/cosyvoice/cosyvoice_sft_param_playbook_wsl_ubuntu2404.md|CosyVoice SFT 参数手册（WSL Ubuntu24.04）"
  "docs/workflows/cosyvoice/cosyvoice_xingtong_sft_wsl_ubuntu2404.md|CosyVoice 星瞳 SFT（WSL Ubuntu24.04）"
  "docs/workflows/cosyvoice/cosyvoice_dream_sft_wsl_ubuntu2404.md|CosyVoice Dream SFT（WSL Ubuntu24.04）"
)

sidebar="${out_dir}/_Sidebar.md"
{
  echo "- [[Home]]"
} >"${sidebar}"

for entry in "${pages[@]}"; do
  src="${entry%%|*}"
  name="${entry#*|}"
  # Avoid unsafe filenames.
  safe_name="${name//\//-}"
  copy "${src}" "${safe_name}.md"
  echo "- [[${safe_name}]]" >>"${sidebar}"
done

sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat >"${out_dir}/_Footer.md" <<EOF
Synced from \`${sha}\`.
EOF

echo "exported wiki markdown to: ${out_dir}"
