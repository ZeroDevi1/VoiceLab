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
}

# Home
copy "docs/index.md" "Home.md"

# Export docs/**/*.md (flattened page names).
sidebar="${out_dir}/_Sidebar.md"
{
  echo "- [[Home]]"
} >"${sidebar}"

while IFS= read -r -d '' f; do
  rel="${f#docs/}"
  base="${rel%.md}"
  if [[ "${rel}" == "index.md" ]]; then
    continue
  fi
  # Flatten "workflows/rvc/foo.md" -> "workflows__rvc__foo.md"
  page="${base//\//__}"
  copy "${f}" "${page}.md"
  echo "- [[${page}]]" >>"${sidebar}"
done < <(find docs -type f -name "*.md" -print0 | sort -z)

sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat >"${out_dir}/_Footer.md" <<EOF
Synced from \`${sha}\`.
EOF

echo "exported wiki markdown to: ${out_dir}"

