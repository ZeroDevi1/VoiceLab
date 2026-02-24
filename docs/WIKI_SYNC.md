# GitHub Wiki Sync

本仓库支持将 `docs/` 自动同步到 GitHub Wiki（`<repo>.wiki.git`）。

## 1) 在 GitHub 启用 Wiki

在 GitHub 仓库页面：
- Settings -> Features -> 勾选 **Wikis**

如果未启用 Wiki，`<repo>.wiki.git` 可能不存在，GitHub Actions 会 push 失败。

## 2) Actions 权限 / Token

默认使用 `secrets.WIKI_TOKEN || github.token`：
- 一般情况下 `github.token` 即可（workflow 已设置 `permissions: contents: write`）
- 如果遇到权限问题，建议配置 `WIKI_TOKEN`（PAT），最小权限建议包含 `contents: write`

## 3) 本地预览导出结果

```bash
bash scripts/wiki_export.sh .wiki-export
ls -la .wiki-export
```

导出目录 `.wiki-export/` 的内容会被 rsync 到 `<repo>.wiki.git`。

