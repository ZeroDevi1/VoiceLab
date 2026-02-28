from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORKFLOWS: list[str] = ["cosyvoice", "rvc", "msst"]
KNOWN_WORKFLOWS: set[str] = set(DEFAULT_WORKFLOWS)


def parse_workflows(value: str) -> list[str]:
    """
    Parse a comma-separated workflow list.

    - Empty string => DEFAULT_WORKFLOWS
    - Dedup while preserving order
    - Validate against KNOWN_WORKFLOWS
    """
    s = (value or "").strip()
    if not s:
        return list(DEFAULT_WORKFLOWS)

    out: list[str] = []
    seen: set[str] = set()
    for raw in s.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in KNOWN_WORKFLOWS:
            raise ValueError(f"Unknown workflow: {name}")
        if name in seen:
            continue
        out.append(name)
        seen.add(name)
    if not out:
        return list(DEFAULT_WORKFLOWS)
    return out


def resolve_assets_dir(arg: str | None) -> Path:
    env = os.environ.get("VOICELAB_ASSETS_DIR")
    base = (arg or env or "~/.cache/voicelab/assets").strip()
    return Path(base).expanduser().resolve()


def apply_git_mirror_prefix(url: str, prefix: str | None) -> str:
    """
    Apply a GitHub HTTPS mirror prefix (e.g. ghproxy) to a URL.

    We only rewrite URLs that start with "https://github.com/".
    """
    if not prefix:
        return url
    if url.startswith(prefix):
        return url
    if not url.startswith("https://github.com/"):
        return url
    p = prefix.rstrip("/") + "/"
    # For ghproxy-style mirrors, the expected form is:
    #   https://github.com/<owner>/<repo>.git
    return f"{p}{url}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_symlink(*, src: Path, dst: Path, force: bool) -> None:
    """
    Create/replace a symlink. If symlinks are not supported, fall back to copy.

    Windows 注意：必须正确传入 target_is_directory，否则会把“目录链接”创建成“文件链接”，
    后续对 dst 做 Path.exists()/os.stat() 可能直接触发 PermissionError（WinError 5）。
    """

    def _is_reparse_point(p: Path) -> bool:
        # 保护性判断：Windows 下某些目录联接/重解析点不是 symlink，但递归删除会误删目标目录内容。
        if os.name != "nt":
            return False
        try:
            st = os.lstat(p)
        except OSError:
            return False
        attrs = getattr(st, "st_file_attributes", 0)
        try:
            import stat as _stat

            rp = getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        except Exception:
            rp = 0
        return bool(rp and (attrs & rp))

    try:
        dst_exists = dst.exists()
    except OSError:
        # 例如：错误类型的符号链接（file symlink -> directory）在 Windows 上可能导致 exists() 抛错
        dst_exists = False

    if dst.is_symlink() or dst_exists:
        if not force:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink(missing_ok=True)
        elif _is_reparse_point(dst):
            # 对 reparse point 只删除“链接本身”，不要递归删（避免误删目标目录内容）
            os.rmdir(dst)
        else:
            # Only delete directories when the caller explicitly opts in.
            import shutil

            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src), str(dst), target_is_directory=src.is_dir())
    except OSError:
        # Rare on WSL/Linux, but keep a conservative fallback.
        import shutil

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


@dataclass(frozen=True)
class DownloadSpec:
    url: str
    dest: Path


def _http_download(
    *,
    url: str,
    dest: Path,
    force: bool,
    timeout_s: int = 60,
    retries: int = 3,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return

    partial = dest.with_suffix(dest.suffix + ".partial")
    if force and partial.exists():
        partial.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            resume_from = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "voicelab-bootstrap/0.1"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                # If server ignored Range, restart.
                if resume_from > 0 and getattr(resp, "status", None) == 200:
                    resume_from = 0
                mode = "ab" if resume_from > 0 else "wb"
                with partial.open(mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            os.replace(partial, dest)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt >= retries:
                raise
            # Exponential-ish backoff.
            sleep_s = min(10, 1 + attempt * 2)
            print(
                f"[voicelab] WARN: download failed (attempt {attempt}/{retries}): {e}; retry in {sleep_s}s"
            )
            time.sleep(sleep_s)


def download_many(
    *, specs: list[DownloadSpec], force: bool, timeout_s: int = 60, retries: int = 3
) -> None:
    for s in specs:
        print(f"[voicelab] download: {s.dest.name}", flush=True)
        _http_download(
            url=s.url, dest=s.dest, force=force, timeout_s=timeout_s, retries=retries
        )


def rvc_required_assets(*, hf_base: str, dest_root: Path) -> list[DownloadSpec]:
    """
    Build the 'train required' RVC asset set (v2 + 48k + f0).
    """
    hf = hf_base.rstrip("/")
    base = f"{hf}/lj1995/VoiceConversionWebUI/resolve/main"
    return [
        DownloadSpec(
            url=f"{base}/hubert_base.pt", dest=dest_root / "hubert" / "hubert_base.pt"
        ),
        DownloadSpec(url=f"{base}/rmvpe.pt", dest=dest_root / "rmvpe" / "rmvpe.pt"),
        DownloadSpec(
            url=f"{base}/pretrained_v2/f0G48k.pth",
            dest=dest_root / "pretrained_v2" / "f0G48k.pth",
        ),
        DownloadSpec(
            url=f"{base}/pretrained_v2/f0D48k.pth",
            dest=dest_root / "pretrained_v2" / "f0D48k.pth",
        ),
    ]
