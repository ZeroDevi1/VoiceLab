from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VendorRepo:
    name: str
    url: str
    dest: Path
    post_clone: tuple[tuple[str, ...], ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _vendor_dir() -> Path:
    env = os.environ.get("VOICELAB_VENDOR_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "vendor"


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    return int(p.returncode)


def _git(*args: str, cwd: Path | None = None) -> int:
    return _run(["git", *args], cwd=cwd)


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _is_dirty(path: Path) -> bool:
    p = subprocess.run(["git", "status", "--porcelain"], cwd=str(path), capture_output=True, text=True)
    if p.returncode != 0:
        return True
    return bool(p.stdout.strip())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _vendor_repos() -> list[VendorRepo]:
    vendor = _vendor_dir()
    return [
        VendorRepo(
            name="CosyVoice",
            url="https://github.com/FunAudioLLM/CosyVoice",
            dest=vendor / "CosyVoice",
            post_clone=(("submodule", "update", "--init", "--recursive"),),
        ),
        VendorRepo(
            name="Retrieval-based-Voice-Conversion-WebUI",
            url="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
            dest=vendor / "RVC",
        ),
        VendorRepo(
            name="GPT-SoVITS",
            url="https://github.com/RVC-Boss/GPT-SoVITS",
            dest=vendor / "GPT-SoVITS",
        ),
    ]


def cmd_vendor_sync(args: argparse.Namespace) -> int:
    _ensure_dir(_vendor_dir())
    repos = _vendor_repos()

    for repo in repos:
        dest = repo.dest
        if dest.exists():
            # Symlink is allowed; treat it as "existing".
            if not _is_git_repo(dest):
                print(f"[voicelab] SKIP: {repo.name} exists but is not a git repo: {dest}")
                continue

            if _is_dirty(dest) and not args.force:
                print(f"[voicelab] SKIP: {repo.name} has local changes (use --force to reset): {dest}")
                continue

            if args.force:
                rc = _git("reset", "--hard", cwd=dest)
                if rc != 0:
                    return rc
                rc = _git("clean", "-fd", cwd=dest)
                if rc != 0:
                    return rc

            rc = _git("fetch", "--all", "--prune", cwd=dest)
            if rc != 0:
                return rc
            # Try fast-forward only.
            rc = _git("pull", "--ff-only", cwd=dest)
            if rc != 0:
                print(f"[voicelab] WARN: pull failed (maybe detached head or no upstream): {dest}")
            # Keep submodules fresh for CosyVoice
            for post in repo.post_clone:
                rc = _git(*post, cwd=dest)
                if rc != 0:
                    return rc
            print(f"[voicelab] OK: updated {repo.name} -> {dest}")
            continue

        clone_cmd = ["clone"]
        if args.depth and args.depth > 0:
            clone_cmd += ["--depth", str(args.depth)]
        clone_cmd += [repo.url, str(dest)]
        rc = _git(*clone_cmd)
        if rc != 0:
            return rc
        for post in repo.post_clone:
            rc = _git(*post, cwd=dest)
            if rc != 0:
                return rc
        print(f"[voicelab] OK: cloned {repo.name} -> {dest}")

    return 0


def cmd_vendor_status(_args: argparse.Namespace) -> int:
    repos = _vendor_repos()
    for repo in repos:
        dest = repo.dest
        if not dest.exists():
            print(f"[voicelab] MISSING: {repo.name} -> {dest}")
            continue
        if not _is_git_repo(dest):
            print(f"[voicelab] PRESENT (non-git): {repo.name} -> {dest}")
            continue
        dirty = "dirty" if _is_dirty(dest) else "clean"
        print(f"[voicelab] PRESENT ({dirty}): {repo.name} -> {dest}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    rc = cmd_vendor_sync(args)
    if rc != 0:
        return rc
    print("")
    print("[voicelab] Next steps:")
    print("  - CosyVoice env: cd workflows/cosyvoice && uv sync")
    print("  - CosyVoice doc: workflows/cosyvoice/docs/cosyvoice_xuan_sft_wsl_ubuntu2404.md")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="voicelab", description="VoiceLab bootstrap utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Clone/update vendor repos, then print next steps.")
    p_init.add_argument("--force", action="store_true", help="Reset+clean vendor repos before pulling.")
    p_init.add_argument("--depth", type=int, default=1, help="Git clone depth (0 for full clone).")
    p_init.set_defaults(func=cmd_init)

    p_vendor = sub.add_parser("vendor", help="Manage vendor repos.")
    vendor_sub = p_vendor.add_subparsers(dest="vendor_cmd", required=True)

    p_sync = vendor_sub.add_parser("sync", help="Clone/update vendor repos under vendor/.")
    p_sync.add_argument("--force", action="store_true", help="Reset+clean vendor repos before pulling.")
    p_sync.add_argument("--depth", type=int, default=1, help="Git clone depth (0 for full clone).")
    p_sync.set_defaults(func=cmd_vendor_sync)

    p_status = vendor_sub.add_parser("status", help="Print vendor repos status.")
    p_status.set_defaults(func=cmd_vendor_status)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

