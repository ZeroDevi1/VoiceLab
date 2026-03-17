from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from voicelab.bootstrap import (
    apply_git_mirror_prefix,
    download_many,
    ensure_symlink,
    parse_workflows,
    resolve_annotation_dir,
    resolve_assets_dir,
    rvc_required_assets,
)


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


def _git_get_stdout(*args: str, cwd: Path) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    return int(p.returncode), str(p.stdout or "")


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _is_dirty(path: Path) -> bool:
    p = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(path), capture_output=True, text=True
    )
    if p.returncode != 0:
        return True
    return bool(p.stdout.strip())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _workflow_dir(name: str) -> Path:
    return _repo_root() / "workflows" / name


def _workflow_has_env(name: str) -> bool:
    return (_workflow_dir(name) / ".venv").exists()


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _runtime_has_rvc_core(rt: Path) -> bool:
    return all(
        _safe_exists(p)
        for p in [
            rt / "infer",
            rt / "configs",
            rt / "assets" / "hubert" / "hubert_base.pt",
            rt / "assets" / "rmvpe" / "rmvpe.pt",
        ]
    )


def _runtime_has_msst_core(rt: Path) -> bool:
    return all(
        _safe_exists(p)
        for p in [
            rt / "inference",
            rt / "modules",
            rt / "utils",
            rt / "configs",
            rt / "pretrain" / "vocal_models" / "inst_v1e.ckpt",
            rt / "pretrain" / "vocal_models" / "big_beta5e.ckpt",
            rt
            / "pretrain"
            / "single_stem_models"
            / "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        ]
    )


def _print_check(status: str, label: str, detail: str) -> None:
    print(f"[{status}] {label}: {detail}")


def _gpt_pretrained_paths() -> list[Path]:
    shared = resolve_assets_dir(None) / "gpt_sovits" / "GPT_SoVITS"
    return [
        shared / "pretrained_models" / "chinese-roberta-wwm-ext-large",
        shared / "pretrained_models" / "chinese-hubert-base",
        shared / "pretrained_models" / "s2G488k.pth",
        shared
        / "pretrained_models"
        / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        shared / "text" / "G2PWModel",
    ]


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
        VendorRepo(
            name="MSST-WebUI",
            url="https://github.com/SUC-DriverOld/MSST-WebUI",
            dest=vendor / "MSST-WebUI",
        ),
    ]


def cmd_vendor_sync(args: argparse.Namespace) -> int:
    _ensure_dir(_vendor_dir())
    repos = _vendor_repos()

    git_mirror_prefix = getattr(args, "git_mirror_prefix", None)

    for repo in repos:
        dest = repo.dest
        if dest.exists():
            # Symlink is allowed; treat it as "existing".
            if not _is_git_repo(dest):
                print(
                    f"[voicelab] SKIP: {repo.name} exists but is not a git repo: {dest}"
                )
                continue

            # Ensure future fetch/pull uses the mirror URL (if provided).
            if git_mirror_prefix:
                mirrored = apply_git_mirror_prefix(repo.url, git_mirror_prefix)
                # Preserve original as "upstream" for easy recovery/debug.
                rc_up, _ = _git_get_stdout("remote", "get-url", "upstream", cwd=dest)
                if rc_up == 0:
                    _git("remote", "set-url", "upstream", repo.url, cwd=dest)
                else:
                    _git("remote", "add", "upstream", repo.url, cwd=dest)
                _git("remote", "set-url", "origin", mirrored, cwd=dest)

            if _is_dirty(dest) and not args.force:
                print(
                    f"[voicelab] SKIP: {repo.name} has local changes (use --force to reset): {dest}"
                )
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
                print(
                    f"[voicelab] WARN: pull failed (maybe detached head or no upstream): {dest}"
                )
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
        clone_url = apply_git_mirror_prefix(repo.url, git_mirror_prefix)
        clone_cmd += [clone_url, str(dest)]
        rc = _git(*clone_cmd)
        if rc != 0:
            return rc
        if git_mirror_prefix:
            # Keep upstream pointing at the original repo.
            _git("remote", "add", "upstream", repo.url, cwd=dest)
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
    print(
        "  - CosyVoice doc: docs/workflows/cosyvoice/cosyvoice_xuan_sft_wsl_ubuntu2404.md"
    )
    print("  - GPT-SoVITS env: cd workflows/gpt_sovits && uv sync")
    print(
        "  - GPT-SoVITS doc: docs/workflows/gpt_sovits/gpt_sovits_prepare_train_wsl_ubuntu2404.md"
    )
    print("  - MSST env:      cd workflows/msst && uv sync")
    print(
        "  - MSST init:     cd workflows/msst && uv run python tools/msst_init_runtime.py"
    )
    return 0


def _run_checked(
    cmd: list[str], *, cwd: Path | None, env: dict[str, str] | None, dry_run: bool
) -> None:
    pretty = " ".join(cmd)
    prefix = f"[voicelab] $ {pretty}"
    if cwd is not None:
        prefix += f"  (cwd={cwd})"
    print(prefix, flush=True)
    if dry_run:
        return
    p = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def cmd_bootstrap(args: argparse.Namespace) -> int:
    repo_root = _repo_root()

    # Defaults (profile + env) with explicit CLI args taking priority.
    profile = str(args.profile or "cn").strip().lower()
    hf_base = (
        str(
            args.hf_base
            or os.environ.get("VOICELAB_HF_BASE")
            or (
                "https://hf-mirror.com" if profile == "cn" else "https://huggingface.co"
            )
        )
        .strip()
        .rstrip("/")
    )
    git_mirror_prefix = (
        args.git_mirror_prefix
        or os.environ.get("VOICELAB_GIT_MIRROR_PREFIX")
        or ("" if profile == "cn" else None)
    )

    workflows = parse_workflows(str(args.workflows or ""))
    assets_dir = resolve_assets_dir(args.assets_dir)
    rvc_assets_dir = assets_dir / "rvc"
    msst_assets_pretrain = assets_dir / "msst" / "pretrain"
    cosy_assets_root = assets_dir / "cosyvoice" / "pretrained_models"
    cosy_assets_model = cosy_assets_root / "Fun-CosyVoice3-0.5B"
    gpt_note_dir = assets_dir / "gpt_sovits"

    print(
        f"[voicelab] bootstrap profile={profile} workflows={','.join(workflows)}",
        flush=True,
    )
    print(f"[voicelab] assets dir: {assets_dir}", flush=True)
    if "gpt_sovits" in workflows:
        print(
            "[voicelab] NOTE: GPT-SoVITS bootstrap will prepare workflow env and common shared pretrained assets.",
            flush=True,
        )

    if not args.no_vendor:
        # Reuse vendor sync, with mirror enabled by default for cn.
        ns = argparse.Namespace(
            force=bool(getattr(args, "vendor_force", False)),
            depth=1,
            git_mirror_prefix=git_mirror_prefix,
        )
        if args.dry_run:
            print(
                "[voicelab] (dry-run) vendor sync: "
                + f"force={bool(ns.force)} depth={int(ns.depth)} git_mirror_prefix={git_mirror_prefix}",
                flush=True,
            )
        else:
            rc = cmd_vendor_sync(ns)
            if rc != 0:
                return rc

    # Ensure Python 3.10 is available to uv.
    _run_checked(
        ["uv", "python", "install", "3.10"],
        cwd=repo_root,
        env=None,
        dry_run=bool(args.dry_run),
    )

    # Workflow env sync
    if not args.no_env_sync:
        for wf in workflows:
            wf_dir = repo_root / "workflows" / wf
            if not wf_dir.exists():
                raise SystemExit(f"Workflow dir not found: {wf_dir}")

            _run_checked(
                ["uv", "python", "pin", "3.10"],
                cwd=wf_dir,
                env=None,
                dry_run=bool(args.dry_run),
            )
            env = os.environ.copy()
            env.setdefault("UV_HTTP_TIMEOUT", "600")
            # 约束构建隔离环境里的构建依赖（例如 openai-whisper sdist 需要 pkg_resources）。
            # uv 支持通过 UV_BUILD_CONSTRAINT 指向 constraints 文件来 pin 构建依赖版本。
            build_constraints = wf_dir / "uv-build-constraints.txt"
            if build_constraints.exists():
                env.setdefault("UV_BUILD_CONSTRAINT", str(build_constraints))
            _run_checked(
                ["uv", "sync"], cwd=wf_dir, env=env, dry_run=bool(args.dry_run)
            )

    # Assets + runtime init
    if not args.no_assets:
        if "rvc" in workflows:
            specs = rvc_required_assets(hf_base=hf_base, dest_root=rvc_assets_dir)
            if args.dry_run:
                for s in specs:
                    print(f"[voicelab] (dry-run) download: {s.url} -> {s.dest}")
            else:
                download_many(specs=specs, force=bool(args.force))

        if "msst" in workflows:
            msst_dir = repo_root / "workflows" / "msst"
            _run_checked(
                [
                    "uv",
                    "run",
                    "python",
                    "tools/msst_download_models.py",
                    "--hf-base",
                    hf_base,
                    "--dest-root",
                    str(msst_assets_pretrain),
                    "--dest-layout",
                    "pretrain",
                    *([] if not args.force else ["--force"]),
                ],
                cwd=msst_dir,
                env=None,
                dry_run=bool(args.dry_run),
            )

        if "cosyvoice" in workflows:
            cosy_dir = repo_root / "workflows" / "cosyvoice"
            _run_checked(
                [
                    "uv",
                    "run",
                    "python",
                    "tools/download_pretrained_cosyvoice3.py",
                    "--source",
                    "modelscope" if profile == "cn" else "hf",
                    "--local_dir",
                    str(cosy_assets_model),
                ],
                cwd=cosy_dir,
                env=None,
                dry_run=bool(args.dry_run),
            )

        if "gpt_sovits" in workflows:
            gpt_dir = repo_root / "workflows" / "gpt_sovits"
            _run_checked(
                [
                    "uv",
                    "run",
                    "python",
                    "tools/download_pretrained_gpt_sovits.py",
                    "--hf-base",
                    hf_base,
                    "--dest-root",
                    str(gpt_note_dir),
                    *([] if not args.force else ["--force"]),
                ],
                cwd=gpt_dir,
                env=None,
                dry_run=bool(args.dry_run),
            )

    if not args.no_runtime:
        if "rvc" in workflows:
            rvc_dir = repo_root / "workflows" / "rvc"
            _run_checked(
                [
                    "uv",
                    "run",
                    "python",
                    "tools/rvc_init_runtime.py",
                    *(["--force"] if args.force else []),
                    "--assets-src",
                    str(rvc_assets_dir),
                ],
                cwd=rvc_dir,
                env=None,
                dry_run=bool(args.dry_run),
            )

        if "msst" in workflows:
            msst_dir = repo_root / "workflows" / "msst"
            _run_checked(
                [
                    "uv",
                    "run",
                    "python",
                    "tools/msst_init_runtime.py",
                    *(["--force"] if args.force else []),
                    "--assets-src",
                    str(msst_assets_pretrain),
                    "--hf-base",
                    hf_base,
                ],
                cwd=msst_dir,
                env=None,
                dry_run=bool(args.dry_run),
            )

        if "cosyvoice" in workflows and not args.dry_run:
            # Keep workflow paths stable (docs use relative "pretrained_models/...").
            cosy_dir = repo_root / "workflows" / "cosyvoice"
            ensure_symlink(
                src=cosy_assets_root,
                dst=cosy_dir / "pretrained_models",
                force=bool(args.force),
            )

        if "gpt_sovits" in workflows and not args.dry_run:
            gpt_runtime = _workflow_dir("gpt_sovits") / "runtime"
            _ensure_dir(gpt_runtime / "weights" / "gpt")
            _ensure_dir(gpt_runtime / "weights" / "sovits")

    print("[voicelab] OK: bootstrap complete")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    assets_dir = resolve_assets_dir(None)
    annotation_dir = resolve_annotation_dir(None)
    failed = 0
    warned = 0

    print(f"[voicelab] doctor repo={repo_root}")

    rsync_required = os.name != "nt"
    tool_checks: list[tuple[str, str, bool]] = [
        ("git", "git", True),
        ("uv", "uv", True),
        ("ffmpeg", "ffmpeg", True),
        ("rsync", "rsync", rsync_required),
        ("nvidia-smi", "nvidia-smi", False),
    ]
    for label, binary, required in tool_checks:
        hit = shutil.which(binary)
        if hit:
            _print_check("OK", label, hit)
            continue
        if label == "rsync" and os.name == "nt":
            _print_check(
                "INFO",
                label,
                "optional on Windows; required only for dataset stage scripts",
            )
            continue
        if required:
            failed += 1
            _print_check("FAIL", label, "not found in PATH")
        else:
            warned += 1
            _print_check("WARN", label, "not found in PATH")

    for repo in _vendor_repos():
        if repo.dest.exists() and _is_git_repo(repo.dest):
            dirty = "dirty" if _is_dirty(repo.dest) else "clean"
            _print_check("OK", f"vendor:{repo.name}", f"{repo.dest} ({dirty})")
        elif repo.dest.exists():
            warned += 1
            _print_check(
                "WARN",
                f"vendor:{repo.name}",
                f"present but not a git repo: {repo.dest}",
            )
        else:
            failed += 1
            _print_check("FAIL", f"vendor:{repo.name}", f"missing: {repo.dest}")

    for wf in parse_workflows(""):
        wf_dir = _workflow_dir(wf)
        if not wf_dir.exists():
            failed += 1
            _print_check("FAIL", f"workflow:{wf}", f"missing dir: {wf_dir}")
            continue
        if _workflow_has_env(wf):
            _print_check("OK", f"env:{wf}", str(wf_dir / ".venv"))
        else:
            warned += 1
            _print_check(
                "WARN",
                f"env:{wf}",
                f"missing {wf_dir / '.venv'}; run bootstrap or uv sync",
            )

    _print_check("OK", "assets-dir", str(assets_dir))
    if annotation_dir.exists():
        _print_check("OK", "annotation-dir", str(annotation_dir))
    else:
        warned += 1
        _print_check(
            "WARN",
            "annotation-dir",
            f"missing {annotation_dir}; set VOICELAB_ANNOTATION_DIR if you use centralized .list files",
        )

    rvc_runtime = _workflow_dir("rvc") / "runtime"
    if _runtime_has_rvc_core(rvc_runtime):
        _print_check("OK", "runtime:rvc", str(rvc_runtime))
    else:
        warned += 1
        _print_check(
            "WARN",
            "runtime:rvc",
            "runtime incomplete; run `uv run -m voicelab bootstrap --workflows rvc`",
        )

    msst_runtime = _workflow_dir("msst") / "runtime"
    if _runtime_has_msst_core(msst_runtime):
        _print_check("OK", "runtime:msst", str(msst_runtime))
    else:
        warned += 1
        _print_check(
            "WARN",
            "runtime:msst",
            "runtime incomplete; run `uv run -m voicelab bootstrap --workflows msst`",
        )

    cosy_pretrained = _workflow_dir("cosyvoice") / "pretrained_models"
    if cosy_pretrained.exists():
        _print_check("OK", "runtime:cosyvoice", str(cosy_pretrained))
    else:
        warned += 1
        _print_check(
            "WARN",
            "runtime:cosyvoice",
            "missing pretrained_models link; run `uv run -m voicelab bootstrap --workflows cosyvoice`",
        )

    gpt_runtime = _workflow_dir("gpt_sovits") / "runtime" / "weights"
    if gpt_runtime.exists():
        _print_check("OK", "runtime:gpt_sovits", str(gpt_runtime))
    else:
        warned += 1
        _print_check(
            "WARN",
            "runtime:gpt_sovits",
            "missing runtime weights dirs; run `uv run -m voicelab bootstrap --workflows gpt_sovits`",
        )

    gpt_pretrained = _gpt_pretrained_paths()
    missing_gpt_pretrained = [p for p in gpt_pretrained if not _safe_exists(p)]
    if not missing_gpt_pretrained:
        _print_check(
            "OK",
            "pretrained:gpt_sovits",
            str((resolve_assets_dir(None) / "gpt_sovits" / "GPT_SoVITS").resolve()),
        )
    else:
        warned += 1
        _print_check(
            "WARN",
            "pretrained:gpt_sovits",
            "missing common shared pretrained assets: "
            + ", ".join(p.name for p in missing_gpt_pretrained),
        )

    print(f"[voicelab] doctor summary: fail={failed} warn={warned}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="voicelab", description="VoiceLab bootstrap utilities."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser(
        "init", help="Clone/update vendor repos, then print next steps."
    )
    p_init.add_argument(
        "--force", action="store_true", help="Reset+clean vendor repos before pulling."
    )
    p_init.add_argument(
        "--depth", type=int, default=1, help="Git clone depth (0 for full clone)."
    )
    p_init.add_argument(
        "--git-mirror-prefix",
        default=None,
        help="Optional GitHub mirror prefix (e.g. ) for cloning/updating vendor repos.",
    )
    p_init.set_defaults(func=cmd_init)

    p_vendor = sub.add_parser("vendor", help="Manage vendor repos.")
    vendor_sub = p_vendor.add_subparsers(dest="vendor_cmd", required=True)

    p_sync = vendor_sub.add_parser(
        "sync", help="Clone/update vendor repos under vendor/."
    )
    p_sync.add_argument(
        "--force", action="store_true", help="Reset+clean vendor repos before pulling."
    )
    p_sync.add_argument(
        "--depth", type=int, default=1, help="Git clone depth (0 for full clone)."
    )
    p_sync.add_argument(
        "--git-mirror-prefix",
        default=None,
        help="Optional GitHub mirror prefix (e.g. ) for cloning/updating vendor repos.",
    )
    p_sync.set_defaults(func=cmd_vendor_sync)

    p_status = vendor_sub.add_parser("status", help="Print vendor repos status.")
    p_status.set_defaults(func=cmd_vendor_status)

    p_bootstrap = sub.add_parser(
        "bootstrap",
        help="One-command bootstrap: vendor sync + uv sync + model downloads + runtime init.",
    )
    p_bootstrap.add_argument(
        "--profile",
        default="cn",
        choices=["cn", "global"],
        help="Network/profile defaults.",
    )
    p_bootstrap.add_argument(
        "--workflows",
        default=",".join(["cosyvoice", "rvc", "msst", "gpt_sovits"]),
        help="Comma-separated list: cosyvoice,rvc,msst,gpt_sovits",
    )
    p_bootstrap.add_argument(
        "--assets-dir",
        default=None,
        help="Assets cache dir (default: <repo>/.cache/voicelab/assets).",
    )
    p_bootstrap.add_argument(
        "--git-mirror-prefix",
        default=None,
        help="GitHub mirror prefix for vendor clone/pull.",
    )
    p_bootstrap.add_argument(
        "--hf-base",
        default=None,
        help="HuggingFace base URL (default for cn: https://hf-mirror.com).",
    )
    p_bootstrap.add_argument(
        "--dry-run", action="store_true", help="Print steps without executing."
    )
    p_bootstrap.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and recreate symlinks where safe.",
    )
    p_bootstrap.add_argument(
        "--vendor-force",
        action="store_true",
        help="Reset+clean vendor repos before pulling (destructive).",
    )
    p_bootstrap.add_argument(
        "--no-vendor", action="store_true", help="Skip vendor sync."
    )
    p_bootstrap.add_argument(
        "--no-env-sync",
        action="store_true",
        help="Skip uv python pin + uv sync for workflows.",
    )
    p_bootstrap.add_argument(
        "--no-assets", action="store_true", help="Skip downloading models/assets."
    )
    p_bootstrap.add_argument(
        "--no-runtime", action="store_true", help="Skip runtime initialization steps."
    )
    p_bootstrap.set_defaults(func=cmd_bootstrap)

    p_doctor = sub.add_parser(
        "doctor", help="Check system tools, vendors, envs, assets, and runtimes."
    )
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
