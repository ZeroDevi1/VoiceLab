from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from random import shuffle

from rvc_init_runtime import init_runtime
from rvc_stage_dataset import stage_dataset
from voicelab_bootstrap import voicelab_root


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    # Keep stdout/stderr attached for long-running training.
    print("[rvc] $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _exp_dir(rt: Path, exp_name: str) -> Path:
    return rt / "logs" / exp_name


def _write_config_json(rt: Path, exp_name: str, *, version: str, sr_tag: str) -> None:
    exp = _exp_dir(rt, exp_name)
    exp.mkdir(parents=True, exist_ok=True)

    cfg_src = rt / "configs" / version / f"{sr_tag}.json"
    if not cfg_src.exists():
        raise SystemExit(f"Missing config template: {cfg_src} (did you run tools/rvc_init_runtime.py?)")

    cfg_dst = exp / "config.json"
    if cfg_dst.exists():
        return

    # Normalize via JSON load/dump to avoid accidental BOM/encoding issues.
    data = json.loads(cfg_src.read_text(encoding="utf-8"))
    cfg_dst.write_text(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True) + "\n", encoding="utf-8")


def _write_filelist(
    rt: Path,
    exp_name: str,
    *,
    version: str,
    if_f0: bool,
    sr_tag: str,
    spk_id: int = 0,
) -> None:
    exp = _exp_dir(rt, exp_name)
    gt_wavs_dir = exp / "0_gt_wavs"
    fea_dir = exp / ("3_feature256" if version == "v1" else "3_feature768")
    if if_f0:
        f0_dir = exp / "2a_f0"
        f0nsf_dir = exp / "2b-f0nsf"

    if not gt_wavs_dir.exists():
        raise SystemExit(f"Missing preprocess output: {gt_wavs_dir}")
    if not fea_dir.exists():
        raise SystemExit(f"Missing feature output: {fea_dir}")

    # In upstream, f0 npy names are like "xxx.wav.npy" (stem "xxx.wav").
    gt = {x.stem for x in gt_wavs_dir.glob("*.wav")}
    fea = {x.name.replace(".npy", "").replace(".wav", "") for x in fea_dir.glob("*.npy")}
    if if_f0:
        f0 = {x.name.replace(".npy", "").replace(".wav", "") for x in f0_dir.glob("*.npy")}
        f0nsf = {x.name.replace(".npy", "").replace(".wav", "") for x in f0nsf_dir.glob("*.npy")}
        names = gt & fea & f0 & f0nsf
    else:
        names = gt & fea

    if not names:
        raise SystemExit("No aligned training items found (gt/features/f0 mismatch).")

    lines: list[str] = []
    exp_abs = exp.resolve()
    gt_abs = (exp_abs / "0_gt_wavs").as_posix()
    fea_abs = (exp_abs / ("3_feature256" if version == "v1" else "3_feature768")).as_posix()
    if if_f0:
        f0_abs = (exp_abs / "2a_f0").as_posix()
        f0nsf_abs = (exp_abs / "2b-f0nsf").as_posix()

    for name in sorted(names):
        if if_f0:
            lines.append(
                f"{gt_abs}/{name}.wav|{fea_abs}/{name}.npy|{f0_abs}/{name}.wav.npy|{f0nsf_abs}/{name}.wav.npy|{spk_id}"
            )
        else:
            lines.append(f"{gt_abs}/{name}.wav|{fea_abs}/{name}.npy|{spk_id}")

    # Append 2 mute samples, matching upstream behavior.
    mute_root = (rt / "logs" / "mute").resolve()
    fea_dim = "256" if version == "v1" else "768"
    if if_f0:
        for _ in range(2):
            lines.append(
                f"{mute_root.as_posix()}/0_gt_wavs/mute{sr_tag}.wav|"
                f"{mute_root.as_posix()}/3_feature{fea_dim}/mute.npy|"
                f"{mute_root.as_posix()}/2a_f0/mute.wav.npy|"
                f"{mute_root.as_posix()}/2b-f0nsf/mute.wav.npy|{spk_id}"
            )
    else:
        for _ in range(2):
            lines.append(
                f"{mute_root.as_posix()}/0_gt_wavs/mute{sr_tag}.wav|"
                f"{mute_root.as_posix()}/3_feature{fea_dim}/mute.npy|{spk_id}"
            )

    shuffle(lines)
    (exp / "filelist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Note: if `logs/mute` isn't present in runtime, training still works, but upstream expects it.


def main() -> int:
    ap = argparse.ArgumentParser(description="Train RVC model for XingTong dataset (v2 + 48k + f0).")
    ap.add_argument("--exp-name", default="xingtong_v2_48k_f0")
    ap.add_argument(
        "--dataset-dir",
        default=str(voicelab_root() / "datasets" / "XingTong"),
        help="Prefer a WSL-native ext4 path (default: VoiceLab/datasets/XingTong).",
    )
    ap.add_argument(
        "--stage-dataset",
        action="store_true",
        help="If dataset is on /mnt/*, copy it into WSL native ext4 path before preprocessing (faster I/O).",
    )
    ap.add_argument(
        "--dataset-wsl-dir",
        default=None,
        help="Destination directory for --stage-dataset (default: VoiceLab/datasets/XingTong).",
    )
    ap.add_argument("--sr", type=int, default=48000, help="Preprocess sample rate (Hz).")
    ap.add_argument("--sr-tag", default="48k", choices=["48k", "40k", "32k"], help="RVC config tag for training.")
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--if-f0", action="store_true", default=True)
    ap.add_argument("--preprocess-np", type=int, default=8, help="Preprocess/extract parallelism.")
    ap.add_argument("--per", type=float, default=3.7, help="Preprocess chunk length (seconds).")

    ap.add_argument("--gpu", default="0", help="GPU id string used by upstream scripts.")
    ap.add_argument("--batch-size", type=int, default=4)
    # Based on the user's observed ~10-13 min/epoch on RTX 3060 Laptop 6GB,
    # saving every ~3 epochs gives a 30-40 min checkpoint cadence.
    ap.add_argument("--save-every-epoch", type=int, default=3)
    ap.add_argument("--total-epoch", type=int, default=50)
    ap.add_argument("--if-latest", type=int, default=1, choices=[0, 1])
    ap.add_argument("--if-cache-gpu", type=int, default=0, choices=[0, 1])
    ap.add_argument("--save-every-weights", type=int, default=0, choices=[0, 1])

    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-f0", action="store_true")
    ap.add_argument("--skip-feature", action="store_true")
    ap.add_argument("--skip-filelist", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument(
        "--quiet-warnings",
        action="store_true",
        help="Suppress noisy FutureWarning/UserWarning from upstream scripts (does not affect training).",
    )

    args = ap.parse_args()

    # Ensure runtime exists and is correctly wired.
    rt = init_runtime(force=False, assets_src=None)
    run_env = os.environ.copy()
    if args.quiet_warnings:
        # Keep stderr clean; upstream prints lots of harmless deprecation warnings on newer torch.
        run_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    dataset_dir = Path(args.dataset_dir).expanduser()
    # Only validate/copy dataset when we actually need to preprocess.
    if not args.skip_preprocess:
        if args.stage_dataset:
            # Only impacts the preprocess step. After preprocess, all training I/O happens in runtime/logs on ext4.
            dst = (
                Path(args.dataset_wsl_dir).expanduser()
                if args.dataset_wsl_dir
                else (voicelab_root() / "datasets" / "XingTong")
            )
            dataset_dir = stage_dataset(dataset_dir, dst, force=False)
        if not dataset_dir.exists():
            raise SystemExit(
                f"Dataset dir not found: {dataset_dir}\n"
                "If your dataset is on Windows (/mnt/c/...), run:\n"
                "  uv run python tools/rvc_stage_dataset.py --src /mnt/c/AIGC/数据集/XingTong\n"
                "then rerun with --dataset-dir pointing to VoiceLab/datasets/XingTong."
            )

    exp = _exp_dir(rt, args.exp_name)
    exp_rel = Path("logs") / args.exp_name  # what upstream scripts expect

    # Step 1: preprocess dataset
    if not args.skip_preprocess:
        # upstream preprocess.py opens exp_dir/preprocess.log before mkdir, so ensure it exists.
        (rt / exp_rel).mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "infer/modules/train/preprocess.py",
                str(dataset_dir),
                str(int(args.sr)),
                str(int(args.preprocess_np)),
                str(exp_rel),
                "False",  # noparallel
                str(float(args.per)),
            ],
            cwd=rt,
            env=run_env,
        )

    # Step 2: extract f0 (RMVPE GPU)
    if args.if_f0 and not args.skip_f0:
        _run(
            [
                sys.executable,
                "infer/modules/train/extract/extract_f0_rmvpe.py",
                "1",  # n_part
                "0",  # i_part
                str(args.gpu),
                str(exp_rel),
                "True",  # is_half
            ],
            cwd=rt,
            env=run_env,
        )

    # Step 3: extract hubert features
    if not args.skip_feature:
        _run(
            [
                sys.executable,
                "infer/modules/train/extract_feature_print.py",
                "cuda:0",
                "1",  # n_part
                "0",  # i_part
                str(args.gpu),
                str(exp_rel),
                args.version,
                "True",  # is_half
            ],
            cwd=rt,
            env=run_env,
        )

    # Step 4: config + filelist
    if not args.skip_filelist:
        _write_config_json(rt, args.exp_name, version=args.version, sr_tag=args.sr_tag)
        _write_filelist(
            rt,
            args.exp_name,
            version=args.version,
            if_f0=bool(args.if_f0),
            sr_tag=args.sr_tag,
            spk_id=0,
        )

    # Step 5: train (auto-resume supported by upstream train.py)
    if not args.skip_train:
        pre_g = rt / "assets" / ("pretrained_v2" if args.version == "v2" else "pretrained") / f"f0G{args.sr_tag}.pth"
        pre_d = rt / "assets" / ("pretrained_v2" if args.version == "v2" else "pretrained") / f"f0D{args.sr_tag}.pth"
        _run(
            [
                sys.executable,
                "infer/modules/train/train.py",
                "-e",
                args.exp_name,
                "-sr",
                args.sr_tag,
                "-f0",
                "1" if args.if_f0 else "0",
                "-bs",
                str(int(args.batch_size)),
                "-g",
                str(args.gpu),
                "-te",
                str(int(args.total_epoch)),
                "-se",
                str(int(args.save_every_epoch)),
                "-pg",
                str(pre_g) if pre_g.exists() else "",
                "-pd",
                str(pre_d) if pre_d.exists() else "",
                "-l",
                str(int(args.if_latest)),
                "-c",
                str(int(args.if_cache_gpu)),
                "-sw",
                str(int(args.save_every_weights)),
                "-v",
                args.version,
            ],
            cwd=rt,
            env=run_env,
        )

    model_pth = rt / "assets" / "weights" / f"{args.exp_name}.pth"
    print("")
    if model_pth.exists():
        print(f"[rvc] model ready: {model_pth}")
    else:
        print(f"[rvc] NOTE: model not found yet (expected after training): {model_pth}")
    print(f"[rvc] exp dir: {exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
