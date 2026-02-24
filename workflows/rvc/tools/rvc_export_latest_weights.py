from __future__ import annotations

import argparse
import os
from pathlib import Path

from rvc_init_runtime import init_runtime
from voicelab_bootstrap import ensure_runtime_pythonpath


def _pick_latest_checkpoint(exp_dir: Path) -> Path:
    cands = [p for p in exp_dir.glob("G_*.pth") if p.is_file()]
    if not cands:
        raise SystemExit(
            f"No generator checkpoints found under: {exp_dir}\n"
            "If you haven't reached the first checkpoint yet, wait for an epoch save.\n"
            "Tip: reduce --save-every-epoch (e.g. 1/2/5) so checkpoints appear sooner."
        )
    # Prefer mtime so it works even when `--if-latest 1` keeps overwriting G_2333333.pth.
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export an inference-ready weight under runtime/assets/weights/ from the latest saved RVC checkpoint.\n"
            "Useful when you Ctrl+C training: it converts the latest saved epoch checkpoint into <exp-name>.pth."
        )
    )
    ap.add_argument("--exp-name", required=True, help="Experiment name under runtime/logs/<exp-name>/")
    ap.add_argument(
        "--out-name",
        default=None,
        help="Output weight filename (without .pth). Default: <exp-name> (overwrite/refresh <exp-name>.pth).",
    )
    ap.add_argument(
        "--also-snapshot",
        action="store_true",
        help="Also write a snapshot weight with epoch suffix: <exp-name>_e<epoch>_export.pth",
    )
    ap.add_argument("--sr-tag", default="48k", choices=["48k", "40k", "32k"])
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--if-f0", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    rt = init_runtime(force=False, assets_src=None)
    ensure_runtime_pythonpath()
    os.chdir(rt)

    exp_dir = rt / "logs" / args.exp_name
    if not exp_dir.exists():
        raise SystemExit(f"Experiment dir not found: {exp_dir}\nRun training first.")

    ckpt_path = _pick_latest_checkpoint(exp_dir)

    try:
        import torch  # type: ignore

        from infer.lib.train import utils  # type: ignore
        from infer.lib.train.process_ckpt import savee  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"Missing dependencies for exporting weights: {e}\n"
            "Run: cd workflows/rvc && uv sync"
        )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    epoch = int(ckpt.get("iteration", 0))
    state = ckpt.get("model", ckpt)

    # We only need config/model fields for savee(); config.json lives in exp_dir.
    hps = utils.get_hparams_from_dir(str(exp_dir))

    out_name = str(args.out_name) if args.out_name else str(args.exp_name)
    r = savee(state, str(args.sr_tag), int(args.if_f0), out_name, epoch, str(args.version), hps)
    if "Success" not in str(r):
        raise SystemExit(f"Export failed: {r}")

    if args.also_snapshot:
        snap = f"{args.exp_name}_e{epoch}_export"
        r2 = savee(state, str(args.sr_tag), int(args.if_f0), snap, epoch, str(args.version), hps)
        if "Success" not in str(r2):
            raise SystemExit(f"Snapshot export failed: {r2}")
        print(f"[rvc] wrote snapshot: {rt / 'assets' / 'weights' / f'{snap}.pth'}")

    print(f"[rvc] exported from checkpoint: {ckpt_path}")
    print(f"[rvc] wrote weight: {rt / 'assets' / 'weights' / f'{out_name}.pth'} (epoch={epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

