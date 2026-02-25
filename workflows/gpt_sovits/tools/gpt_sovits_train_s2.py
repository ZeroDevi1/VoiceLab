from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from voicelab_bootstrap import data_root, gpt_sovits_vendor_root, runtime_root


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[gpt_sovits] $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train SoVITS (stage2) for GPT-SoVITS using prepared dataset under workflow data/.")
    ap.add_argument("--exp-name", required=True, help="Experiment name (data/<exp-name>).")
    ap.add_argument("--version", default="v2", help="Model version (sets env 'version' + config model.version).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save-every-epoch", type=int, default=1)
    ap.add_argument("--text-low-lr-rate", type=float, default=0.4)
    ap.add_argument("--gpu-numbers", default="0", help="gpu_numbers string written to config train.gpu_numbers.")
    ap.add_argument("--grad-ckpt", type=int, default=0, choices=[0, 1])
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--fp16-run", type=int, default=1, choices=[0, 1])
    ap.add_argument("--pretrained-s2g", default="", help="Optional pretrained_s2G path.")
    ap.add_argument("--pretrained-s2d", default="", help="Optional pretrained_s2D path.")
    ap.add_argument("--if-save-latest", type=int, default=1, choices=[0, 1])
    ap.add_argument("--if-save-every-weights", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    vendor = gpt_sovits_vendor_root()
    if not vendor.exists():
        raise SystemExit(f"[gpt_sovits] vendor not found: {vendor}")

    exp_name = args.exp_name.strip()
    opt_dir = (data_root() / exp_name).resolve()
    if not opt_dir.exists():
        raise SystemExit(f"[gpt_sovits] dataset not prepared: {opt_dir} (run gpt_sovits_prepare_dataset.py first)")

    required = [
        opt_dir / "2-name2text.txt",
        opt_dir / "4-cnhubert",
        opt_dir / "5-wav32k",
        opt_dir / "6-name2semantic.tsv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit("[gpt_sovits] missing prepared dataset artifacts:\n" + "\n".join(f"  - {p}" for p in missing))

    # Base template depends on version family.
    tmpl = vendor / "GPT_SoVITS" / "configs" / (
        "s2.json" if args.version not in {"v2Pro", "v2ProPlus"} else f"s2{args.version}.json"
    )
    data = json.loads(tmpl.read_text(encoding="utf-8"))

    out_cfg_dir = (runtime_root() / exp_name / "configs").resolve()
    out_cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_cfg_dir / "s2.json"

    weights_dir = (runtime_root() / "weights" / "sovits").resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Mutate config like vendor webui does.
    data["train"]["batch_size"] = int(args.batch_size)
    data["train"]["epochs"] = int(args.epochs)
    data["train"]["text_low_lr_rate"] = float(args.text_low_lr_rate)
    data["train"]["fp16_run"] = bool(int(args.fp16_run))
    data["train"]["if_save_latest"] = bool(int(args.if_save_latest))
    data["train"]["if_save_every_weights"] = bool(int(args.if_save_every_weights))
    data["train"]["save_every_epoch"] = int(args.save_every_epoch)
    data["train"]["gpu_numbers"] = str(args.gpu_numbers)
    data["train"]["grad_ckpt"] = bool(int(args.grad_ckpt))
    data["train"]["lora_rank"] = int(args.lora_rank)

    if str(args.pretrained_s2g).strip():
        data["train"]["pretrained_s2G"] = str(Path(args.pretrained_s2g).expanduser().resolve())
    if str(args.pretrained_s2d).strip():
        data["train"]["pretrained_s2D"] = str(Path(args.pretrained_s2d).expanduser().resolve())

    data["model"]["version"] = str(args.version)
    # Put checkpoints next to the prepared dataset, matching vendor behavior.
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(opt_dir)
    data["save_weight_dir"] = str(weights_dir)
    data["name"] = exp_name
    data["version"] = str(args.version)

    cfg_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    env = os.environ.copy()
    env.update({"version": str(args.version), "_CUDA_VISIBLE_DEVICES": str(args.gpu_numbers)})

    train_script = vendor / "GPT_SoVITS" / ("s2_train.py" if args.version in {"v1", "v2", "v2Pro", "v2ProPlus"} else "s2_train_v3_lora.py")
    _run([sys.executable, str(train_script), "--config", str(cfg_path)], cwd=vendor, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

