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
    ap = argparse.ArgumentParser(description="Train GPT (stage1) for GPT-SoVITS using prepared dataset under workflow data/.")
    ap.add_argument("--exp-name", required=True, help="Experiment name (data/<exp-name>).")
    ap.add_argument("--version", default="v2", help="Version tag (sets env 'version'). Default: v2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--save-every-epoch", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed", help="e.g. 16-mixed or 32")
    ap.add_argument("--gpu-numbers", default="0", help="Used as _CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--pretrained-s1", default="", help="Optional pretrained_s1 ckpt path.")
    ap.add_argument("--if-save-latest", type=int, default=1, choices=[0, 1])
    ap.add_argument("--if-save-every-weights", type=int, default=0, choices=[0, 1])
    ap.add_argument("--if-dpo", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    vendor = gpt_sovits_vendor_root()
    if not vendor.exists():
        raise SystemExit(f"[gpt_sovits] vendor not found: {vendor}")

    exp_name = args.exp_name.strip()
    opt_dir = (data_root() / exp_name).resolve()
    if not opt_dir.exists():
        raise SystemExit(f"[gpt_sovits] dataset not prepared: {opt_dir} (run gpt_sovits_prepare_dataset.py first)")

    # These are produced by prepare script (merged, not per-part).
    phoneme = opt_dir / "2-name2text.txt"
    semantic = opt_dir / "6-name2semantic.tsv"
    if not phoneme.exists() or not semantic.exists():
        raise SystemExit(f"[gpt_sovits] missing prepared files under {opt_dir} (need 2-name2text.txt + 6-name2semantic.tsv)")

    # Load vendor template then inject the keys that webui normally injects.
    # We import yaml lazily so repo-level unit tests don't require it.
    import yaml  # type: ignore

    tmpl = vendor / "GPT_SoVITS" / "configs" / ("s1longer.yaml" if args.version == "v1" else "s1longer-v2.yaml")
    data = yaml.safe_load(tmpl.read_text(encoding="utf-8"))

    # Outputs
    out_cfg_dir = (runtime_root() / exp_name / "configs").resolve()
    out_cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_cfg_dir / "s1.yaml"

    weights_dir = (runtime_root() / "weights" / "gpt").resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Update train section
    data.setdefault("train", {})
    data["train"]["batch_size"] = int(args.batch_size)
    data["train"]["epochs"] = int(args.epochs)
    data["train"]["save_every_n_epoch"] = int(args.save_every_epoch)
    data["train"]["precision"] = str(args.precision)
    data["train"]["if_save_latest"] = bool(int(args.if_save_latest))
    data["train"]["if_save_every_weights"] = bool(int(args.if_save_every_weights))
    data["train"]["if_dpo"] = bool(int(args.if_dpo))
    data["train"]["half_weights_save_dir"] = str(weights_dir)
    data["train"]["exp_name"] = exp_name

    if str(args.pretrained_s1).strip():
        data["pretrained_s1"] = str(Path(args.pretrained_s1).expanduser().resolve())

    # Dataset pointers
    data["train_semantic_path"] = str(semantic)
    data["train_phoneme_path"] = str(phoneme)
    data["output_dir"] = str(opt_dir / f"logs_s1_{args.version}")

    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # Also write a JSON mirror for quick inspection without yaml tooling.
    (out_cfg_dir / "s1.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "_CUDA_VISIBLE_DEVICES": str(args.gpu_numbers),
            "hz": "25hz",
            "version": str(args.version),
        }
    )
    _run(
        [sys.executable, str(vendor / "GPT_SoVITS" / "s1_train.py"), "--config_file", str(cfg_path)],
        cwd=vendor,
        env=env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

