from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _import_process_chain():
    root = Path(__file__).resolve().parents[1]
    tools = root / "tools"
    sys.path.insert(0, str(tools))
    import msst_process_chain  # type: ignore

    return msst_process_chain


@pytest.mark.skipif(os.environ.get("MSST_RUN_INTEGRATION") != "1", reason="set MSST_RUN_INTEGRATION=1 to run")
def test_full_chain_smoke(tmp_path: Path) -> None:
    m = _import_process_chain()

    # Require initialized runtime + models.
    try:
        m._assert_runtime_ready()  # noqa: SLF001 - intentionally using internal guard for smoke test
    except SystemExit as e:
        pytest.skip(str(e))

    import numpy as np
    import soundfile as sf

    # Very short stereo clip to keep the test bounded (still slow due to models).
    sr = 44100
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False, dtype=np.float32)
    tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    stereo = np.stack([tone, tone], axis=1)

    inp = tmp_path / "smoke.wav"
    sf.write(inp, stereo, sr, subtype="FLOAT")

    outs = m.process_one(
        input_path=inp,
        output_dir=tmp_path / "out",
        device_mode="cpu",
        gpu_ids=[0],
        output_format="wav",
        wav_bit_depth="FLOAT",
        preconvert=False,
    )

    assert outs.other.exists()
    assert outs.vocals.exists()
    assert outs.vocals_karaoke.exists()
    assert outs.vocals_other.exists()
    assert outs.vocals_karaoke_noreverb.exists()
    assert outs.vocals_karaoke_noreverb_dry.exists()

