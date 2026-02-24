from __future__ import annotations

import sys
from pathlib import Path


def _import_process_chain():
    # Allow importing workflows/msst/tools/msst_process_chain.py from tests.
    root = Path(__file__).resolve().parents[1]
    tools = root / "tools"
    sys.path.insert(0, str(tools))
    import msst_process_chain  # type: ignore

    return msst_process_chain


def test_expected_outputs_naming() -> None:
    m = _import_process_chain()
    outdir = Path("/tmp/out")
    outs = m.expected_outputs(base_stem="song", output_dir=outdir)
    assert outs.other.name == "song_other.wav"
    assert outs.vocals.name == "song_vocals.wav"
    assert outs.vocals_karaoke.name == "song_vocals_karaoke.wav"
    assert outs.vocals_other.name == "song_vocals_other.wav"
    assert outs.vocals_karaoke_noreverb.name == "song_vocals_karaoke_noreverb.wav"
    assert outs.vocals_karaoke_noreverb_dry.name == "song_vocals_karaoke_noreverb_dry.wav"

