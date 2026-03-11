"""Unit tests for IndexTTS2ForDub inference.

Verifies that `IndexTTS2ForDub.infer_dub` returns a result object with:
- a `wavs` tensor of shape (C, T) where C ≥ 1 and T > 0
- a `sampling_rate` equal to 22050

The test is skipped automatically when the checkpoint directory is unavailable.
"""

import sys
import os
from pathlib import Path

import pytest

# ---- path setup --------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
_DUB_ROOT = _HERE.parent
_PROJ_ROOT = _DUB_ROOT.parent
_INDEX_ROOT = _PROJ_ROOT / "index-tts2"

for _p in [str(_INDEX_ROOT), str(_PROJ_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
logging.getLogger().setLevel(logging.WARNING)

# ---- checkpoint path ---------------------------------------------------
_CHECKPOINT_PATH = Path("/data2/ruixin/index-tts2/checkpoints")
_SPK_PROMPT      = Path("/data2/ruixin/datasets/MELD_raw/audios/ost/dev_dia0_utt0.wav")

_SKIP = pytest.mark.skipif(
    not _CHECKPOINT_PATH.exists(),
    reason=f"IndexTTS2 checkpoint not found: {_CHECKPOINT_PATH}",
)


# ========================================================================
# Tests
# ========================================================================

@_SKIP
def test_infer_dub_returns_result():
    """Model loads and returns a non-empty InferDubResult."""
    from indextts.inferDub import IndexTTS2ForDub

    model = IndexTTS2ForDub(
        model_dir=str(_CHECKPOINT_PATH),
        cfg_path=str(_CHECKPOINT_PATH / "config.yaml"),
        is_fp16=False,
    )

    spk_prompt = (
        str(_SPK_PROMPT) if _SPK_PROMPT.exists()
        else str(next(_CHECKPOINT_PATH.rglob("*.wav"), None) or _SPK_PROMPT)
    )

    result = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=["I left my guitar in their apartment."],
        output_path=None,
        use_emo_text=True,
        emo_text=["surprise"],
        emo_alpha=1.2,
        verbose=False,
        method="hmm",
        max_text_tokens_per_sentence=200,
        do_sample=True,
        top_p=0.8,
        top_k=30,
        temperature=0.8,
        length_penalty=0,
        num_beams=1,
        repetition_penalty=10.0,
        max_mel_tokens=2000,
        return_stats=True,
    )

    assert result is not None, "infer_dub returned None"
    assert hasattr(result, "wavs"), "result has no 'wavs' attribute"
    assert result.wavs is not None, "result.wavs is None"
    assert result.wavs.ndim >= 1,  "result.wavs must be at least 1-D"
    assert result.wavs.numel() > 0, "result.wavs is empty"
    assert hasattr(result, "sampling_rate"), "result has no 'sampling_rate' attribute"
    assert result.sampling_rate == 22050, \
        f"Expected sampling_rate=22050, got {result.sampling_rate}"


@_SKIP
def test_infer_dub_multi_sentence():
    """Multi-sentence input produces a longer waveform than single-sentence."""
    import torch
    from indextts.inferDub import IndexTTS2ForDub

    model = IndexTTS2ForDub(
        model_dir=str(_CHECKPOINT_PATH),
        cfg_path=str(_CHECKPOINT_PATH / "config.yaml"),
        is_fp16=False,
    )

    spk_prompt = (
        str(_SPK_PROMPT) if _SPK_PROMPT.exists()
        else str(next(_CHECKPOINT_PATH.rglob("*.wav"), None) or _SPK_PROMPT)
    )

    single = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=["Hello."],
        output_path=None,
        use_emo_text=False,
        verbose=False,
        method="hmm",
        do_sample=False,
        num_beams=1,
        max_mel_tokens=1000,
        return_stats=True,
    )

    multi = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=["Hello.", "Well you can let me in later."],
        output_path=None,
        use_emo_text=False,
        verbose=False,
        method="hmm",
        do_sample=False,
        num_beams=1,
        max_mel_tokens=2000,
        return_stats=True,
    )

    assert multi.wavs.numel() > single.wavs.numel(), \
        "Multi-sentence output should be longer than single-sentence output"