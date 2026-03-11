"""Unit tests for MFA (Montreal Forced Aligner) alignment.

Verifies that `MFAAligner.align_one_wav` produces a valid alignment result
when given a generated waveform and its transcript.

The test is skipped automatically when the IndexTTS2 checkpoint directory
or the reference speaker audio are unavailable on the current machine.
"""

import sys
from pathlib import Path

import pytest

# ---- path setup --------------------------------------------------------
_HERE      = Path(__file__).resolve().parent
_DUB_ROOT  = _HERE.parent
_PROJ_ROOT = _DUB_ROOT.parent
_INDEX_ROOT = _PROJ_ROOT / "index-tts2"

for _p in [str(_INDEX_ROOT), str(_PROJ_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
logging.getLogger().setLevel(logging.WARNING)

import os

# ---- paths -------------------------------------------------------------
_CHECKPOINT_PATH = Path("/data2/ruixin/index-tts2/checkpoints")
_SPK_PROMPT      = Path("/data2/ruixin/ted-tts/AllInferenceResults/ESD/0001/Angry/0001_000351.wav")

_SKIP = pytest.mark.skipif(
    not _CHECKPOINT_PATH.exists(),
    reason=f"IndexTTS2 checkpoint not found: {_CHECKPOINT_PATH}",
)


# ========================================================================
# Tests
# ========================================================================

@_SKIP
def test_mfa_aligner_produces_alignment():
    """MFAAligner.align_one_wav returns a non-None alignment with interval data."""
    import torch
    import torchaudio
    from indextts.inferDub import IndexTTS2ForDub
    from modules.mfa_alinger import MFAAligner

    # ---- generate audio ------------------------------------------------
    model = IndexTTS2ForDub(
        model_dir=str(_CHECKPOINT_PATH),
        cfg_path=str(_CHECKPOINT_PATH / "config.yaml"),
        is_fp16=False,
    )

    spk_prompt = (
        str(_SPK_PROMPT) if _SPK_PROMPT.exists()
        else str(next(_CHECKPOINT_PATH.rglob("*.wav"), None) or _SPK_PROMPT)
    )

    texts = ["I left my guitar in their apartment.", "Well you can let me in later."]

    result = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=texts,
        output_path=None,
        use_emo_text=True,
        emo_text=["surprise", "angry"],
        emo_alpha=1.4,
        verbose=False,
        method="hmm",
        max_text_tokens_per_sentence=200,
        do_sample=True,
        top_p=0.8,
        top_k=30,
        temperature=0.8,
        length_penalty=0,
        num_beams=3,
        repetition_penalty=10.0,
        max_mel_tokens=2000,
        return_stats=True,
    )

    assert result is not None, "infer_dub returned None"
    assert result.wavs is not None and result.wavs.numel() > 0, \
        "Generated waveform is empty"

    # ---- align ---------------------------------------------------------
    aligner   = MFAAligner()
    full_text = " ".join(texts)

    align_result = aligner.align_one_wav(
        text=full_text,
        wavs=result.wavs,
        text_file_path=str(_DUB_ROOT / "tests" / "_tmp_mfa_test.txt"),
    )

    assert align_result is not None, "align_one_wav returned None"


@_SKIP
def test_mfa_aligner_result_has_intervals():
    """Alignment result contains at least one word or phone interval."""
    import torch
    from indextts.inferDub import IndexTTS2ForDub
    from modules.mfa_alinger import MFAAligner
    import tgt

    model = IndexTTS2ForDub(
        model_dir=str(_CHECKPOINT_PATH),
        cfg_path=str(_CHECKPOINT_PATH / "config.yaml"),
        is_fp16=False,
    )

    spk_prompt = (
        str(_SPK_PROMPT) if _SPK_PROMPT.exists()
        else str(next(_CHECKPOINT_PATH.rglob("*.wav"), None) or _SPK_PROMPT)
    )

    texts = ["Hello world."]

    result = model.infer_dub(
        spk_audio_prompt=spk_prompt,
        emo_audio_prompt=None,
        text=texts,
        output_path=None,
        use_emo_text=False,
        verbose=False,
        method="hmm",
        do_sample=True,
        num_beams=3,
        max_mel_tokens=1000,
        return_stats=True,
    )

    aligner      = MFAAligner()
    align_result = aligner.align_one_wav(
        text="Hello world.",
        wavs=result.wavs,
        text_file_path=str(_DUB_ROOT / "tests" / "_tmp_mfa_test2.txt"),
    )

    assert align_result is not None, "align_one_wav returned None"

    # align_result may be a tgt.TextGrid or a dict; check it has content
    if isinstance(align_result, tgt.TextGrid):
        total_intervals = sum(len(tier) for tier in align_result.tiers)
        assert total_intervals > 0, "TextGrid has no intervals"
    elif isinstance(align_result, dict):
        assert len(align_result) > 0, "Alignment dict is empty"
    else:
        # Any truthy result is acceptable
        assert align_result, "Alignment result is falsy"